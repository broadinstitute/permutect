import time

import cyvcf2
import numpy as np
import pytest
import torch
from torch import Tensor

from permutect import misc_utils
from permutect.architecture.posterior_model import PosteriorModel
from permutect.architecture.posterior_model_priors import PosteriorModelPriors
from permutect.architecture.spectra.posterior_model_spectra import PosteriorModelSpectra
from permutect.data.batch import BatchIndexedTensor
from permutect.data.datum import DEFAULT_CPU_FLOAT
from permutect.data.datum import DEFAULT_GPU_FLOAT
from permutect.data.datum import RAW_READS_ARRAY_DTYPE
from permutect.data.datum import Datum
from permutect.metrics.evaluation_metrics import EvaluationMetrics
from permutect.metrics.loss_metrics import AccuracyMetrics
from permutect.metrics.loss_metrics import LossMetrics
from permutect.misc_utils import ConsistentValue
from permutect.misc_utils import StreamingAverage
from permutect.misc_utils import Timer
from permutect.misc_utils import encode
from permutect.misc_utils import encode_datum
from permutect.misc_utils import encode_variant
from permutect.misc_utils import float_dtype
from permutect.misc_utils import get_first_numeric_element
from permutect.misc_utils import overlapping_filters
from permutect.misc_utils import resolve_device
from permutect.utils.enums import Epoch
from permutect.utils.enums import Label
from permutect.utils.enums import Variation

VCF_HEADER = (
    "##fileformat=VCFv4.2\n"
    "##contig=<ID=chr1,length=248956422>\n"
    "##contig=<ID=chrX,length=156040895>\n"
    '##FILTER=<ID=PASS,Description="All filters passed">\n'
    '##FILTER=<ID=weak_evidence,Description="Weak evidence">\n'
    '##FILTER=<ID=germline,Description="Germline">\n'
    '##FILTER=<ID=normal_artifact,Description="Normal artifact">\n'
    '##INFO=<ID=POPAF,Number=A,Type=Float,Description="Negative log10 population allele frequency">\n'
    '##INFO=<ID=DP,Number=1,Type=Integer,Description="Depth">\n'
    "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n"
)


def vcf_line(ref: str, alt: str, pos: int = 100, chrom: str = "chr1", filters: str = ".", info: str = ".") -> str:
    return "\t".join([chrom, str(pos), ".", ref, alt, ".", filters, info])


def read_variants(tmp_path, *lines: str) -> list[cyvcf2.Variant]:
    path = tmp_path / "variants.vcf"
    path.write_text(VCF_HEADER + "".join(line + "\n" for line in lines))
    return list(cyvcf2.VCF(str(path)))


def make_datum(contig: int, position: int, ref: str, alt: str) -> Datum:
    # the ref allele goes in the middle of an odd-length reference context, as in GATK data
    flank = "GATCCTAGGATCCTAGGATC"
    ref_sequence = flank + ref + flank[: len(flank) + 1 - len(ref)]
    return Datum.from_gatk(
        label=Label.UNLABELED,
        variant_type=Variation.get_type(ref, alt),
        source=0,
        original_depth=10,
        original_alt_count=3,
        original_normal_depth=0,
        original_normal_alt_count=0,
        contig=contig,
        position=position,
        ref_allele=ref,
        alt_allele=alt,
        seq_error_log_lk=0.0,
        normal_seq_error_log_lk=0.0,
        ref_sequence_string=ref_sequence,
        gatk_info_array=np.zeros(3),
        ref_reads_array_re=None,
        alt_reads_array_re=np.zeros((3, 5), dtype=RAW_READS_ARRAY_DTYPE),
    )


def test_consistent_value():
    cv = ConsistentValue()
    assert cv.value is None

    # the first check sets the value; later checks must match it
    cv.check(5)
    assert cv.value == 5
    cv.check(5)
    cv.check(5)
    assert cv.value == 5

    with pytest.raises(AssertionError):
        cv.check(6)


def test_consistent_value_with_initial_value():
    cv = ConsistentValue(3)
    cv.check(3)
    with pytest.raises(AssertionError):
        cv.check(4)


def test_consistent_value_edge_cases():
    # zero is a real value, not "uninitialized"
    cv = ConsistentValue()
    cv.check(0)
    assert cv.value == 0
    with pytest.raises(AssertionError):
        cv.check(1)

    # a failed check doesn't change the stored value
    cv = ConsistentValue()
    cv.check(7)
    with pytest.raises(AssertionError):
        cv.check(8)
    assert cv.value == 7

    # non-integer values that support ==
    cv = ConsistentValue()
    cv.check((2, 3))
    cv.check((2, 3))
    with pytest.raises(AssertionError):
        cv.check((3, 2))


def test_streaming_average_record():
    avg = StreamingAverage()
    for value in [1.0, 2.0, 3.0, 4.0]:
        avg.record(value)
    assert not avg.is_empty()
    assert avg.get() == pytest.approx(2.5, rel=1e-3)


def test_streaming_average_record_with_weights():
    avg = StreamingAverage()
    avg.record(1.0, weight=3)
    avg.record(5.0, weight=1)
    assert avg.get() == pytest.approx((3 * 1.0 + 5.0) / 4, rel=1e-3)


def test_streaming_average_record_sum():
    # this is how the posterior model records the loss of each batch
    avg = StreamingAverage()
    avg.record_sum(value_sum=10.0, count=4)
    avg.record_sum(value_sum=20.0, count=6)
    assert avg.get() == pytest.approx(30.0 / 10, rel=1e-3)


def test_streaming_average_record_with_mask():
    avg = StreamingAverage()
    values = Tensor([1, 2, 3, 100])
    avg.record_with_mask(values, torch.tensor([True, True, True, False]))
    assert avg.get() == pytest.approx(2.0, rel=1e-3)

    # a float mask of zeros and ones behaves the same as a boolean mask
    avg = StreamingAverage()
    avg.record_with_mask(values, Tensor([1, 1, 1, 0]))
    assert avg.get() == pytest.approx(2.0, rel=1e-3)


def test_streaming_average_record_with_tensor_weights():
    avg = StreamingAverage()
    avg.record_with_weights(Tensor([1, 5]), Tensor([3, 1]))
    assert avg.get() == pytest.approx(2.0, rel=1e-3)


def test_streaming_average_combines_recording_methods():
    avg = StreamingAverage()
    avg.record(1.0)  # sum 1, count 1
    avg.record(2.0, weight=2)  # sum 5, count 3
    avg.record_sum(value_sum=10.0, count=2)  # sum 15, count 5
    avg.record_with_mask(Tensor([3, 100]), torch.tensor([True, False]))  # sum 18, count 6
    avg.record_with_weights(Tensor([6]), Tensor([2]))  # sum 30, count 8
    assert avg.get() == pytest.approx(30.0 / 8, rel=1e-3)


def test_streaming_average_matches_mean_of_many_values():
    torch.manual_seed(0)
    values = torch.randn(10000)
    avg = StreamingAverage()
    for value in values.tolist():
        avg.record(value)
    assert avg.get() == pytest.approx(values.mean().item(), rel=1e-3, abs=1e-6)


def test_streaming_average_edge_cases():
    # empty: no division by zero
    avg = StreamingAverage()
    assert avg.is_empty()
    assert avg.get() == 0.0

    # recording a value of zero makes it non-empty
    avg = StreamingAverage()
    avg.record(0.0)
    assert not avg.is_empty()
    assert avg.get() == 0.0

    # zero weight records nothing
    avg = StreamingAverage()
    avg.record(5.0, weight=0)
    assert avg.is_empty()

    # an all-false mask records nothing
    avg = StreamingAverage()
    avg.record_with_mask(Tensor([1, 2, 3]), torch.tensor([False, False, False]))
    assert avg.is_empty()

    # empty tensors record nothing
    avg = StreamingAverage()
    avg.record_with_weights(Tensor([]), Tensor([]))
    assert avg.is_empty()

    # negative values
    avg = StreamingAverage()
    avg.record(-2.0)
    avg.record(-4.0)
    assert avg.get() == pytest.approx(-3.0, rel=1e-3)


def test_timer_prints_message_on_creation(capsys):
    Timer("Doing something. . .")
    assert capsys.readouterr().out == "Doing something. . .\n"

    Timer()
    assert capsys.readouterr().out == ""


def test_timer_reports_elapsed_time(monkeypatch, capsys):
    clock = iter([10.0, 11.5, 13.25])
    monkeypatch.setattr(time, "perf_counter", lambda: next(clock))

    timer = Timer()
    timer.report("first")
    timer.report("second")

    # each report measures from when the timer was created, not from the previous report
    assert capsys.readouterr().out.splitlines() == ["first: 1.5000 seconds", "second: 3.2500 seconds"]


def test_timer_with_real_clock(capsys):
    timer = Timer()
    timer.report("elapsed")
    output = capsys.readouterr().out.strip()
    assert output.startswith("elapsed: ")
    assert output.endswith(" seconds")
    elapsed = float(output.removeprefix("elapsed: ").removesuffix(" seconds"))
    assert 0 <= elapsed < 1


def test_get_first_numeric_element(tmp_path):
    biallelic, multiallelic = read_variants(
        tmp_path,
        vcf_line("A", "T", info="POPAF=3.5;DP=20"),
        vcf_line("G", "C,T", pos=200, info="POPAF=1.5,4.0"),
    )
    # cyvcf2 returns a single value as a scalar, even for a Number=A field
    assert get_first_numeric_element(biallelic, "DP") == 20
    assert get_first_numeric_element(biallelic, "POPAF") == 3.5

    # cyvcf2 returns multiple values as a tuple
    assert get_first_numeric_element(multiallelic, "POPAF") == 1.5


def test_get_first_numeric_element_edge_cases(tmp_path):
    zero, missing = read_variants(
        tmp_path,
        vcf_line("A", "T", info="POPAF=0"),
        vcf_line("A", "T", pos=200),
    )
    # zero is a real value, not a missing one
    assert get_first_numeric_element(zero, "POPAF") == 0

    # a missing INFO field raises an error rather than returning a default
    with pytest.raises(KeyError):
        get_first_numeric_element(missing, "POPAF")


def test_encode():
    assert encode("chr1", 100, "A", "T") == "chr1:100:A:T"
    assert encode("chr1", 100, "ACG", "A") == "chr1:100:ACG:A"
    assert encode("chr1", 100, "A", "AGT") == "chr1:100:A:AGT"
    assert encode("chrX", 123456789, "C", "G") == "chrX:123456789:C:G"


def test_encode_distinguishes_ref_alleles():
    # different variants at one position with the same alt allele, eg from a split multiallelic site
    assert encode("1", 100, "AT", "A") != encode("1", 100, "ATT", "A")
    assert encode("1", 100, "A", "AT") != encode("1", 100, "AC", "AT")


def test_encode_trims_alleles():
    # non-minimal representations encode the same as minimal ones
    assert encode("chr1", 100, "CGG", "TGG") == encode("chr1", 100, "C", "T")
    assert encode("chr1", 100, "ACGT", "AT") == encode("chr1", 100, "ACG", "A")
    assert encode("chr1", 100, "AT", "ACGT") == encode("chr1", 100, "A", "ACG")

    # trimming stops at one base, so the anchor base of an indel is kept
    assert encode("chr1", 100, "CTT", "CT") == "chr1:100:CT:C"


def test_encode_truncates_long_alleles():
    # alleles of 13 bases are kept whole, and longer ones are truncated to 13 bases
    insertion_13 = "A" + "C" * 12
    assert encode("chr1", 100, "A", insertion_13) == "chr1:100:A:" + insertion_13
    assert encode("chr1", 100, "A", insertion_13 + "G") == "chr1:100:A:" + insertion_13

    long_insertion = "A" + "CGT" * 10
    assert encode("chr1", 100, "A", long_insertion) == "chr1:100:A:" + long_insertion[:13]
    assert encode("chr1", 100, long_insertion, "A") == "chr1:100:" + long_insertion[:13] + ":A"

    # trimming happens before truncation.  Truncating first would give CAAAAAAAAAAAG, which trimming against the
    # ref allele CG would then shorten to CAAAAAAAAAAA
    assert encode("chr1", 100, "CG", "C" + "A" * 11 + "GTG") == "chr1:100:C:C" + "A" * 11 + "G"


def test_encode_datum():
    contig_names = {0: "chr1", 1: "chrX"}
    assert encode_datum(make_datum(0, 100, "C", "T"), contig_names) == "chr1:100:C:T"
    assert encode_datum(make_datum(1, 100, "C", "T"), contig_names) == "chrX:100:C:T"
    assert encode_datum(make_datum(0, 200, "CAG", "C"), contig_names) == "chr1:200:CAG:C"
    assert encode_datum(make_datum(0, 300, "C", "CTT"), contig_names) == "chr1:300:C:CTT"


def test_encode_datum_edge_cases():
    contig_names = {0: "chr1"}

    # the datum stores positions as two 16-bit integers, so check positions that need the high bits
    assert encode_datum(make_datum(0, 2**16 + 1, "C", "T"), contig_names) == f"chr1:{2**16 + 1}:C:T"
    assert encode_datum(make_datum(0, 4_000_000_000, "C", "T"), contig_names) == "chr1:4000000000:C:T"

    # non-minimal and long alleles
    assert encode_datum(make_datum(0, 100, "CGG", "TGG"), contig_names) == "chr1:100:C:T"
    long_insertion = "C" + "AGT" * 6
    assert encode_datum(make_datum(0, 100, "C", long_insertion), contig_names) == "chr1:100:C:" + long_insertion[:13]

    # a contig index missing from the map
    with pytest.raises(KeyError):
        encode_datum(make_datum(5, 100, "C", "T"), contig_names)


def test_encode_variant(tmp_path):
    snv, deletion, insertion = read_variants(
        tmp_path,
        vcf_line("A", "T", pos=100),
        vcf_line("ACG", "A", pos=200),
        vcf_line("C", "CTT", pos=300, chrom="chrX"),
    )
    # cyvcf2's start is zero-based, and zero_based=True converts it back to the one-based VCF position
    assert encode_variant(snv, zero_based=True) == "chr1:100:A:T"
    assert encode_variant(deletion, zero_based=True) == "chr1:200:ACG:A"
    assert encode_variant(insertion, zero_based=True) == "chrX:300:C:CTT"

    # without zero_based=True, the zero-based start is used as is
    assert encode_variant(snv) == "chr1:99:A:T"


def test_encode_variant_edge_cases(tmp_path):
    long_insertion = "A" + "CGT" * 10
    multiallelic, non_minimal, long, first_base = read_variants(
        tmp_path,
        vcf_line("G", "C,T", pos=100),
        vcf_line("CGG", "TGG", pos=200),
        vcf_line("A", long_insertion, pos=300),
        vcf_line("A", "T", pos=1),
    )
    # only the first alt allele is encoded
    assert encode_variant(multiallelic, zero_based=True) == "chr1:100:G:C"

    # alleles are trimmed and truncated
    assert encode_variant(non_minimal, zero_based=True) == "chr1:200:C:T"
    assert encode_variant(long, zero_based=True) == "chr1:300:A:" + long_insertion[:13]

    # the first base of a contig
    assert encode_variant(first_base, zero_based=True) == "chr1:1:A:T"
    assert encode_variant(first_base) == "chr1:0:A:T"


def test_vcf_and_dataset_encodings_agree(tmp_path):
    # filter_variants and memory_mapped_data match dataset variants to VCF variants by encoding, so these must agree
    variants = [
        ("chr1", 100, "C", "T"),
        ("chr1", 200, "CAG", "C"),
        ("chr1", 300, "C", "CTT"),
        ("chr1", 400, "CGG", "TGG"),
        ("chr1", 500, "C", "C" + "AGT" * 6),
        ("chr1", 600, "CG", "C" + "A" * 11 + "GTG"),  # the datum must also trim before truncating
        ("chr1", 700, "CAG", "C"),  # two deletions at one position, eg from a split multiallelic site
        ("chr1", 700, "CA", "C"),
        ("chr1", 800, "C" + "AGT" * 6, "C"),  # the datum truncates the ref allele of a long deletion
        ("chrX", 150_000_000, "C", "T"),
    ]
    contig_names = {0: "chr1", 1: "chrX"}
    contig_indices = {name: index for index, name in contig_names.items()}

    vcf_variants = read_variants(
        tmp_path, *[vcf_line(ref, alt, pos=pos, chrom=chrom) for chrom, pos, ref, alt in variants]
    )
    assert len(vcf_variants) == len(variants)
    encodings = []
    for (chrom, pos, ref, alt), v in zip(variants, vcf_variants):
        datum = make_datum(contig_indices[chrom], pos, ref, alt)
        assert encode_variant(v, zero_based=True) == encode_datum(datum, contig_names)
        encodings.append(encode_datum(datum, contig_names))

    # different variants must not share an encoding, or one's allele frequency, filters and posterior go to the other
    assert len(set(encodings)) == len(variants)


def test_overlapping_filters(tmp_path):
    single, multiple = read_variants(
        tmp_path,
        vcf_line("A", "T", filters="germline"),
        vcf_line("A", "T", pos=200, filters="weak_evidence;germline;normal_artifact"),
    )
    assert overlapping_filters(single, {"germline", "weak_evidence"}) == {"germline"}
    assert overlapping_filters(multiple, {"germline", "weak_evidence", "contamination"}) == {
        "germline",
        "weak_evidence",
    }
    assert overlapping_filters(multiple, {"contamination"}) == set()


def test_overlapping_filters_edge_cases(tmp_path):
    passing, unfiltered, filtered = read_variants(
        tmp_path,
        vcf_line("A", "T", filters="PASS"),
        vcf_line("A", "T", pos=200, filters="."),
        vcf_line("A", "T", pos=300, filters="germline"),
    )
    # cyvcf2 reports both PASS and a missing FILTER as no filters, so PASS never overlaps
    assert overlapping_filters(passing, {"germline", "PASS"}) == set()
    assert overlapping_filters(unfiltered, {"germline"}) == set()

    # an empty set never overlaps
    assert overlapping_filters(filtered, set()) == set()

    # filter names must match exactly
    assert overlapping_filters(filtered, {"germ", "Germline", "germline_risk"}) == set()


@pytest.fixture
def default_device_is_meta(monkeypatch):
    # PyTorch's meta device stands in for a default device that differs from the one when permutect was imported
    monkeypatch.setattr(misc_utils, "gpu_if_available", lambda exploit_mps=False: torch.device("meta"))


def test_resolve_device(default_device_is_meta):
    assert resolve_device() == torch.device("meta")
    assert resolve_device("cpu") == torch.device("cpu")
    assert resolve_device(torch.device("cpu")) == torch.device("cpu")


def test_float_dtype():
    assert float_dtype("cpu") == float_dtype(torch.device("cpu")) == DEFAULT_CPU_FLOAT
    # every non-CPU device, including MPS, gets the GPU dtype
    for device in ("cuda", torch.device("cuda:1"), "mps", "meta"):
        assert float_dtype(device) == DEFAULT_GPU_FLOAT


# the device of each object made with the default device
DEFAULT_DEVICES = {
    "LossMetrics": lambda: LossMetrics(num_sources=1).totals_slvra.device,
    "AccuracyMetrics": lambda: AccuracyMetrics.create(num_sources=1).device,
    "EvaluationMetrics": lambda: EvaluationMetrics(num_sources=1).accuracy_metrics_by_epoch_type[Epoch.TRAIN].device,
    "BatchIndexedTensor.zeros": lambda: BatchIndexedTensor.zeros(num_sources=1).device,
    "BatchIndexedTensor.ones": lambda: BatchIndexedTensor.ones(num_sources=1).device,
    "PosteriorModelPriors": lambda: (
        PosteriorModelPriors(-10, -10, no_germline_mode=False).somatic_snv_log_priors_rrra.device
    ),
    "SNV context totals": lambda: PosteriorModelPriors.initialize_snv_context_totals_rrra().device,
    "PosteriorModelSpectra": lambda: PosteriorModelSpectra()._device,
    "PosteriorModel": lambda: next(PosteriorModel(-10, -10).parameters()).device,
}


@pytest.mark.parametrize("name", DEFAULT_DEVICES)
def test_default_device_is_resolved_when_called(default_device_is_meta, name):
    # a default argument of device=gpu_if_available() is evaluated once, at import
    assert DEFAULT_DEVICES[name]() == torch.device("meta")


def test_models_accept_string_devices():
    model = PosteriorModel(-10, -10, device="cpu")
    assert model._device == model.spectra._device == model.priors._device == torch.device("cpu")
    assert model._dtype == DEFAULT_CPU_FLOAT
