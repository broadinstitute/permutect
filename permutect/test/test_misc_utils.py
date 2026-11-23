import itertools
import math
import time
from types import SimpleNamespace

import cyvcf2
import numpy as np
import pytest
import torch

from permutect import misc_utils
from permutect.architecture.posterior_model import PosteriorModel
from permutect.architecture.posterior_model_priors import PosteriorModelPriors
from permutect.architecture.spectra.posterior_model_spectra import PosteriorModelSpectra
from permutect.data.batch import BatchIndexedTensor
from permutect.data.datum import DEFAULT_CPU_FLOAT
from permutect.data.datum import DEFAULT_GPU_FLOAT
from permutect.data.datum import FLOAT_DTYPE
from permutect.data.datum import INTEGER_DTYPE
from permutect.data.datum import RAW_READS_ARRAY_DTYPE
from permutect.data.datum import Data
from permutect.data.datum import Datum
from permutect.data.datum import gatk_variant_features
from permutect.metrics.evaluation_metrics import EvaluationMetrics
from permutect.metrics.loss_metrics import AccuracyMetrics
from permutect.metrics.loss_metrics import LossMetrics
from permutect.misc_utils import Timer
from permutect.misc_utils import VcfEditor
from permutect.misc_utils import allele_key
from permutect.misc_utils import encode
from permutect.misc_utils import encode_datum
from permutect.misc_utils import encode_variant
from permutect.misc_utils import float_dtype
from permutect.misc_utils import get_first_numeric_element
from permutect.misc_utils import get_tumor_sample_index
from permutect.misc_utils import overlapping_filters
from permutect.misc_utils import position_key
from permutect.misc_utils import report_device
from permutect.misc_utils import resolve_device
from permutect.misc_utils import str_repeat_count_change
from permutect.misc_utils import tumor_alt_orientation_counts
from permutect.misc_utils import tumor_depths
from permutect.misc_utils import tumor_ref_orientation_counts
from permutect.misc_utils import vcf_contig_ranks
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
    # the variants remain usable after the reader is closed
    with cyvcf2.VCF(str(path)) as vcf:
        return list(vcf)


def make_datum(contig: int, position: int, ref: str, alt: str) -> Datum:
    """
    An unlabeled datum with three alt reads, as preprocessing makes it from GATK's plain text data, with the float fields
    that GATK doesn't give NaN
    """
    # the ref allele goes in the middle of an odd-length reference context, as in GATK data
    flank = "GATCCTAGGATCCTAGGATC"
    ref_sequence = flank + ref + flank[: len(flank) + 1 - len(ref)]
    str_info, haplotypes, ref_base5, alt_base5 = gatk_variant_features(ref_sequence, ref, alt)
    int_array = np.zeros(Data.NUM_SCALAR_INT_ELEMENTS + len(haplotypes), dtype=INTEGER_DTYPE)
    int_array[Data.HAPLOTYPES_START_IDX :] = haplotypes
    float_array = np.full(Data.NUM_SCALAR_FLOAT_ELEMENTS + 3 + len(str_info), np.nan, dtype=FLOAT_DTYPE)
    float_array[Data.INFO_START_IDX :] = np.hstack([np.zeros(3), str_info])
    datum = Datum(int_array, float_array, reads_re=np.zeros((3, 5), dtype=RAW_READS_ARRAY_DTYPE))
    fields = {
        Data.ALT_COUNT: 3,
        Data.LABEL: Label.UNLABELED,
        Data.VARIANT_TYPE: Variation.get_type(ref, alt),
        Data.ORIGINAL_DEPTH: 10,
        Data.ORIGINAL_ALT_COUNT: 3,
        Data.CONTIG: contig,
        Data.POSITION: position,
        Data.REF_ALLELE_AS_BASE_5: ref_base5,
        Data.ALT_ALLELE_AS_BASE_5: alt_base5,
        Data.SEQ_ERROR_LOG_LK: 0.0,
        Data.NORMAL_SEQ_ERROR_LOG_LK: 0.0,
    }
    for field, value in fields.items():
        datum.set(field, value)
    return datum


def test_timer_prints_message_on_creation(capsys):
    Timer("Doing something. . .")
    assert capsys.readouterr().out == "Doing something. . .\n"

    Timer()
    assert capsys.readouterr().out == ""


def test_timer_reports_elapsed_time(monkeypatch, capsys):
    # fake the clock only where the Timer reads it, so that nothing else, eg another thread, consumes the fake times
    clock = iter([10.0, 11.5, 13.25])
    monkeypatch.setattr(misc_utils, "time", SimpleNamespace(perf_counter=lambda: next(clock)))

    timer = Timer()
    time.perf_counter()  # other code still reads the real clock
    timer.report("first")
    timer.report("second")

    # each report measures from when the timer was created, not from the previous report
    assert capsys.readouterr().out.splitlines() == ["first: 1.5000 seconds", "second: 3.2500 seconds"]


def test_timer_with_real_clock(capsys):
    start = time.perf_counter()
    timer = Timer()
    timer.report("elapsed")
    max_elapsed = time.perf_counter() - start
    output = capsys.readouterr().out.strip()
    assert output.startswith("elapsed: ")
    assert output.endswith(" seconds")
    elapsed = float(output.removeprefix("elapsed: ").removesuffix(" seconds"))
    # bounded by the time measured around the Timer, plus rounding to 4 decimals, rather than a fixed time that a
    # loaded machine could exceed
    assert 0 <= elapsed <= max_elapsed + 0.00005


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

    # a missing INFO field raises an error naming the variant, rather than returning a default
    with pytest.raises(ValueError, match="chr1:200 has no POPAF INFO field"):
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
    # the one-based VCF position, not cyvcf2's zero-based start
    assert encode_variant(snv) == "chr1:100:A:T"
    assert encode_variant(deletion) == "chr1:200:ACG:A"
    assert encode_variant(insertion) == "chrX:300:C:CTT"


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
    assert encode_variant(multiallelic) == "chr1:100:G:C"

    # alleles are trimmed and truncated
    assert encode_variant(non_minimal) == "chr1:200:C:T"
    assert encode_variant(long) == "chr1:300:A:" + long_insertion[:13]

    # the first base of a contig
    assert encode_variant(first_base) == "chr1:1:A:T"


# variants whose records in a VCF and data in a dataset must match, in the order of a sorted VCF
MATCHED_VARIANTS = [
    ("chr1", 100, "C", "T"),
    ("chr1", 200, "CAG", "C"),
    ("chr1", 300, "C", "CTT"),
    ("chr1", 400, "CGG", "TGG"),
    ("chr1", 500, "C", "C" + "AGT" * 6),
    ("chr1", 600, "CG", "C" + "A" * 11 + "GTG"),  # the datum must also trim before truncating
    ("chr1", 700, "CAG", "C"),  # two deletions at one position, eg from a split multiallelic site
    ("chr1", 700, "CA", "C"),
    ("chr1", 800, "C" + "AGT" * 6, "C"),  # the datum truncates the ref allele of a long deletion
    ("chr1", 900, "AT", "GC"),  # an MNV
    ("chr1", 900, "AT", "GT"),  # an SNV at the MNV's first base, padded as at a split multiallelic site
    ("chr1", 901, "T", "C"),  # an SNV at the MNV's second base
    ("chrX", 150_000_000, "C", "T"),
]


def test_vcf_and_dataset_encodings_agree(tmp_path):
    # filter_variants and memory_mapped_data match dataset variants to VCF variants by encoding, so these must agree
    contig_names = {0: "chr1", 1: "chrX"}
    contig_indices = {name: index for index, name in contig_names.items()}

    vcf_variants = read_variants(
        tmp_path, *[vcf_line(ref, alt, pos=pos, chrom=chrom) for chrom, pos, ref, alt in MATCHED_VARIANTS]
    )
    assert len(vcf_variants) == len(MATCHED_VARIANTS)
    encodings = []
    for (chrom, pos, ref, alt), v in zip(MATCHED_VARIANTS, vcf_variants):
        datum = make_datum(contig_indices[chrom], pos, ref, alt)
        assert encode_variant(v) == encode_datum(datum, contig_names)
        encodings.append(encode_datum(datum, contig_names))

    # different variants must not share an encoding, or one's allele frequency, filters and posterior go to the other
    assert len(set(encodings)) == len(MATCHED_VARIANTS)


def test_vcf_and_dataset_keys_agree(tmp_path):
    # filter_variants matches the results of dataset variants to VCF variants by position and allele keys, so these must
    # agree like the encodings, distinct variants must have distinct keys, and the keys must sort like the VCF
    contig_names = {0: "chr1", 1: "chrX"}
    contig_indices = {name: index for index, name in contig_names.items()}
    vcf_variants = read_variants(
        tmp_path, *[vcf_line(ref, alt, pos=pos, chrom=chrom) for chrom, pos, ref, alt in MATCHED_VARIANTS]
    )
    keys = []
    for (chrom, pos, ref, alt), v in zip(MATCHED_VARIANTS, vcf_variants, strict=True):
        datum = make_datum(contig_indices[chrom], pos, ref, alt)
        rank = contig_indices[chrom]
        vcf_key = (position_key(rank, v.POS), allele_key(v.REF, v.ALT[0]))
        datum_alleles = allele_key(datum.get_ref_allele(), datum.get_alt_allele())
        assert vcf_key == (position_key(rank, datum.get(Data.POSITION)), datum_alleles)
        keys.append(vcf_key)
    assert len(set(keys)) == len(MATCHED_VARIANTS)
    assert [key for key, _ in keys] == sorted(key for key, _ in keys)


def test_allele_keys_match_like_encodings():
    # two variants have the same allele key if and only if encode gives them the same alleles
    alleles = [(ref, alt) for _, _, ref, alt in MATCHED_VARIANTS] + [("A", "AC" * 7), ("A", "AC" * 8), ("ATT", "AT")]
    for (ref1, alt1), (ref2, alt2) in itertools.product(alleles, repeat=2):
        same_encoding = encode("chr1", 1, ref1, alt1) == encode("chr1", 1, ref2, alt2)
        assert (allele_key(ref1, alt1) == allele_key(ref2, alt2)) == same_encoding, (ref1, alt1, ref2, alt2)

    # alleles with a base other than A, C, G or T, which no datum has, have no key
    assert allele_key("N", "A") is None and allele_key("A", "AN") is None


def test_vcf_contig_ranks(tmp_path):
    # the order of the header's contig lines, or of the contigs table for a header without them
    contig_names = {0: "chrX", 1: "chr1"}
    with_contigs = tmp_path / "with_contigs.vcf"
    with_contigs.write_text(VCF_HEADER)
    assert vcf_contig_ranks(str(with_contigs), contig_names) == {"chr1": 0, "chrX": 1}

    without_contigs = tmp_path / "without_contigs.vcf"
    without_contigs.write_text(
        "".join(line + "\n" for line in VCF_HEADER.splitlines() if not line.startswith("##contig"))
    )
    assert vcf_contig_ranks(str(without_contigs), contig_names) == {"chrX": 0, "chr1": 1}

    # a position key sorts by contig, then by position
    assert position_key(0, 2**32 - 1) < position_key(1, 1)


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


# a tumor-normal Mutect2 header without a FORMAT/DP line for its DP values, like the output of SplitMultiallelics
TUMOR_NORMAL_VCF_HEADER = VCF_HEADER.replace(
    "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n",
    '##FORMAT=<ID=AD,Number=R,Type=Integer,Description="Allelic depths">\n'
    "##tumor_sample=tumor\n"
    "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tnormal\ttumor\n",
)


def test_vcf_editor(tmp_path):
    input_vcf, output_vcf = tmp_path / "input.vcf", str(tmp_path / "output.vcf")
    lines = [
        vcf_line("A", "T", pos=100) + "\tAD:DP\t10,0:10\t6,3:9",
        vcf_line("C", "G", pos=200) + "\tAD:DP\t8,1:9\t5,0:5",
    ]
    input_vcf.write_text(TUMOR_NORMAL_VCF_HEADER + "".join(line + "\n" for line in lines))

    info_fields = {"SCORE": ("Integer", 1, "a score"), "PER_ALT": ("Float", "A", "a value for each alt allele")}
    editor = VcfEditor(str(input_vcf), output_vcf, info_fields, filters={"low_score": "the score is low"})
    assert editor.tumor_sample_index == 1
    depths_by_position = {}
    for v in editor:
        depths_by_position[v.POS] = tumor_depths(v, editor.tumor_sample_index)
        v.INFO["SCORE"] = v.POS // 100
        if v.POS == 100:
            continue  # every record is written, including those that skip the rest of the loop
        v.FILTER = "low_score"
    assert depths_by_position == {100: (6, 3), 200: (5, 0)}

    with cyvcf2.VCF(output_vcf) as output:
        assert [output.get_header_type(key)["Number"] for key in ("SCORE", "PER_ALT")] == ["1", "A"]
        assert '##FILTER=<ID=low_score,Description="the score is low">' in output.raw_header
        assert [(v.POS, v.INFO["SCORE"], v.FILTER) for v in output] == [(100, 1, None), (200, 2, "low_score")]


@pytest.fixture
def default_device_is_meta(monkeypatch):
    # PyTorch's meta device stands in for a default device that differs from the one when permutect was imported
    monkeypatch.setattr(misc_utils, "gpu_if_available", lambda exploit_mps=False: torch.device("meta"))


def test_report_device_explains_running_on_the_cpu(capsys, monkeypatch):
    # eg the CUDA 13 docker image on a GPU whose driver is too old for CUDA 13
    monkeypatch.setattr(torch.version, "cuda", "13.0")
    report_device(torch.device("cpu"))
    output = capsys.readouterr().out
    assert "Running on cpu" in output and "CUDA 13.0" in output


@pytest.mark.parametrize("world_size", [None, "1"])
def test_init_distributed_in_a_single_process(default_device_is_meta, monkeypatch, world_size):
    # without torchrun, or under torchrun with one process, training runs as before, without a process group
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    if world_size is not None:
        monkeypatch.setenv("WORLD_SIZE", world_size)
    assert misc_utils.init_distributed() == torch.device("meta")
    assert not misc_utils.is_distributed() and misc_utils.is_main_process()


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


def test_tumor_alt_orientation_counts(tmp_path):
    # the tumor's alt F1R2 and F2R1 counts, each NaN where the record lacks it or its value is missing
    header = TUMOR_NORMAL_VCF_HEADER.replace(
        "##tumor_sample",
        '##FORMAT=<ID=F1R2,Number=R,Type=Integer,Description="Count of reads in F1R2 pair orientation">\n'
        '##FORMAT=<ID=F2R1,Number=R,Type=Integer,Description="Count of reads in F2R1 pair orientation">\n'
        "##tumor_sample",
    )
    lines = [
        vcf_line("A", "T", pos=100) + "\tAD:F1R2:F2R1\t10,0:5,0:5,0\t6,3:4,3:2,0",
        vcf_line("C", "G", pos=200) + "\tAD\t8,1\t5,2",
        vcf_line("G", "A", pos=300) + "\tAD:F1R2:F2R1\t8,1:4,.:4,1\t5,2:3,.:2,2",
    ]
    path = tmp_path / "variants.vcf"
    path.write_text(header + "".join(line + "\n" for line in lines))
    with cyvcf2.VCF(str(path)) as vcf:
        tumor_sample_index = get_tumor_sample_index(vcf)
        counts = [tumor_alt_orientation_counts(v, tumor_sample_index) for v in vcf]
    with cyvcf2.VCF(str(path)) as vcf:
        ref_counts = [tumor_ref_orientation_counts(v, tumor_sample_index) for v in vcf]
    assert counts[0] == (3.0, 0.0)
    assert all(math.isnan(count) for count in counts[1])
    assert math.isnan(counts[2][0]) and counts[2][1] == 2.0
    # likewise the ref reads'
    assert ref_counts[0] == (4.0, 2.0) and ref_counts[2] == (3.0, 2.0)
    assert all(math.isnan(count) for count in ref_counts[1])


def test_tumor_alt_orientation_counts_without_the_annotations_in_the_header(tmp_path):
    lines = [vcf_line("A", "T", pos=100) + "\tAD\t10,0\t6,3"]
    path = tmp_path / "variants.vcf"
    path.write_text(TUMOR_NORMAL_VCF_HEADER + "".join(line + "\n" for line in lines))
    with cyvcf2.VCF(str(path)) as vcf:
        (counts,) = [tumor_alt_orientation_counts(v, get_tumor_sample_index(vcf)) for v in vcf]
    assert all(math.isnan(count) for count in counts)


def test_str_repeat_count_change(tmp_path):
    # the change in the number of repeat units from the reference to the alt allele, from RPA, or 0 outside an STR
    header = VCF_HEADER.replace(
        "#CHROM",
        '##INFO=<ID=RU,Number=1,Type=String,Description="Tandem repeat unit">\n'
        '##INFO=<ID=RPA,Number=R,Type=Integer,Description="Repeats per allele">\n#CHROM',
    )
    lines = [
        vcf_line("CAG", "C", pos=100, info="RU=AG;RPA=6,5"),
        vcf_line("C", "CAG", pos=200, info="RU=AG;RPA=6,7"),
        vcf_line("CAGAG", "C", pos=300, info="RU=AG;RPA=6,4"),
        vcf_line("C", "T", pos=400),
    ]
    path = tmp_path / "variants.vcf"
    path.write_text(header + "".join(line + "\n" for line in lines))
    with cyvcf2.VCF(str(path)) as vcf:
        assert [str_repeat_count_change(v) for v in vcf] == [-1, 1, -2, 0]
