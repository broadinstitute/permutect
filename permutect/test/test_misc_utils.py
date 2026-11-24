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


def test_timer(monkeypatch, capsys):
    # it prints its message, if any, when it's created
    Timer()
    assert capsys.readouterr().out == ""

    # fake the clock only where the Timer reads it, so that nothing else, eg another thread, consumes the fake times
    clock = iter([10.0, 11.5, 13.25])
    monkeypatch.setattr(misc_utils, "time", SimpleNamespace(perf_counter=lambda: next(clock)))
    timer = Timer("Doing something. . .")
    time.perf_counter()  # other code still reads the real clock
    timer.report("first")
    timer.report("second")

    # each report measures from when the timer was created, not from the previous report
    lines = capsys.readouterr().out.splitlines()
    assert lines == ["Doing something. . .", "first: 1.5000 seconds", "second: 3.2500 seconds"]


def test_get_first_numeric_element(tmp_path):
    biallelic, multiallelic, zero, missing = read_variants(
        tmp_path,
        vcf_line("A", "T", info="POPAF=3.5;DP=20"),
        vcf_line("G", "C,T", pos=200, info="POPAF=1.5,4.0"),
        vcf_line("A", "T", pos=300, info="POPAF=0"),
        vcf_line("A", "T", pos=400),
    )
    # cyvcf2 returns a single value as a scalar, even for a Number=A field, and multiple values as a tuple
    assert get_first_numeric_element(biallelic, "DP") == 20
    assert get_first_numeric_element(biallelic, "POPAF") == 3.5
    assert get_first_numeric_element(multiallelic, "POPAF") == 1.5

    # zero is a real value, not a missing one
    assert get_first_numeric_element(zero, "POPAF") == 0

    # a missing INFO field raises an error naming the variant, rather than returning a default
    with pytest.raises(ValueError, match="chr1:400 has no POPAF INFO field"):
        get_first_numeric_element(missing, "POPAF")


def test_allele_keys():
    # the ref allele distinguishes variants at one position with the same alt allele, eg from a split multiallelic site
    assert allele_key("AT", "A") != allele_key("ATT", "A") and allele_key("A", "AT") != allele_key("AC", "AT")

    # non-minimal representations have the keys of minimal ones, and trimming keeps the anchor base of an indel
    assert allele_key("CGG", "TGG") == allele_key("C", "T") and allele_key("ACGT", "AT") == allele_key("ACG", "A")
    assert allele_key("CTT", "CT") == allele_key("CT", "C") != allele_key("C", "C")

    # alleles of 13 bases are kept whole, and longer ones are truncated to 13 bases
    insertion_13 = "A" + "C" * 12
    assert allele_key("A", insertion_13 + "G") == allele_key("A", insertion_13) != allele_key("A", insertion_13[:-1])
    long_insertion = "A" + "CGT" * 10
    assert allele_key("A", long_insertion) == allele_key("A", long_insertion[:13] + "A")
    assert allele_key(long_insertion, "A") == allele_key(long_insertion[:13] + "A", "A")

    # trimming comes before truncation.  Truncating first would give CAAAAAAAAAAAG, which trimming against the ref allele
    # CG would then shorten to CAAAAAAAAAAA.
    assert allele_key("CG", "C" + "A" * 11 + "GTG") == allele_key("C", "C" + "A" * 11 + "G")
    assert allele_key("CG", "C" + "A" * 11 + "GTG") != allele_key("C", "C" + "A" * 11)

    # alleles with a base other than A, C, G or T, which no datum has, have no key
    assert allele_key("N", "A") is None and allele_key("A", "AN") is None


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


def test_vcf_and_dataset_keys_agree(tmp_path):
    # filter_variants and memory_mapped_data match dataset variants to VCF variants by position and allele keys, so these
    # must agree, distinct variants must have distinct keys, or one's allele frequency, filters and posterior go to the
    # other, and the keys must sort like the VCF
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
    single, multiple, passing, unfiltered = read_variants(
        tmp_path,
        vcf_line("A", "T", filters="germline"),
        vcf_line("A", "T", pos=200, filters="weak_evidence;germline;normal_artifact"),
        vcf_line("A", "T", pos=300, filters="PASS"),
        vcf_line("A", "T", pos=400, filters="."),
    )
    assert overlapping_filters(single, {"germline", "weak_evidence"}) == {"germline"}
    assert overlapping_filters(multiple, {"germline", "weak_evidence", "contamination"}) == {
        "germline",
        "weak_evidence",
    }
    assert overlapping_filters(multiple, {"contamination"}) == set()

    # cyvcf2 reports both PASS and a missing FILTER as no filters, so PASS never overlaps
    assert overlapping_filters(passing, {"germline", "PASS"}) == set()
    assert overlapping_filters(unfiltered, {"germline"}) == set()

    # an empty set never overlaps, and filter names must match exactly
    assert overlapping_filters(single, set()) == set()
    assert overlapping_filters(single, {"germ", "Germline", "germline_risk"}) == set()


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
