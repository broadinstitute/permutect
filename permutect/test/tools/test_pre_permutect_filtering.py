import tempfile

import cyvcf2
import pytest

from permutect import constants
from permutect.test.test_file_names import *
from permutect.tools import pre_permutect_filtering


def run_pre_permutect_filtering(input_vcf: str, reference: str, extra_args: tuple[str, ...] = ()):
    # the output is a temporary file, deleted when the returned file object is garbage collected
    output_vcf = tempfile.NamedTemporaryFile(suffix=".vcf")
    argv = ["--" + constants.INPUT_NAME, input_vcf, "--" + constants.REFERENCE_NAME, reference, *extra_args]
    pre_permutect_filtering.main_without_parsing(
        pre_permutect_filtering.parse_arguments(argv + ["--" + constants.OUTPUT_NAME, output_vcf.name])
    )
    return output_vcf


def record_filters(record) -> set:
    # cyvcf2 gives None for PASS and for a missing FILTER
    return set(record.FILTER.split(";")) if record.FILTER else set()


def test_hairpin_filtering_on_seqc2_wes_chr20():
    output_vcf = run_pre_permutect_filtering(MUTECT2_SEQC2_WES_VCF, HG38_CHR20_REFERENCE)

    target_records = [v for v in cyvcf2.VCF(output_vcf.name) if v.CHROM == "chr20" and v.POS == 4932703]
    assert len(target_records) == 1
    record = target_records[0]
    assert pre_permutect_filtering.HAIRPIN_FILTER in record.FILTER.split(";"), (
        "chr20:4932703 did not receive the hairpin filter; inspect its nearby StemLoop energy threshold"
    )
    assert record.INFO.get(pre_permutect_filtering.STEM_LOOP_FREE_ENERGY_INFO_KEY) == pytest.approx(1.3)
    assert record.INFO.get(pre_permutect_filtering.STEM_LOOP_DISTANCE_INFO_KEY) == 0


def test_no_hairpin_filter():
    output_vcf = run_pre_permutect_filtering(
        MUTECT2_SEQC2_WES_VCF, HG38_CHR20_REFERENCE, ("--" + constants.NO_HAIRPIN_FILTER_NAME,)
    )
    records = list(cyvcf2.VCF(output_vcf.name))
    assert records
    assert not any(pre_permutect_filtering.HAIRPIN_FILTER in record_filters(v) for v in records)


def test_multiallelic_indel_filter_matches_mutect2():
    # FilterMutectCalls has filtered this VCF, whose records are not split, so the multiallelic indel filter should
    # filter exactly the records with an indel allele that FilterMutectCalls filtered as multiallelic
    output_vcf = run_pre_permutect_filtering(MUTECT2_SEQC2_WES_VCF, HG38_CHR20_REFERENCE)
    num_filtered = 0
    for v in cyvcf2.VCF(output_vcf.name):
        filters = record_filters(v)
        expected = "multiallelic" in filters and pre_permutect_filtering.has_indel_allele(v)
        assert (pre_permutect_filtering.MULTIALLELIC_INDEL_FILTER in filters) == expected, f"{v.CHROM}:{v.POS}"
        num_filtered += expected
    assert num_filtered > 0


def test_multiallelic_indel_filter_on_split_records(tmp_path):
    # as in permutect.wdl, where SplitMultiallelics puts each alt allele of a site in a record of its own
    with open(MUTECT2_SEQC2_WES_VCF) as file:
        header = [line for line in file if line.startswith("#")]
    num_samples = len(header[-1].rstrip("\n").split("\t")) - 9
    samples = "\t".join(["0/1:10,5"] * num_samples)
    # (position, ref, alt, TLOD) of split records at three sites.  The TLOD threshold is 5 / ln(10) = 2.17.  The first
    # site has two passing indels, the second only one passing alt, and the third two passing SNVs and a passing indel.
    alleles = [
        (1000000, "CT", "C", 3.17),
        (1000000, "C", "CT", 4.70),
        (2000000, "A", "G", 22.6),
        (2000000, "A", "AG", 1.5),
        (3000000, "A", "G", 22.6),
        (3000000, "A", "T", 10.0),
        (3000000, "A", "AG", 3.0),
    ]
    input_vcf = tmp_path / "split.vcf"
    with open(input_vcf, "w") as file:
        file.writelines(header)
        for position, ref, alt, tlod in alleles:
            file.write(f"chr20\t{position}\t.\t{ref}\t{alt}\t.\tPASS\tTLOD={tlod}\tGT:AD\t{samples}\n")

    output_vcf = run_pre_permutect_filtering(
        str(input_vcf), HG38_CHR20_REFERENCE, ("--" + constants.NO_HAIRPIN_FILTER_NAME,)
    )
    filtered = {
        (v.POS, v.REF, v.ALT[0])
        for v in cyvcf2.VCF(output_vcf.name)
        if pre_permutect_filtering.MULTIALLELIC_INDEL_FILTER in record_filters(v)
    }
    assert filtered == {(1000000, "CT", "C"), (1000000, "C", "CT"), (3000000, "A", "AG")}


def test_vcf_without_format_dp_header():
    # like the output of SplitMultiallelics in permutect.wdl, this VCF has FORMAT/DP values but no FORMAT/DP header line
    output_vcf = run_pre_permutect_filtering(MUTECT2_CHR20_FILTERED_VCF, HG19_CHR20_REFERENCE)

    num_input_records = sum(1 for _ in cyvcf2.VCF(MUTECT2_CHR20_FILTERED_VCF))
    assert sum(1 for _ in cyvcf2.VCF(output_vcf.name)) == num_input_records
