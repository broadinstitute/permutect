import tempfile

import cyvcf2
import pytest

from permutect import constants
from permutect.test.test_file_names import *
from permutect.tools import pre_permutect_filtering


def run_pre_permutect_filtering(input_vcf: str, reference: str):
    # the output is a temporary file, deleted when the returned file object is garbage collected
    output_vcf = tempfile.NamedTemporaryFile(suffix=".vcf")
    argv = ["--" + constants.INPUT_NAME, input_vcf, "--" + constants.REFERENCE_NAME, reference]
    pre_permutect_filtering.main_without_parsing(
        pre_permutect_filtering.parse_arguments(argv + ["--" + constants.OUTPUT_NAME, output_vcf.name])
    )
    return output_vcf


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


def test_vcf_without_format_dp_header():
    # like the output of SplitMultiallelics in permutect.wdl, this VCF has FORMAT/DP values but no FORMAT/DP header line
    output_vcf = run_pre_permutect_filtering(MUTECT2_CHR20_FILTERED_VCF, HG19_CHR20_REFERENCE)

    num_input_records = sum(1 for _ in cyvcf2.VCF(MUTECT2_CHR20_FILTERED_VCF))
    assert sum(1 for _ in cyvcf2.VCF(output_vcf.name)) == num_input_records
