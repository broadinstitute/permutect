import tempfile

import cyvcf2
import pytest

from permutect import constants
from permutect.test.test_file_names import *
from permutect.tools import filter_variants
from permutect.tools import pre_permutect_filtering
from permutect.utils.enums import Call


def test_filtering_on_dream1_chr20():
    artifact_model = PERMUTECT_V0_9_2_ARTIFACT_MODEL

    permutect_vcf = tempfile.NamedTemporaryFile()
    tensorboard_dir = tempfile.TemporaryDirectory()

    filtering_args = filter_variants.parse_arguments(
        [
            "--" + constants.INPUT_NAME,
            MUTECT2_CHR20_FILTERED_VCF,
            "--" + constants.TEST_DATASET_NAME,
            DREAM_1_CHR20_PLAIN_TEXT_DATA,
            "--" + constants.ARTIFACT_MODEL_NAME,
            artifact_model,
            "--" + constants.TRAINABLE_PARAMETERS_NAME,
            "CLUSTERING",
            "--" + constants.OUTPUT_NAME,
            permutect_vcf.name,
            "--" + constants.TENSORBOARD_DIR_NAME,
            tensorboard_dir.name,
            "--" + constants.GENOMIC_SPAN_NAME,
            "60000000",
            "--" + constants.CONTIGS_TABLE_NAME,
            HG19_CONTIGS_TABLE,
            "--" + constants.BATCH_SIZE_NAME,
            "64",
            "--" + constants.NUM_WORKERS_NAME,
            "2",
            "--" + constants.NUM_EPOCHS_NAME,
            "2",
            "--" + constants.WEIGHT_DECAY_NAME,
            "0.01",
        ]
    )

    filter_variants.main_without_parsing(filtering_args)

    # the output has one record per input record, in order.  Variants without a Permutect posterior, such as those
    # missing from the dataset, keep the filters they had in the input instead of becoming PASS
    num_unscored = 0
    input_records, output_records = cyvcf2.VCF(MUTECT2_CHR20_FILTERED_VCF), cyvcf2.VCF(permutect_vcf.name)
    for input_record, output_record in zip(input_records, output_records, strict=True):
        assert (input_record.CHROM, input_record.POS) == (output_record.CHROM, output_record.POS)
        if output_record.INFO.get(filter_variants.POST_PROB_INFO_KEY) is None:
            num_unscored += 1
            assert record_filters(output_record) >= record_filters(input_record)
    assert num_unscored > 0


def record_filters(record) -> set:
    # cyvcf2 gives None for PASS and for a missing FILTER
    return set(record.FILTER.split(";")) if record.FILTER else set()


def test_filtering_on_seqc2_wes_chr20():
    artifact_model = PERMUTECT_V0_9_2_ARTIFACT_MODEL

    pre_filtered_vcf = tempfile.NamedTemporaryFile(suffix=".vcf")
    permutect_vcf = tempfile.NamedTemporaryFile()
    tensorboard_dir = tempfile.TemporaryDirectory()

    pre_filtering_args = pre_permutect_filtering.parse_arguments(
        [
            "--" + constants.INPUT_NAME,
            MUTECT2_SEQC2_WES_VCF,
            "--" + constants.REFERENCE_NAME,
            HG38_CHR20_REFERENCE,
            "--" + constants.OUTPUT_NAME,
            pre_filtered_vcf.name,
        ]
    )
    pre_permutect_filtering.main_without_parsing(pre_filtering_args)

    filtering_args = filter_variants.parse_arguments(
        [
            "--" + constants.INPUT_NAME,
            pre_filtered_vcf.name,
            "--" + constants.TEST_DATASET_NAME,
            SEQC2_WES_PLAIN_TEXT_DATA,
            "--" + constants.ARTIFACT_MODEL_NAME,
            artifact_model,
            "--" + constants.TRAINABLE_PARAMETERS_NAME,
            "CLUSTERING",
            "--" + constants.OUTPUT_NAME,
            permutect_vcf.name,
            "--" + constants.TENSORBOARD_DIR_NAME,
            tensorboard_dir.name,
            "--" + constants.GENOMIC_SPAN_NAME,
            "60000000",
            "--" + constants.CONTIGS_TABLE_NAME,
            HG38_CONTIGS_TABLE,
            "--" + constants.MAF_SEGMENTS_NAME,
            SEQC2_WES_TUMOR_SEGMENTS,
            "--" + constants.NORMAL_MAF_SEGMENTS_NAME,
            SEQC2_WES_NORMAL_SEGMENTS,
            "--" + constants.BATCH_SIZE_NAME,
            "64",
            "--" + constants.NUM_WORKERS_NAME,
            "2",
            "--" + constants.NUM_EPOCHS_NAME,
            "2",
            "--" + constants.WEIGHT_DECAY_NAME,
            "0.01",
        ]
    )

    filter_variants.main_without_parsing(filtering_args)

    # variants with trusted non-artifact filters are output unchanged, while variants with trusted artifact filters
    # keep them and are seen by the posterior model as definite artifacts or normal artifacts
    input_records = {(v.CHROM, v.POS, v.REF, v.ALT[0]): str(v) for v in cyvcf2.VCF(pre_filtered_vcf.name)}
    num_non_artifact_filtered, num_artifact_filtered = 0, 0
    for v in cyvcf2.VCF(permutect_vcf.name):
        filters = set(v.FILTER.split(";")) if v.FILTER else set()
        if filters & filter_variants.TRUSTED_NON_ARTIFACT_FILTERS:
            num_non_artifact_filtered += 1
            assert str(v) == input_records[(v.CHROM, v.POS, v.REF, v.ALT[0])]
        elif filters & filter_variants.TRUSTED_ARTIFACT_FILTERS:
            num_artifact_filtered += 1
            posterior_probs = v.INFO.get(filter_variants.POST_PROB_INFO_KEY)
            assert posterior_probs[Call.ARTIFACT] + posterior_probs[Call.NORMAL_ARTIFACT] == pytest.approx(1, abs=0.01)
            assert v.INFO.get(filter_variants.ARTIFACT_LOD_INFO_KEY) is None
    assert num_non_artifact_filtered > 0 and num_artifact_filtered > 0

    target_records = [v for v in cyvcf2.VCF(permutect_vcf.name) if v.CHROM == "chr20" and v.POS == 4932703]
    assert len(target_records) == 1
    assert pre_permutect_filtering.HAIRPIN_FILTER in target_records[0].FILTER.split(";")


def segment_mafs(segmentation, contig: str, position: int) -> list[float]:
    return [interval.data for interval in segmentation[contig][position]]


def test_get_segmentation_from_model_segments():
    segmentation = filter_variants.get_segmentation(SEQC2_WES_TUMOR_SEGMENTS)
    # the segment chr20:187825-2536731 has median minor allele fraction posterior 0.176770
    for position in (187825, 1000000, 2536731):
        assert segment_mafs(segmentation, "chr20", position) == [pytest.approx(0.176770)]
    # before the first chr20 segment, and between it and the next segment, which starts at 2537107
    assert segment_mafs(segmentation, "chr20", 187824) == []
    assert segment_mafs(segmentation, "chr20", 2536732) == []


def test_get_segmentation_from_calculate_contamination():
    segmentation = filter_variants.get_segmentation(SEGMENTS_TABLE)
    for position in (17280822, 51065361):
        assert segment_mafs(segmentation, "22", position) == [pytest.approx(0.44015695469901417)]
    assert segment_mafs(segmentation, "22", 51065362) == []
