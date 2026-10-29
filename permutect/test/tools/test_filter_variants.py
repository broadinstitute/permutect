import random
import tempfile

import cyvcf2
import numpy as np
import pytest
import torch

from permutect import constants
from permutect.architecture.artifact_model import ArtifactModel
from permutect.architecture.artifact_model import load_model
from permutect.architecture.posterior_model import PosteriorModel
from permutect.data.datum import Data
from permutect.metrics.posterior_result import PosteriorResult
from permutect.test.architecture.test_posterior_model_priors import make_batch
from permutect.test.test_file_names import *
from permutect.tools import filter_variants
from permutect.tools import pre_permutect_filtering
from permutect.utils.enums import Call
from permutect.utils.enums import Variation


def test_filtering_on_dream1_chr20(monkeypatch):
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

    # every PosteriorResult that filter_variants makes
    posterior_results = []

    class RecordedPosteriorResult(PosteriorResult):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            posterior_results.append(self)

    monkeypatch.setattr(filter_variants, "PosteriorResult", RecordedPosteriorResult)
    filter_variants.main_without_parsing(filtering_args)

    # the values written to the VCF are plain numbers copied from the device once per batch.  Rows of device tensors
    # were read number by number while writing the VCF, which made the CPU wait for the GPU many times per variant.
    assert posterior_results
    for result in posterior_results:
        for values in (result.posterior_probabilities, result.log_priors, result.spectra_lls, result.normal_lls):
            assert isinstance(values, list) and all(isinstance(value, float) for value in values)
        assert result.embedding.device.type == "cpu"

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

    # these INFO fields have one value per Call type, which the header must declare for allele-aware tools
    per_call_info_keys = [
        filter_variants.POST_PROB_INFO_KEY,
        filter_variants.LOG_PRIOR_INFO_KEY,
        filter_variants.SPECTRA_LOG_LIKELIHOOD_INFO_KEY,
        filter_variants.NORMAL_LOG_LIKELIHOOD_INFO_KEY,
    ]
    output_vcf = cyvcf2.VCF(permutect_vcf.name)
    for info_key in per_call_info_keys:
        assert output_vcf.get_header_type(info_key)["Number"] == str(len(Call))
    for v in output_vcf:
        for info_key in per_call_info_keys:
            values = v.INFO.get(info_key)
            assert values is None or len(values) == len(Call)

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


def save_dropout_model(path):
    # the v0.9.2 hyperparameters with dropout, which makes the artifact logits random in training mode
    saved = torch.load(PERMUTECT_V0_9_2_ARTIFACT_MODEL, map_location="cpu", weights_only=False)
    hyperparams = saved[constants.HYPERPARAMS_NAME]
    hyperparams.dropout_p = 0.3
    model = ArtifactModel(
        hyperparams,
        num_read_features=saved[constants.NUM_READ_FEATURES_NAME],
        num_info_features=saved[constants.NUM_INFO_FEATURES_NAME],
        haplotypes_length=saved[constants.REF_SEQUENCE_LENGTH_NAME],
        device=torch.device("cpu"),
    )
    saved[constants.STATE_DICT_NAME] = model.state_dict()
    torch.save(saved, path)


def test_filtering_checks_germline_mode_like_the_posterior_model(tmp_path):
    # filter_variants used to compute error probabilities itself, skipping the PosteriorModel check that germline mode
    # is not combined with a model in no-germline mode
    model = PosteriorModel(-10, -10, no_germline_mode=True, device=torch.device("cpu"))
    values = {Data.ORIGINAL_DEPTH: 30, Data.ORIGINAL_ALT_COUNT: 5, Data.ORIGINAL_NORMAL_DEPTH: 30, Data.MAF: 0.5}
    loader = [make_batch([0.01, 0.1], values)]
    thresholds = {var_type: 0.5 for var_type in Variation}
    output_vcf = str(tmp_path / "filtered.vcf")
    with pytest.raises(AssertionError, match="incompatible"):
        filter_variants.apply_filtering_to_vcf(
            MUTECT2_CHR20_FILTERED_VCF, output_vcf, {0: "20"}, thresholds, loader, model, None, germline_mode=True
        )


def test_load_model_returns_model_in_eval_mode():
    model = load_model(PERMUTECT_V0_9_2_ARTIFACT_MODEL, device=torch.device("cpu"))
    assert not any(module.training for module in model.modules())


def test_posterior_data_artifact_logits_are_deterministic(tmp_path):
    model_path = tmp_path / "dropout-model.pt"
    save_dropout_model(model_path)
    model = load_model(model_path, device=torch.device("cpu"))

    contig_index_to_name_map = {}
    with open(HG38_CONTIGS_TABLE) as file:
        for line in file:
            contig, index = line.split()
            contig_index_to_name_map[int(index)] = contig
    dataset = filter_variants.make_annotated_dataset(
        SEQC2_WES_PLAIN_TEXT_DATA, MUTECT2_SEQC2_WES_VCF, contig_index_to_name_map
    )

    def artifact_logits():
        random.seed(0)  # the dataset shuffles its data
        posterior_data = filter_variants.generate_posterior_data(dataset, model, batch_size=64, num_workers=0)
        return np.array([datum.get(Data.CACHED_ARTIFACT_LOGIT) for datum in posterior_data])

    # first straight from load_model, then in training mode, as test-time adaptation leaves it
    first_logits = artifact_logits()
    model.train(True)
    second_logits = artifact_logits()
    assert np.array_equal(first_logits, second_logits)
