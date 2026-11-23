import math
import random
import re
import tempfile

import cyvcf2
import numpy as np
import pytest
import torch
from torch.utils.tensorboard import SummaryWriter

from permutect import constants
from permutect.architecture.artifact_model import ArtifactModel
from permutect.architecture.artifact_model import load_model
from permutect.architecture.posterior_model import PosteriorModel
from permutect.data.batch import Batch
from permutect.data.compact_posterior_data import CompactPosteriorData
from permutect.data.datum import Data
from permutect.metrics.evaluation_metrics import EvaluationMetrics
from permutect.metrics.posterior_result import PosteriorResult
from permutect.misc_utils import allele_key
from permutect.misc_utils import get_tumor_sample_index
from permutect.misc_utils import str_repeat_count_change
from permutect.misc_utils import str_repeat_lengths
from permutect.misc_utils import tumor_alt_orientation_counts
from permutect.misc_utils import tumor_ref_orientation_counts
from permutect.segmentation.germline_model import GermlineModel
from permutect.segmentation.germline_model import SampleSegmentation
from permutect.segmentation.segmenter import Segment
from permutect.slippage.slippage_model import MIN_SLIPPAGE_STR_LENGTH
from permutect.test.architecture.test_posterior_model import make_read_orientation_model
from permutect.test.architecture.test_posterior_model_priors import make_batch
from permutect.test.test_file_names import *
from permutect.test.test_misc_utils import VCF_HEADER
from permutect.test.test_misc_utils import make_datum as make_datum_with_alleles
from permutect.test.test_misc_utils import vcf_line
from permutect.test.tools.test_pre_permutect_filtering import run_pre_permutect_filtering
from permutect.tools import filter_variants
from permutect.tools import pre_permutect_filtering
from permutect.tools import segment_pileups
from permutect.utils.allele_utils import find_variant_type
from permutect.utils.enums import INDEL_VARIATIONS
from permutect.utils.enums import Call
from permutect.utils.enums import Label
from permutect.utils.enums import Variation

# the arguments of both end-to-end filtering tests, including brief test-time adaptation
SHARED_FILTERING_ARGS = {
    constants.ARTIFACT_MODEL_NAME: PERMUTECT_V0_9_2_ARTIFACT_MODEL,
    constants.TRAINABLE_PARAMETERS_NAME: "CLUSTERING",
    constants.GENOMIC_SPAN_NAME: "60000000",
    constants.BATCH_SIZE_NAME: "64",
    constants.NUM_WORKERS_NAME: "2",
    constants.NUM_EPOCHS_NAME: "2",
    constants.WEIGHT_DECAY_NAME: "0.01",
}


def parse_filtering_arguments(args: dict):
    # the shared arguments and the given ones, each by name
    return filter_variants.parse_arguments(
        [word for name, value in (SHARED_FILTERING_ARGS | args).items() for word in ("--" + name, value)]
    )


def test_filtering_on_dream1_chr20(monkeypatch, capsys):
    permutect_vcf = tempfile.NamedTemporaryFile()
    tensorboard_dir = tempfile.TemporaryDirectory()

    filtering_args = parse_filtering_arguments(
        {
            constants.INPUT_NAME: MUTECT2_CHR20_FILTERED_VCF,
            constants.TEST_DATASET_NAME: DREAM_1_CHR20_PLAIN_TEXT_DATA,
            constants.CONTIGS_TABLE_NAME: HG19_CONTIGS_TABLE,
            constants.OUTPUT_NAME: permutect_vcf.name,
            constants.TENSORBOARD_DIR_NAME: tensorboard_dir.name,
            constants.SPECTRUM_BATCH_SIZE_NAME: "256",
        }
    )

    # every CompactPosteriorData that filter_variants makes
    compact_data = []

    class RecordedCompactPosteriorData(CompactPosteriorData):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            compact_data.append(self)

    monkeypatch.setattr(filter_variants, "CompactPosteriorData", RecordedCompactPosteriorData)

    # every PosteriorResult that filter_variants makes
    posterior_results = []

    class RecordedPosteriorResult(PosteriorResult):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            posterior_results.append(self)

    monkeypatch.setattr(filter_variants, "PosteriorResult", RecordedPosteriorResult)

    # every SummaryWriter that filter_variants makes
    writers = []

    class RecordedSummaryWriter(SummaryWriter):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.closed = False
            writers.append(self)

        def close(self):
            super().close()
            self.closed = True

    monkeypatch.setattr(filter_variants, "SummaryWriter", RecordedSummaryWriter)
    filter_variants.main_without_parsing(filtering_args)
    # the log says which device it ran on, so that a job with a GPU that runs on the CPU is visible
    assert "Running on " in capsys.readouterr().out

    # the writers for test-time adaptation and for the posterior model are closed, so that their last events, such as
    # the final plots, aren't lost when the program exits
    assert len(writers) == 2 and all(writer.closed for writer in writers)

    # fitting the spectra and finding the thresholds use the spectrum batch size, not the data loaders' batch size of 64
    assert [data.batch_size for data in compact_data] == [256]

    # the values written to the VCF are plain numbers copied from the device once per batch.  Rows of device tensors
    # were read number by number while writing the VCF, which made the CPU wait for the GPU many times per variant.
    assert posterior_results
    for result in posterior_results:
        for values in (result.posterior_probabilities, result.log_priors, result.spectra_lls, result.normal_lls):
            assert isinstance(values, list) and all(isinstance(value, float) for value in values)
        assert result.embedding.device.type == "cpu"

    # each variant has a twin simulated from the feature clustering model, from the nonartifact component or an artifact
    # cluster but not the outlier pseudocluster.  The model gives every variant here an artifact probability of at
    # least about 0.8, so there may be no nonartifact twins at all.
    for result in posterior_results:
        assert result.simulated_embedding.shape == result.embedding.shape
        assert torch.isfinite(result.simulated_embedding).all()
        assert re.fullmatch(r"nonartifact|artifact cluster \d+", result.simulated_source), result.simulated_source
    assert any(result.simulated_source.startswith("artifact cluster") for result in posterior_results)

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

    # the INFO descriptions name Permutect, not its old name Mutect3
    for key in (filter_variants.POST_PROB_INFO_KEY, filter_variants.ARTIFACT_LOD_INFO_KEY):
        assert "Permutect" in output_records.get_header_type(key)["Description"]


def record_filters(record) -> set:
    # cyvcf2 gives None for PASS and for a missing FILTER
    return set(record.FILTER.split(";")) if record.FILTER else set()


def test_filtering_on_seqc2_wes_chr20(tmp_path):
    pre_filtered_vcf = run_pre_permutect_filtering(MUTECT2_SEQC2_WES_VCF, HG38_CHR20_REFERENCE)
    permutect_vcf = tempfile.NamedTemporaryFile()
    tensorboard_dir = tempfile.TemporaryDirectory()
    germline_model = tmp_path / "germline_model.pt"
    read_orientation_model = tmp_path / "read_orientation_model.pt"
    make_read_orientation_model(sample=tumor_sample(pre_filtered_vcf.name)).save(read_orientation_model)
    segment_pileups.segment_pileups(
        SEQC2_WES_TUMOR_PILEUPS,
        tmp_path / "tumor.segments.table",
        normal_input_table=SEQC2_WES_NORMAL_PILEUPS,
        germline_model=germline_model,
    )

    filtering_args = parse_filtering_arguments(
        {
            constants.INPUT_NAME: pre_filtered_vcf.name,
            constants.TEST_DATASET_NAME: SEQC2_WES_PLAIN_TEXT_DATA,
            constants.CONTIGS_TABLE_NAME: HG38_CONTIGS_TABLE,
            constants.GERMLINE_MODEL_NAME: str(germline_model),
            constants.READ_ORIENTATION_MODEL_NAME: str(read_orientation_model),
            constants.OUTPUT_NAME: permutect_vcf.name,
            constants.TENSORBOARD_DIR_NAME: tensorboard_dir.name,
            # normalizing the test dataset in a process of its own
            constants.NUM_PROCESSES_NAME: "2",
        }
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

    # variants with trusted artifact filters keep them and are seen by the posterior model as definite artifacts or
    # normal artifacts, while variants that Mutect2 filtered for contamination are scored by the posterior model, which
    # has its own contamination call
    mutect2_contamination = {
        (v.CHROM, v.POS, v.REF, v.ALT[0])
        for v in cyvcf2.VCF(pre_filtered_vcf.name)
        if "contamination" in record_filters(v)
    }
    num_scored_mutect2_contamination, num_artifact_filtered = 0, 0
    for v in cyvcf2.VCF(permutect_vcf.name):
        filters = record_filters(v)
        posterior_probs = v.INFO.get(filter_variants.POST_PROB_INFO_KEY)
        if (v.CHROM, v.POS, v.REF, v.ALT[0]) in mutect2_contamination and posterior_probs is not None:
            num_scored_mutect2_contamination += 1
        if filters & filter_variants.TRUSTED_ARTIFACT_FILTERS:
            num_artifact_filtered += 1
            assert posterior_probs[Call.ARTIFACT] + posterior_probs[Call.NORMAL_ARTIFACT] == pytest.approx(1, abs=0.01)
            assert v.INFO.get(filter_variants.ARTIFACT_LOD_INFO_KEY) is None
    assert num_scored_mutect2_contamination > 0 and num_artifact_filtered > 0

    target_records = [v for v in cyvcf2.VCF(permutect_vcf.name) if v.CHROM == "chr20" and v.POS == 4932703]
    assert len(target_records) == 1
    assert pre_permutect_filtering.HAIRPIN_FILTER in target_records[0].FILTER.split(";")

    # given a read orientation model, a variant filtered as a sequencing error only because of damage gets the
    # orientation filter instead, and only SNVs can
    assert filter_variants.ORIENTATION_FILTER in cyvcf2.VCF(permutect_vcf.name).raw_header
    for v in cyvcf2.VCF(permutect_vcf.name):
        if filter_variants.ORIENTATION_FILTER in record_filters(v):
            assert find_variant_type(v) == Variation.SNV
            assert FILTER_NAMES_OF_CALLS[Call.SEQ_ERROR] not in record_filters(v)

    # Mutect2's slippage filter isn't trusted, so the posterior model scores the variants it filtered.  The posterior
    # model's own slippage calls are one-unit indels of STRs of at least the minimum length.
    mutect2_slippage = {
        (v.CHROM, v.POS, v.REF, v.ALT[0]) for v in cyvcf2.VCF(pre_filtered_vcf.name) if "slippage" in record_filters(v)
    }
    num_scored_mutect2_slippage, num_slippage_calls = 0, 0
    for v in cyvcf2.VCF(permutect_vcf.name):
        scored = v.INFO.get(filter_variants.POST_PROB_INFO_KEY) is not None
        num_scored_mutect2_slippage += scored and (v.CHROM, v.POS, v.REF, v.ALT[0]) in mutect2_slippage
        if scored and "slippage" in record_filters(v):
            num_slippage_calls += 1
            assert find_variant_type(v) in INDEL_VARIATIONS
            assert abs(str_repeat_count_change(v)) == 1 and str_repeat_lengths(v)[1] >= MIN_SLIPPAGE_STR_LENGTH
    assert num_scored_mutect2_slippage > 0 and num_slippage_calls > 0


FILTER_NAMES_OF_CALLS = filter_variants.FILTER_NAMES


def posterior_result_with(probabilities: list, error_probability: float, without_damage=None) -> PosteriorResult:
    return PosteriorResult(
        artifact_logit=0.0,
        posterior_probabilities=probabilities,
        error_probability=error_probability,
        log_priors=None,
        spectra_lls=None,
        normal_lls=None,
        label=None,
        alt_count=6,
        depth=100,
        var_type=Variation.SNV,
        embedding=None,
        simulated_embedding=None,
        simulated_source="",
        without_damage=without_damage,
    )


def one_hot(call: Call, probability: float = 0.9) -> list:
    """posterior probabilities of probability for the given call and the rest spread evenly over the others"""
    return [probability if c == call else (1 - probability) / (len(Call) - 1) for c in Call]


def test_most_likely_error_call():
    assert filter_variants.most_likely_error_call(one_hot(Call.SEQ_ERROR), Call.SOMATIC) == Call.SEQ_ERROR
    # the passing call doesn't count
    probabilities = one_hot(Call.SOMATIC)
    probabilities[Call.ARTIFACT] += 0.01
    assert filter_variants.most_likely_error_call(probabilities, Call.SOMATIC) == Call.ARTIFACT


@pytest.mark.parametrize(
    "without_damage, expected",
    [
        (None, False),  # without a read orientation model
        ((one_hot(Call.SEQ_ERROR), 0.9), False),  # a sequencing error anyway
        ((one_hot(Call.SOMATIC), 0.1), True),  # it would pass
        ((one_hot(Call.ARTIFACT), 0.9), True),  # another call
    ],
)
def test_error_only_because_of_damage(without_damage, expected):
    posterior_result = posterior_result_with(one_hot(Call.SEQ_ERROR), 0.9, without_damage)
    assert filter_variants.is_error_only_because_of_damage(posterior_result, 0.5, Call.SOMATIC) == expected


def tumor_sample(vcf_path) -> str:
    vcf = cyvcf2.VCF(vcf_path)
    return vcf.samples[get_tumor_sample_index(vcf)]


def test_the_read_orientation_model_must_be_of_the_tumor_sample(tmp_path):
    model_path = tmp_path / "read_orientation_model.pt"
    make_read_orientation_model(sample=tumor_sample(MUTECT2_SEQC2_WES_VCF)).save(model_path)
    assert filter_variants.load_read_orientation_model(model_path, MUTECT2_SEQC2_WES_VCF).sample == "WES_IL_T_1"

    make_read_orientation_model(sample="WES_IL_N_1").save(model_path)
    with pytest.raises(ValueError, match="not the tumor sample WES_IL_T_1"):
        filter_variants.load_read_orientation_model(model_path, MUTECT2_SEQC2_WES_VCF)


def contig_index_to_name_map(contigs_table) -> dict[int, str]:
    result = {}
    with open(contigs_table) as file:
        for line in file:
            contig, index = line.split()
            result[int(index)] = contig
    return result


def test_annotated_mafs_come_from_the_germline_model():
    # chr20 has MAF 0.3 in the tumor, except for a segment at its start, and 0.45 in the normal
    tumor_segments = [Segment("chr20", 1, 1000000, 0.1, 0.1), Segment("chr20", 1000001, 70000000, 0.3, 0.3)]
    tumor, normal = (
        SampleSegmentation("tumor", tumor_segments),
        SampleSegmentation("normal", [Segment("chr20", 1, 70000000, 0.45, 0.45)]),
    )
    dataset = filter_variants.make_annotated_dataset(
        SEQC2_WES_PLAIN_TEXT_DATA,
        MUTECT2_SEQC2_WES_VCF,
        contig_index_to_name_map(HG38_CONTIGS_TABLE),
        germline_model=GermlineModel(tumor, normal),
    )
    data = list(dataset.memory_mapped_data.generate())
    assert data
    positions = np.array([datum.get(Data.POSITION) for datum in data])
    mafs, normal_mafs = (np.array([datum.get(field) for datum in data]) for field in (Data.MAF, Data.NORMAL_MAF))
    # stored as float16
    assert np.allclose(mafs, np.where(positions <= 1000000, 0.1, 0.3), atol=1e-3)
    assert np.allclose(normal_mafs, 0.45, atol=1e-3)


def test_annotated_alt_orientation_counts_come_from_the_vcf():
    contig_names = contig_index_to_name_map(HG38_CONTIGS_TABLE)
    dataset = filter_variants.make_annotated_dataset(SEQC2_WES_PLAIN_TEXT_DATA, MUTECT2_SEQC2_WES_VCF, contig_names)
    vcf = cyvcf2.VCF(MUTECT2_SEQC2_WES_VCF)
    tumor_sample_index = get_tumor_sample_index(vcf)
    vcf_counts = {
        (v.CHROM, v.POS, allele_key(v.REF, v.ALT[0])): tumor_alt_orientation_counts(v, tumor_sample_index)
        + tumor_ref_orientation_counts(v, tumor_sample_index)
        for v in vcf
    }
    data = list(dataset.memory_mapped_data.generate())
    assert data
    fields = (Data.ALT_F1R2_COUNT, Data.ALT_F2R1_COUNT, Data.REF_F1R2_COUNT, Data.REF_F2R1_COUNT)
    counts = np.array([[datum.get(field) for field in fields] for datum in data])
    expected_counts = np.array(
        [
            vcf_counts[
                contig_names[datum.get(Data.CONTIG)],
                datum.get(Data.POSITION),
                allele_key(datum.get_ref_allele(), datum.get_alt_allele()),
            ]
            for datum in data
        ]
    )
    assert np.array_equal(counts, expected_counts, equal_nan=True)
    # Mutect2 annotates every variant, and most have alt reads of both orientations
    assert not np.isnan(counts).any() and np.mean(counts.min(axis=1) > 0) > 0.5


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


def test_germline_mode_records_a_missed_artifact_as_a_germline_call(tmp_path, monkeypatch):
    # with a truth label, an artifact that passes is a mistake whose call is the passing call type, which in germline
    # mode is germline.  It used to be recorded as somatic.
    datum = make_datum_with_alleles(0, 100, "C", "T")
    for data_field, value in ((Data.LABEL, Label.ARTIFACT), (Data.LOG_ALLELE_FREQUENCY, math.log(0.01))):
        datum.set(data_field, value)
    for data_field in (Data.MAF, Data.NORMAL_MAF):
        datum.set(data_field, 0.5)
    datum.set(Data.CACHED_ARTIFACT_LOGIT, 0.0)
    input_vcf = tmp_path / "input.vcf"
    input_vcf.write_text(VCF_HEADER + vcf_line("C", "T", pos=100) + "\n")

    mistaken_calls = []
    monkeypatch.setattr(EvaluationMetrics, "record_mistake", lambda self, result, call: mistaken_calls.append(call))
    model = PosteriorModel(-10, -10, device=torch.device("cpu"))
    nothing_is_an_error = {var_type: 1.0 for var_type in Variation}
    filter_variants.apply_filtering_to_vcf(
        str(input_vcf),
        str(tmp_path / "filtered.vcf"),
        {0: "chr1"},
        nothing_is_an_error,
        [Batch([datum])],
        model,
        SummaryWriter(str(tmp_path / "tensorboard")),
        germline_mode=True,
    )
    assert mistaken_calls == [Call.GERMLINE]


def test_filtering_needs_a_vcf_sorted_by_position(tmp_path):
    # the results of the data are read alongside the VCF's records in one pass, which needs the records in order
    data = []
    for position in (100, 200):
        datum = make_datum_with_alleles(0, position, "C", "T")
        datum.set(Data.LOG_ALLELE_FREQUENCY, math.log(0.01))
        for data_field in (Data.MAF, Data.NORMAL_MAF):
            datum.set(data_field, 0.5)
        datum.set(Data.CACHED_ARTIFACT_LOGIT, 0.0)
        data.append(datum)
    input_vcf = tmp_path / "input.vcf"
    input_vcf.write_text(VCF_HEADER + vcf_line("C", "T", pos=200) + "\n" + vcf_line("C", "T", pos=100) + "\n")

    model = PosteriorModel(-10, -10, device=torch.device("cpu"))
    with pytest.raises(ValueError, match="sorted by position"):
        filter_variants.apply_filtering_to_vcf(
            str(input_vcf),
            str(tmp_path / "filtered.vcf"),
            {0: "chr1"},
            {var_type: 0.5 for var_type in Variation},
            [Batch(data)],
            model,
            SummaryWriter(str(tmp_path / "tensorboard")),
        )


def test_posterior_info_holds_the_embeddings_and_the_twins_component():
    embeddings_be, simulated_be, components = torch.randn(3, 5), torch.randn(3, 5), [0, 2, 7]
    info_be = torch.vstack(
        [filter_variants.posterior_info_1d(*args) for args in zip(embeddings_be, simulated_be, components)]
    )
    assert info_be.shape == (3, 11)

    split_embeddings_be, split_simulated_be, split_components_b = filter_variants.split_posterior_info_be(info_be)
    assert torch.equal(split_embeddings_be, embeddings_be)
    assert torch.equal(split_simulated_be, simulated_be)
    assert split_components_b.tolist() == components

    # the posterior data store the INFO vector in float16, which keeps the component exactly
    float16_components_b = filter_variants.split_posterior_info_be(info_be.half().float())[2]
    assert float16_components_b.tolist() == components


def test_load_model_returns_model_in_eval_mode():
    model = load_model(PERMUTECT_V0_9_2_ARTIFACT_MODEL, device=torch.device("cpu"))
    assert not any(module.training for module in model.modules())


def test_posterior_data_artifact_logits_are_deterministic(tmp_path):
    model_path = tmp_path / "dropout-model.pt"
    save_dropout_model(model_path)
    model = load_model(model_path, device=torch.device("cpu"))

    dataset = filter_variants.make_annotated_dataset(
        SEQC2_WES_PLAIN_TEXT_DATA, MUTECT2_SEQC2_WES_VCF, contig_index_to_name_map(HG38_CONTIGS_TABLE)
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


def test_filtering_is_deterministic(tmp_path):
    """
    Two runs on the same inputs write the same records, since their random choices, such as the order of the data in
    test-time adaptation and in fitting the spectra, are seeded.  Each record's filters are sorted, rather than in the
    order of a set, which changes from run to run with Python's hash randomization and so would be the same in both runs
    of this one process.
    """
    pre_filtered_vcf = run_pre_permutect_filtering(MUTECT2_SEQC2_WES_VCF, HG38_CHR20_REFERENCE)

    def filtered_records(run: str) -> list[str]:
        output_vcf = tmp_path / f"{run}.vcf"
        filtering_args = parse_filtering_arguments(
            {
                constants.INPUT_NAME: pre_filtered_vcf.name,
                constants.TEST_DATASET_NAME: SEQC2_WES_PLAIN_TEXT_DATA,
                constants.CONTIGS_TABLE_NAME: HG38_CONTIGS_TABLE,
                constants.OUTPUT_NAME: str(output_vcf),
                constants.TENSORBOARD_DIR_NAME: str(tmp_path / f"{run}-tensorboard"),
                # at most two EM iterations after the warm start, to keep it quick
                constants.NUM_SPECTRUM_ITERATIONS_NAME: "2",
            }
        )
        filter_variants.main_without_parsing(filtering_args)
        with open(output_vcf) as file:
            return [line for line in file if not line.startswith("#")]

    first_records = filtered_records("first")
    assert filtered_records("second") == first_records

    filters = [record.split("\t")[6].split(";") for record in first_records]
    assert any(len(names) > 1 for names in filters)
    assert all(names == sorted(names) for names in filters)
