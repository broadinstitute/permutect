import math

import numpy as np
import pytest
import torch

from permutect import constants
from permutect.architecture.posterior_model import PosteriorModel
from permutect.architecture.posterior_model import best_error_probability_threshold
from permutect.architecture.posterior_model_priors import PosteriorModelPriors
from permutect.data.batch import Batch
from permutect.data.count_binning import COUNT_BIN_SKIP
from permutect.data.count_binning import NUM_ALT_COUNT_BINS
from permutect.data.count_binning import alt_count_bin_name
from permutect.data.datum import Data
from permutect.read_orientation.f1r2_counts_reader import BASES
from permutect.read_orientation.read_orientation_model import CONTEXT_INDICES
from permutect.read_orientation.read_orientation_model import NUM_BASES
from permutect.read_orientation.read_orientation_model import NUM_CONTEXTS
from permutect.read_orientation.read_orientation_model import ReadOrientationModel
from permutect.segmentation.emissions import EmissionModel
from permutect.segmentation.germline_model import GermlineModel
from permutect.segmentation.germline_model import SampleSegmentation
from permutect.test.architecture.test_posterior_model_priors import make_batch
from permutect.test.host_reads import permutect_calls
from permutect.test.host_reads import permutect_device_tensors
from permutect.test.segmentation.test_germline_model import sample_segmentation
from permutect.utils.enums import Call
from permutect.utils.enums import Variation

# a device other than the default one, which on a machine without a GPU is the CPU
META = torch.device("meta")


def make_posterior_model(device) -> PosteriorModel:
    return PosteriorModel(variant_log_prior=-10, artifact_log_prior=-10, device=device)


def test_impossible_calls_use_the_shared_impossible_log_prob(monkeypatch):
    # no-germline mode, known artifacts, negative artifact logits and normal artifacts without normal alt reads all rule
    # out calls with constants.IMPOSSIBLE_LOG_PROB, so that changing it changes every one of them
    impossible = -5000.0
    monkeypatch.setattr(constants, "IMPOSSIBLE_LOG_PROB", impossible)
    model = PosteriorModel(variant_log_prior=-10, artifact_log_prior=-10, no_germline_mode=True, device="cpu")
    assert torch.all(model.priors.log_priors_vc[:, Call.GERMLINE] == impossible)

    values = {Data.ORIGINAL_DEPTH: 30, Data.ORIGINAL_ALT_COUNT: 5, Data.ORIGINAL_NORMAL_DEPTH: 30, Data.MAF: 0.5}
    known_artifact = make_batch([0.01], values | {Data.CACHED_ARTIFACT_LOGIT: math.inf})
    log_priors_bc, _, normal_log_lks_bc, log_posteriors_bc = model.log_posterior_and_ingredients(known_artifact)
    assert log_priors_bc[0, Call.GERMLINE].item() == pytest.approx(impossible, abs=1)  # the priors are normalized
    assert normal_log_lks_bc[0, Call.NORMAL_ARTIFACT].item() == impossible
    for call in set(Call) - {Call.ARTIFACT, Call.NORMAL_ARTIFACT}:
        assert log_posteriors_bc[0, call].item() == impossible

    negative_artifact_logit = make_batch([0.01], values | {Data.CACHED_ARTIFACT_LOGIT: -1})
    assert model.log_relative_posteriors_bc(negative_artifact_logit)[0, Call.ARTIFACT].item() == impossible

    model.priors.disable_context_dependent_snv_priors()
    totals_rrra = PosteriorModelPriors.initialize_snv_context_totals_rrra()
    model.priors.update_priors_m_step(torch.ones(len(Variation), len(Call)), totals_rrra, totals_rrra, 10)
    assert torch.all(model.priors.log_priors_vc[:, Call.GERMLINE] == impossible)


# depths and counts for which every call is plausible
PLAUSIBLE_VALUES = {
    Data.ORIGINAL_DEPTH: 30,
    Data.ORIGINAL_ALT_COUNT: 5,
    Data.ORIGINAL_NORMAL_DEPTH: 30,
    Data.ORIGINAL_NORMAL_ALT_COUNT: 2,
    Data.MAF: 0.5,
    Data.NORMAL_MAF: 0.5,
    Data.SEQ_ERROR_LOG_LK: -10,
}


def assert_no_nan(model: PosteriorModel, batch):
    # in the posteriors, or in the loss of learn_priors_and_spectra and its gradients for the spectra
    assert not torch.isnan(model.posterior_probabilities_bc(batch)).any()
    loss = -torch.mean(torch.logsumexp(model.log_relative_posteriors_bc(batch), dim=1))
    loss.backward()
    assert torch.isfinite(loss)
    for name, param in model.spectra.named_parameters():
        assert param.grad is None or not torch.isnan(param.grad).any(), name


def test_dataset_without_snvs_has_no_nan_posteriors():
    # learn_priors_and_spectra starts without context-dependent SNV priors, and its M step gives a dataset without SNVs
    # an SNV somatic log prior of log(0) = -inf.  Once context-dependent priors were enabled, the somatic log prior of
    # every indel was 0 * -inf = NaN, and so were its posteriors and the gradients of every spectrum.
    model = make_posterior_model(torch.device("cpu"))
    model.priors.disable_context_dependent_snv_priors()
    totals_vc = torch.ones(len(Variation), len(Call))
    totals_vc[Variation.SNV] = 0
    no_snvs_rrra = PosteriorModelPriors.initialize_snv_context_totals_rrra()
    model.priors.update_priors_m_step(totals_vc, no_snvs_rrra, no_snvs_rrra, ignored_to_non_ignored_ratio=10)
    assert model.priors.log_priors_vc[Variation.SNV, Call.SOMATIC] == -math.inf

    model.priors.enable_context_dependent_snv_priors()
    assert_no_nan(model, make_batch([0.01, 0.1], PLAUSIBLE_VALUES | {Data.VARIANT_TYPE: Variation.INSERTION}))


# minor allele fractions of 0 or 1, eg from copy-neutral LOH, and population allele frequencies of 0 or 1 used to give
# 0 * log(0) = NaN
@pytest.mark.parametrize(
    "allele_frequency, values",
    [(0.01, {Data.MAF: 0.0}), (0.01, {Data.MAF: 1.0}), (0.01, {Data.NORMAL_MAF: 0.0}), (0.01, {Data.NORMAL_MAF: 1.0})]
    + [(0.0, {}), (1.0, {})],
)
def test_extreme_allele_fractions_have_no_nan_posteriors(allele_frequency, values):
    model = make_posterior_model(torch.device("cpu"))
    assert_no_nan(model, make_batch([allele_frequency] * 2, PLAUSIBLE_VALUES | values))


def test_contamination_explains_alt_fractions_near_the_contamination():
    # in a tumor contaminated at 5%, 3 alt reads of 100 at a site of population allele frequency 0.1 are close to the 2.5
    # of a het contaminant, but 30 alt reads are not.  Without contamination, alt reads can't come from a contaminant.
    contaminated = GermlineModel(
        SampleSegmentation(emission_model=EmissionModel(contamination=0.05)), SampleSegmentation(), 0.05, 0.005
    )

    def contamination_probability(germline_model, alt_count: int) -> float:
        model = PosteriorModel(-10, -10, device="cpu", germline_model=germline_model)
        # the artifact logit rules out artifacts, and normal artifacts need alt reads in the normal
        values = {
            Data.ORIGINAL_DEPTH: 100,
            Data.ORIGINAL_ALT_COUNT: alt_count,
            Data.MAF: 0.5,
            Data.SEQ_ERROR_LOG_LK: -30,
            Data.CACHED_ARTIFACT_LOGIT: -5,
        }
        return model.posterior_probabilities_bc(make_batch([0.1], values))[0, Call.CONTAMINATION].item()

    assert contamination_probability(contaminated, 3) > 0.99
    assert contamination_probability(contaminated, 30) < 1e-6
    assert contamination_probability(None, 3) < 1e-6


def test_the_spectra_and_priors_move_with_the_posterior_model():
    model = make_posterior_model(META)
    assert model.spectra._device == META
    assert all(tensor.device == META for tensor in model.state_dict().values())

    # the priors' nontrivial contexts, a constant that isn't saved with the model, move with it too
    moved = make_posterior_model(torch.device("cpu")).to(META)
    assert moved.priors.NONTRIVIAL_CONTEXTS_rrra.device == META
    assert not any("NONTRIVIAL_CONTEXTS" in name for name in moved.state_dict())


ACCELERATOR = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else None)


@pytest.mark.skipif(ACCELERATOR is None, reason="needs CUDA or MPS")
def test_posterior_probabilities_on_an_accelerator():
    values = {
        Data.ORIGINAL_DEPTH: 30,
        Data.ORIGINAL_ALT_COUNT: 5,
        Data.ORIGINAL_NORMAL_DEPTH: 30,
        Data.ORIGINAL_NORMAL_ALT_COUNT: 0,
        Data.MAF: 0.5,
        Data.NORMAL_MAF: 0.5,
        Data.SEQ_ERROR_LOG_LK: -10,
    }
    batch = make_batch([0.01, 0.1, 0.5], values)

    posteriors_bc = {}
    for device in (torch.device("cpu"), torch.device(ACCELERATOR)):
        model = make_posterior_model(device)
        posteriors_bc[device.type] = model.posterior_probabilities_bc(batch.copy_to(device=device, dtype=torch.float32))
    assert posteriors_bc[ACCELERATOR].device.type == ACCELERATOR
    torch.testing.assert_close(posteriors_bc[ACCELERATOR].cpu(), posteriors_bc["cpu"])


def test_best_threshold_maximizes_the_expected_f_score():
    # with error probabilities 0.1, 0.2, 0.9 and 0.9, passing the first two expects 1.7 of the 1.9 non-errors and 0.3
    # errors to pass, the best F1 score.  Weighing recall three times as much as precision passes everything.
    probs = [0.9, 0.1, 0.9, 0.2]
    assert best_error_probability_threshold(probs) == 0.2
    assert best_error_probability_threshold(probs, recall_weight=3.0) == 0.9
    assert best_error_probability_threshold([]) == 0


class FigureRecorder:
    # a stand-in for a SummaryWriter that keeps each figure by tag
    def __init__(self):
        self.figures = {}

    def add_figure(self, tag, figure, global_step=None):
        self.figures[tag] = figure


def test_theoretical_roc_has_sensitivity_on_x(monkeypatch):
    # the theoretical ROC plots used to put precision on x, while their axes said sensitivity, like the measured ROC
    model = make_posterior_model(torch.device("cpu"))
    rng = np.random.default_rng(seed=1)
    probs_by_type = {var_type: rng.beta(0.5, 2.0, size=200).astype(np.float32) for var_type in Variation}
    # like posterior data, which have no reads and so an ALT_COUNT of 0.  Each variant type's original alt counts are in a
    # bin other than the first.  There are more variant types than such bins, so some types share a bin.
    alt_count_bins = {var_type: 1 + var_type % (NUM_ALT_COUNT_BINS - 1) for var_type in Variation}
    loader = [
        make_batch(
            [1e-3] * 200,
            {Data.VARIANT_TYPE: var_type, Data.ORIGINAL_ALT_COUNT: 1 + COUNT_BIN_SKIP * alt_count_bins[var_type]},
        )
        for var_type in Variation
    ]
    batch_probs = iter(torch.from_numpy(probs_by_type[var_type]) for var_type in Variation)
    monkeypatch.setattr(model, "error_probabilities_b", lambda batch, germline_mode=False: next(batch_probs))

    summary_writer = FigureRecorder()
    thresholds = model.calculate_probability_thresholds(loader, summary_writer)
    for var_type in Variation:
        assert thresholds[var_type] == best_error_probability_threshold(probs_by_type[var_type].tolist())

    assert len(summary_writer.figures) == 2
    for tag, figure in summary_writer.figures.items():
        curves = [line for axis in figure.axes for line in axis.get_lines() if len(line.get_xdata()) > 1]
        assert curves, tag
        for curve in curves:
            # raising the threshold passes more variants: sensitivity rises and precision falls
            sensitivity, precision = np.array(curve.get_xdata()), np.array(curve.get_ydata())
            assert np.all(np.diff(sensitivity) >= 0) and np.all(np.diff(precision) <= 1e-6), tag
            assert sensitivity[-1] > sensitivity[0] and not math.isnan(precision[0]), tag

    # each variant type's subplot of the ROC by alt count has a curve only for the bin of its original alt counts.  The
    # alt counts used to be ALT_COUNT, which is 0 for posterior data, so every curve was in the first bin.
    by_alt_count_figure = summary_writer.figures["theoretical ROC by variant type and alt count "]
    for var_type, axis in zip(Variation, by_alt_count_figure.axes):
        curve_bins = [line.get_label() for line in axis.get_lines() if len(line.get_xdata()) > 1]
        assert curve_bins == [alt_count_bin_name(alt_count_bins[var_type])], var_type.name


def posterior_loader(num_batches: int) -> list:
    values = {Data.ORIGINAL_DEPTH: 30, Data.ORIGINAL_ALT_COUNT: 5, Data.ORIGINAL_NORMAL_DEPTH: 30, Data.MAF: 0.5}
    return [make_batch([0.01, 0.1, 0.5], values) for _ in range(num_batches)]


@pytest.mark.parametrize("posterior_pass", ["learn_priors_and_spectra", "calculate_probability_thresholds"])
def test_posterior_passes_read_no_values_for_each_batch(monkeypatch, posterior_pass):
    # copying values to the CPU makes the CPU wait for the GPU, so the passes over the posterior data copy values for
    # each iteration or at the end, not for every batch.  Fitting takes one warm-start epoch and exactly two EM
    # iterations however many batches there are, and each M step copies the context-dependent SNV priors' totals once.
    def num_copies(num_batches: int) -> int:
        model = make_posterior_model(torch.device("cpu"))
        with permutect_calls(monkeypatch, "item", "tolist", "cpu") as calls:
            if posterior_pass == "learn_priors_and_spectra":
                model.learn_priors_and_spectra(
                    posterior_loader(num_batches), 2, 10, num_warmup_steps=1, convergence_threshold=0
                )
            else:
                model.calculate_probability_thresholds(posterior_loader(num_batches))
        return len(calls)

    assert num_copies(1) == num_copies(3)


def varied_batches() -> list:
    # batches of variants that differ in alt count and artifact logit, so that the spectra have something to fit
    values = PLAUSIBLE_VALUES | {Data.ORIGINAL_DEPTH: 60}
    return [
        make_batch([1e-6, 1e-4], values | {Data.ORIGINAL_ALT_COUNT: alt, Data.CACHED_ARTIFACT_LOGIT: logit})
        for alt in (2, 4, 8, 15, 30)
        for logit in (-3.0, 0.0, 3.0)
    ]


def test_fitting_stops_once_the_evidence_stops_changing():
    model = make_posterior_model("cpu")
    iterations, converged = model.learn_priors_and_spectra(varied_batches(), 50, 10, num_warmup_steps=30)
    assert converged and iterations < 50

    # with a threshold of 0 it never converges, and stops at the maximum number of EM iterations
    model = make_posterior_model("cpu")
    assert model.learn_priors_and_spectra(varied_batches(), 3, 10, num_warmup_steps=30, convergence_threshold=0) == (
        3,
        False,
    )


def test_calculating_thresholds_keeps_no_autograd_graphs(monkeypatch):
    # the error probabilities of every batch are kept until the thresholds are found, and with gradients each would keep
    # its batch's autograd graph
    model = make_posterior_model(torch.device("cpu"))
    error_probabilities_b, error_probs = model.error_probabilities_b, []

    def recording_error_probabilities_b(batch, germline_mode=False):
        error_probs.append(error_probabilities_b(batch, germline_mode))
        return error_probs[-1]

    monkeypatch.setattr(model, "error_probabilities_b", recording_error_probabilities_b)
    model.calculate_probability_thresholds(posterior_loader(3))
    assert len(error_probs) == 3 and not any(probs.requires_grad for probs in error_probs)


def make_read_orientation_model(damage_mean: float = 0.1, sample: str = "TUMOR") -> ReadOrientationModel:
    """
    A model of a sample in which damage ACG -> ATG, the deamination of the C of ACG on whichever strand reads ACG, has
    rates Beta(2, 18) for the default mean 0.1, and every other physical process has rates of mean 1e-5
    """
    pseudocounts_ca2 = torch.tensor([1e-5 * 1000, (1 - 1e-5) * 1000], dtype=torch.float64)
    pseudocounts_ca2 = pseudocounts_ca2.expand(NUM_CONTEXTS, NUM_BASES, 2).clone()
    pseudocounts_ca2[CONTEXT_INDICES["ACG"], BASES.index("T")] = torch.tensor([damage_mean, 1 - damage_mean]) * 20
    return ReadOrientationModel(sample, pseudocounts_ca2)


def make_snv_batch(snvs: list[tuple[str, str, int, int]], values: dict) -> Batch:
    """
    A batch of SNVs, each given by its reference context, its alt base, and its counts of alt reads in F1R2 and F2R1
    orientation, with the given values of other data fields and a population allele frequency that rules out germline
    variants.  The haplotypes are the context and the context with the alt base.
    """
    batch = make_batch([1e-8] * len(snvs), values | {Data.VARIANT_TYPE: Variation.SNV})
    for b, (context, alt_base, alt_f1r2_count, alt_f2r1_count) in enumerate(snvs):
        ref_haplotype = [BASES.index(base) for base in context]
        alt_haplotype = [ref_haplotype[0], BASES.index(alt_base), ref_haplotype[2]]
        haplotypes_start = Data.HAPLOTYPES_START_IDX
        batch.int_tensor[b, haplotypes_start : haplotypes_start + 6] = torch.tensor(ref_haplotype + alt_haplotype)
        batch.float_tensor[b, Data.ALT_F1R2_COUNT.idx] = alt_f1r2_count
        batch.float_tensor[b, Data.ALT_F2R1_COUNT.idx] = alt_f2r1_count
    return batch


# 6 alt reads of 100, too many for Mutect2's sequencing error, at a site whose normal has none, so that it can't be a
# normal artifact, with an artifact logit that doesn't favor an artifact or a non-artifact
SNV_VALUES = PLAUSIBLE_VALUES | {
    Data.ORIGINAL_DEPTH: 100,
    Data.ORIGINAL_ALT_COUNT: 6,
    Data.ORIGINAL_NORMAL_ALT_COUNT: 0,
    Data.SEQ_ERROR_LOG_LK: -30,
    Data.CACHED_ARTIFACT_LOGIT: 0,
    # the 94 ref reads, half of them in each orientation
    Data.REF_F1R2_COUNT: 47,
    Data.REF_F2R1_COUNT: 47,
}


def test_seq_error_is_mutect2s_without_a_read_orientation_model():
    # and given one, damage changes SEQ_ERROR's likelihood alone
    batch = make_snv_batch([("ACG", "T", 6, 0)], SNV_VALUES)
    without_model = make_posterior_model("cpu")
    with_model = PosteriorModel(-10, -10, device="cpu", read_orientation_model=make_read_orientation_model())
    log_priors_bc, spectra_bc, normal_bc, _ = without_model.log_posterior_and_ingredients(batch)
    assert spectra_bc[0, Call.SEQ_ERROR].item() == SNV_VALUES[Data.SEQ_ERROR_LOG_LK]

    model_log_priors_bc, model_spectra_bc, model_normal_bc, _ = with_model.log_posterior_and_ingredients(batch)
    other_calls = [call for call in Call if call != Call.SEQ_ERROR]
    assert torch.equal(spectra_bc[:, other_calls], model_spectra_bc[:, other_calls])
    assert torch.equal(normal_bc, model_normal_bc) and torch.equal(log_priors_bc, model_log_priors_bc)
    assert model_spectra_bc[0, Call.SEQ_ERROR].item() > spectra_bc[0, Call.SEQ_ERROR].item() + 10


def test_alt_reads_of_one_orientation_from_a_damaging_process_are_seq_errors():
    model = PosteriorModel(-10, -10, device="cpu", read_orientation_model=make_read_orientation_model())
    snvs = [
        ("ACG", "T", 6, 0),  # ACG -> ATG of the + strand, the model's damage, which puts alt reads in F1R2 orientation
        ("CGT", "A", 0, 6),  # ACG -> ATG of the - strand, which reads ACG at a site of CGT, in F2R1 orientation
        ("ACG", "T", 3, 3),  # alt reads of both orientations
        ("ACG", "T", 0, 6),  # CGT -> CAT of the - strand, which the model makes rare
        ("ACA", "T", 6, 0),  # ACA -> ATA of the + strand, rare too
    ]
    batch = make_snv_batch(snvs, SNV_VALUES)
    seq_error_b = model.posterior_probabilities_bc(batch)[:, Call.SEQ_ERROR]
    assert seq_error_b[0].item() > 0.99
    assert seq_error_b[1].item() == pytest.approx(seq_error_b[0].item())
    # alt reads of both orientations need damage of the - strand too, at a rate around 3 / 47, which is rare
    assert seq_error_b[2].item() < 0.2
    assert torch.all(seq_error_b[3:] < 0.01)


def test_capture_bias_toward_the_alt_reads_orientation_makes_them_less_like_damage():
    # all 6 alt reads F1R2, as from the model's damage, with ref reads of both orientations, with 90 of the 94 F1R2, as
    # capture bias can make them, and without ref orientation counts, which leaves Mutect2's sequencing error alone
    model = PosteriorModel(-10, -10, device="cpu", read_orientation_model=make_read_orientation_model())
    batch = make_snv_batch([("ACG", "T", 6, 0)] * 3, SNV_VALUES)
    batch.float_tensor[1, Data.REF_F1R2_COUNT.idx], batch.float_tensor[1, Data.REF_F2R1_COUNT.idx] = 90, 4
    batch.float_tensor[2, Data.REF_F1R2_COUNT.idx] = batch.float_tensor[2, Data.REF_F2R1_COUNT.idx] = math.nan
    seq_error_b = model.spectra.seq_error_log_likelihoods_b(batch)
    assert seq_error_b[0] > seq_error_b[1] + math.log(10)
    assert seq_error_b[2].item() == SNV_VALUES[Data.SEQ_ERROR_LOG_LK]


def test_damage_needs_snvs_with_orientation_counts():
    model = PosteriorModel(-10, -10, device="cpu", read_orientation_model=make_read_orientation_model())
    batch = make_snv_batch([("ACG", "T", 6, 0)] * 4, SNV_VALUES)
    batch.float_tensor[1, Data.ALT_F1R2_COUNT.idx] = math.nan  # not annotated from the VCF
    batch.float_tensor[2, Data.REF_F2R1_COUNT.idx] = math.nan
    batch.int_tensor[3, Data.VARIANT_TYPE.idx] = Variation.INSERTION
    seq_error_b = model.spectra.seq_error_log_likelihoods_b(batch)
    assert seq_error_b[0].item() > SNV_VALUES[Data.SEQ_ERROR_LOG_LK] + 10
    assert torch.all(seq_error_b[1:] == SNV_VALUES[Data.SEQ_ERROR_LOG_LK])
    assert torch.equal(seq_error_b, model.spectra.seq_error_log_likelihoods_b(batch, with_damage=True))
    assert torch.all(model.spectra.seq_error_log_likelihoods_b(batch, with_damage=False) == -30)
    assert_no_nan(model, batch)


def test_learning_leaves_the_germline_and_read_orientation_models_unchanged():
    # the germline model was fit to the pileups of common SNPs and the read orientation model to F1R2 counts, so fitting
    # the spectra to the variants mustn't change them
    germline_model = GermlineModel(sample_segmentation("tumor"), sample_segmentation("normal"), 0.03, 0.002)
    read_orientation_model = make_read_orientation_model()
    model = PosteriorModel(
        -10, -10, device="cpu", germline_model=germline_model, read_orientation_model=read_orientation_model
    )

    def state() -> dict:
        germline_state = {"germline " + name: tensor.clone() for name, tensor in germline_model.state_dict().items()}
        orientation_state = {
            "orientation " + name: buffer.clone() for name, buffer in read_orientation_model.named_buffers()
        }
        return germline_state | orientation_state

    before = state()
    loader = [make_batch([0.01, 0.2, 0.5], PLAUSIBLE_VALUES | {Data.ORIGINAL_ALT_COUNT: alt}) for alt in (2, 15, 29)]
    loader.append(make_snv_batch([("ACG", "T", 6, 0)] * 4, SNV_VALUES | {Data.CACHED_ARTIFACT_LOGIT: 3}))
    model.learn_priors_and_spectra(loader, num_iterations=2, ignored_to_non_ignored_ratio=10, learning_rate=0.1)
    after = state()
    assert before.keys() == after.keys() and all(torch.equal(before[name], after[name]) for name in before)
    assert not any(parameter.requires_grad for parameter in germline_model.parameters())


# the deletion of one repeat unit of a 12-base STR, at depth 100 with 6 alt reads, with the values of SNV_VALUES
STR_INDEL_VALUES = SNV_VALUES | {
    Data.VARIANT_TYPE: Variation.DELETION,
    Data.REPEAT_UNIT_LENGTH: 1,
    Data.REPEAT_LENGTH: 12,
    Data.REPEAT_COUNT_CHANGE: -1,
}


@pytest.mark.parametrize(
    "values, expected_probability",
    [
        ({}, 1),  # a low allele fraction of a one-unit change of a long STR
        ({Data.ORIGINAL_ALT_COUNT: 40}, 0),  # too high an allele fraction
        ({Data.REPEAT_LENGTH: 7}, 0),  # too short an STR
        ({Data.REPEAT_COUNT_CHANGE: -2}, 0),  # two repeat units
        ({Data.REPEAT_LENGTH: math.nan, Data.REPEAT_COUNT_CHANGE: math.nan}, 0),  # not annotated from the VCF
        ({Data.VARIANT_TYPE: Variation.SNV}, 0),
    ],
)
def test_one_unit_indels_of_long_strs_at_low_allele_fractions_are_slippage(values, expected_probability):
    model = make_posterior_model("cpu")
    batch = make_batch([1e-8], STR_INDEL_VALUES | values)
    slippage_probability = model.posterior_probabilities_bc(batch)[0, Call.SLIPPAGE].item()
    assert slippage_probability == pytest.approx(expected_probability, abs=0.01)
    assert_no_nan(model, batch)


def test_slippage_is_a_sequencing_error():
    # with the prior and the normal's likelihood of a sequencing error, and no artifact logit
    model = make_posterior_model("cpu")
    batch = make_batch([1e-8, 1e-8], STR_INDEL_VALUES)
    batch.float_tensor[1, Data.CACHED_ARTIFACT_LOGIT.idx] = 3
    log_priors_bc, _, normal_log_lks_bc, log_posteriors_bc = model.log_posterior_and_ingredients(batch)
    assert torch.equal(log_priors_bc[:, Call.SLIPPAGE], log_priors_bc[:, Call.SEQ_ERROR])
    assert torch.equal(normal_log_lks_bc[:, Call.SLIPPAGE], normal_log_lks_bc[:, Call.SEQ_ERROR])
    assert log_posteriors_bc[0, Call.SLIPPAGE].item() == log_posteriors_bc[1, Call.SLIPPAGE].item()


def test_known_artifacts_rule_out_seq_errors_from_damage_and_slippage():
    model = PosteriorModel(-10, -10, device="cpu", read_orientation_model=make_read_orientation_model())
    known_artifact = {Data.CACHED_ARTIFACT_LOGIT: math.inf}
    damage = make_snv_batch([("ACG", "T", 6, 0)], SNV_VALUES | known_artifact)
    slippage = make_batch([1e-8], STR_INDEL_VALUES | known_artifact)
    for batch, call in ((damage, Call.SEQ_ERROR), (slippage, Call.SLIPPAGE)):
        assert model.log_relative_posteriors_bc(batch)[0, call].item() == constants.IMPOSSIBLE_LOG_PROB


def test_posteriors_copy_no_python_values_to_the_device(monkeypatch):
    # the posteriors are computed for every batch, so a tensor of Python values made on the device for each one, such as
    # the indel variant types used to be, makes the CPU wait for the GPU every batch.  Indels of long STRs can be
    # slippage, which takes all of the code that made them.
    model = make_posterior_model("cpu")
    batch = make_batch([1e-8, 1e-8], STR_INDEL_VALUES)
    with permutect_device_tensors(monkeypatch) as devices:
        model.log_relative_posteriors_bc(batch)
    assert devices == []
