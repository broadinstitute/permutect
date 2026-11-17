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
from permutect.read_orientation.read_orientation_model import CANONICAL_CONTEXTS
from permutect.read_orientation.read_orientation_model import HOM_REF
from permutect.read_orientation.read_orientation_model import STATES
from permutect.read_orientation.read_orientation_model import ContextPrior
from permutect.read_orientation.read_orientation_model import ReadOrientationModel
from permutect.read_orientation.read_orientation_model import flat_prior
from permutect.read_orientation.read_orientation_model import reverse_complement
from permutect.segmentation.emissions import EmissionModel
from permutect.segmentation.germline_model import GermlineModel
from permutect.segmentation.germline_model import SampleSegmentation
from permutect.test.architecture.test_posterior_model_priors import make_batch
from permutect.test.host_reads import permutect_calls
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


def test_spectra_use_the_posterior_model_device():
    model = make_posterior_model(META)
    assert model.spectra._device == META
    assert all(tensor.device == META for tensor in model.state_dict().values())


def test_nontrivial_contexts_move_with_the_posterior_model():
    model = make_posterior_model(torch.device("cpu")).to(META)
    assert model.priors.NONTRIVIAL_CONTEXTS_rrra.device == META

    # it's a constant, so it's not saved with the model
    assert not any("NONTRIVIAL_CONTEXTS" in name for name in model.state_dict())


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


def original_best_threshold(artifact_probs, recall_weight: float = 1.0):
    # the threshold search of plotting.get_theoretical_roc_data before it was moved, for checking that it's unchanged
    beta_sqr = recall_weight**2
    artifact_probs.sort(key=lambda p: p)
    num_calls = len(artifact_probs)
    total_artifact = sum([prob for prob in artifact_probs]) + 0.0001
    total_non_artifact = num_calls - total_artifact + 0.0002
    art_found, non_art_found = total_artifact, 0
    best_threshold, best_harmonic_mean = (0, 1, 0), 0
    for prob in artifact_probs:
        art_found -= prob
        non_art_found += 1 - prob
        tp = non_art_found
        fp = total_artifact - art_found
        sensitivity = tp / total_non_artifact
        precision = tp / (tp + fp)
        harmonic_mean = (1 + beta_sqr) * sensitivity * precision / (sensitivity + (beta_sqr * precision) + 0.0001)
        if harmonic_mean > best_harmonic_mean:
            best_harmonic_mean = harmonic_mean
            best_threshold = (prob, precision, sensitivity)
    return best_threshold[0]


def test_best_threshold_is_unchanged():
    # filter_variants uses this threshold, so moving it out of the plotting code must not change it at all
    rng = np.random.default_rng(seed=0)
    for probs in (rng.uniform(size=1000), rng.beta(0.2, 2.0, size=5000), rng.beta(2.0, 0.2, size=50), []):
        for recall_weight in (0.5, 1.0, 2.0):
            expected = original_best_threshold(list(probs), recall_weight)
            assert best_error_probability_threshold(list(probs), recall_weight) == expected


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
    # each iteration or at the end, not for every batch
    def num_copies(num_batches: int) -> int:
        model = make_posterior_model(torch.device("cpu"))
        # the context-dependent priors' M step fits a PyMC model, which is slow and doesn't depend on the batches
        monkeypatch.setattr(model.priors, "enable_context_dependent_snv_priors", lambda: None)
        with permutect_calls(monkeypatch, "item", "tolist", "cpu") as calls:
            if posterior_pass == "learn_priors_and_spectra":
                model.learn_priors_and_spectra(posterior_loader(num_batches), 2, ignored_to_non_ignored_ratio=10)
            else:
                model.calculate_probability_thresholds(posterior_loader(num_batches))
        return len(calls)

    assert num_copies(1) == num_copies(3)


def test_learning_leaves_the_germline_model_unchanged():
    # the germline model was fit to the pileups of common SNPs, so fitting the spectra to the variants mustn't change it
    germline_model = GermlineModel(sample_segmentation("tumor"), sample_segmentation("normal"), 0.03, 0.002)
    model = PosteriorModel(-10, -10, device="cpu", germline_model=germline_model)
    before = {name: tensor.clone() for name, tensor in germline_model.state_dict().items()}
    loader = [make_batch([0.01, 0.2, 0.5], PLAUSIBLE_VALUES | {Data.ORIGINAL_ALT_COUNT: alt}) for alt in (2, 15, 29)]
    model.learn_priors_and_spectra(loader, num_iterations=2, ignored_to_non_ignored_ratio=10, learning_rate=0.1)
    after = germline_model.state_dict()
    assert all(torch.equal(before[name], after[name]) for name in before)
    assert not any(parameter.requires_grad for parameter in germline_model.parameters())


def make_read_orientation_model(artifact_prior: float = 0.01, sample: str = "TUMOR") -> ReadOrientationModel:
    """
    A model of a sample's read orientation artifacts in which an F1R2 artifact of ACG's ref C to T, as from deamination
    of one strand, has the given prior, and every other artifact is very unlikely
    """
    priors = {}
    for context in CANONICAL_CONTEXTS:
        prior_k = 1e-8 * flat_prior(BASES.index(context[1]))
        if context == "ACG":
            prior_k[STATES.index("F1R2_T")] = artifact_prior
        prior_k[HOM_REF] = 1 - prior_k.sum() + prior_k[HOM_REF]
        prior = ContextPrior(context, prior_k, num_sites=1000, num_alt_sites=100)
        priors[context], priors[reverse_complement(context)] = prior, prior.reverse_complement()
    return ReadOrientationModel(sample, priors)


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


# 6 alt reads of 100, too many for sequencing error, at a site whose normal has none, so that it can't be a normal
# artifact, with an artifact logit that doesn't favor an artifact or a non-artifact
SNV_VALUES = PLAUSIBLE_VALUES | {
    Data.ORIGINAL_DEPTH: 100,
    Data.ORIGINAL_ALT_COUNT: 6,
    Data.ORIGINAL_NORMAL_ALT_COUNT: 0,
    Data.SEQ_ERROR_LOG_LK: -30,
    Data.CACHED_ARTIFACT_LOGIT: 0,
}


def test_read_orientation_artifacts_are_impossible_without_a_model():
    # and the read orientation model changes the likelihoods of no other call
    batch = make_snv_batch([("ACG", "T", 6, 0)], SNV_VALUES)
    without_model = make_posterior_model("cpu")
    with_model = PosteriorModel(-10, -10, device="cpu", read_orientation_model=make_read_orientation_model())
    log_priors_bc, spectra_bc, normal_bc, log_posteriors_bc = without_model.log_posterior_and_ingredients(batch)
    assert log_priors_bc[0, Call.READ_ORIENTATION].item() == pytest.approx(constants.IMPOSSIBLE_LOG_PROB, abs=1)
    assert spectra_bc[0, Call.READ_ORIENTATION].item() == 0
    assert without_model.posterior_probabilities_bc(batch)[0, Call.READ_ORIENTATION].item() == 0

    _, model_spectra_bc, model_normal_bc, _ = with_model.log_posterior_and_ingredients(batch)
    other_calls = [call for call in Call if call != Call.READ_ORIENTATION]
    assert torch.equal(spectra_bc[:, other_calls], model_spectra_bc[:, other_calls])
    assert torch.equal(normal_bc, model_normal_bc)


def test_alt_reads_of_one_orientation_in_an_artifact_context_are_a_read_orientation_artifact():
    model = PosteriorModel(-10, -10, device="cpu", read_orientation_model=make_read_orientation_model())
    snvs = [
        ("ACG", "T", 6, 0),  # an F1R2 C to T in ACG, the model's artifact
        ("CGT", "A", 0, 6),  # the same on the other strand: an F2R1 G to A in CGT, the reverse complement of ACG
        ("ACG", "T", 3, 3),  # alt reads of both orientations
        ("ACG", "T", 0, 6),  # an F2R1 C to T in ACG, which the model makes very unlikely
        ("ACA", "T", 6, 0),  # an F1R2 C to T in another context
    ]
    batch = make_snv_batch(snvs, SNV_VALUES)
    read_orientation_b = model.posterior_probabilities_bc(batch)[:, Call.READ_ORIENTATION]
    assert read_orientation_b[0].item() > 0.99
    assert read_orientation_b[1].item() == pytest.approx(read_orientation_b[0].item())
    # alt reads of both orientations are thousands of times less likely for the artifact, but its prior here is hundreds
    # of times the somatic prior
    assert read_orientation_b[2].item() < 0.1
    assert torch.all(read_orientation_b[3:] < 0.01)


def test_read_orientation_artifacts_need_snvs_with_alt_orientation_counts():
    model = PosteriorModel(-10, -10, device="cpu", read_orientation_model=make_read_orientation_model())
    batch = make_snv_batch([("ACG", "T", 6, 0)] * 3, SNV_VALUES)
    batch.float_tensor[1, Data.ALT_F1R2_COUNT.idx] = math.nan  # not annotated from the VCF
    batch.int_tensor[2, Data.VARIANT_TYPE.idx] = Variation.INSERTION
    log_priors_bc, spectra_bc, _, _ = model.log_posterior_and_ingredients(batch)
    assert log_priors_bc[0, Call.READ_ORIENTATION].item() > -10
    impossible_b = log_priors_bc[1:, Call.READ_ORIENTATION]
    assert torch.allclose(impossible_b, torch.full_like(impossible_b, constants.IMPOSSIBLE_LOG_PROB), atol=1)
    assert torch.all(spectra_bc[1:, Call.READ_ORIENTATION] == 0)
    assert_no_nan(model, batch)


@pytest.mark.parametrize("overlap", [1e-6, 0.25, 1 - 1e-6])
def test_read_orientation_reads_likelihood_interpolates_between_non_artifact_and_artifact(overlap):
    # relative to a non-artifact's: 1 if the reads of a read orientation artifact look like a non-artifact's, the
    # relative likelihood exp(artifact logit) of an artifact's if they look like an artifact's, or a mixture of the two
    model = PosteriorModel(-10, -10, device="cpu", read_orientation_model=make_read_orientation_model())
    with torch.no_grad():
        model.read_orientation_artifact_overlap = torch.tensor(overlap)
    artifact_logits_b = torch.tensor([-5.0, 0.0, 3.0], dtype=torch.float64)
    result_b = model.read_orientation_reads_log_likelihoods_b(artifact_logits_b)
    expected_b = torch.log(overlap * torch.exp(artifact_logits_b) + 1 - overlap)
    assert torch.allclose(result_b, expected_b, atol=1e-6)


def test_known_artifacts_rule_out_read_orientation_artifacts():
    # but negative artifact logits, which rule out other artifacts, don't
    model = PosteriorModel(-10, -10, device="cpu", read_orientation_model=make_read_orientation_model())
    batch = make_snv_batch([("ACG", "T", 6, 0)] * 2, SNV_VALUES)
    batch.float_tensor[0, Data.CACHED_ARTIFACT_LOGIT.idx] = math.inf
    batch.float_tensor[1, Data.CACHED_ARTIFACT_LOGIT.idx] = -5
    log_posteriors_bc = model.log_relative_posteriors_bc(batch)
    assert log_posteriors_bc[0, Call.READ_ORIENTATION].item() == constants.IMPOSSIBLE_LOG_PROB
    assert log_posteriors_bc[1, Call.ARTIFACT].item() == constants.IMPOSSIBLE_LOG_PROB
    assert log_posteriors_bc[1, Call.READ_ORIENTATION].item() > -50


def test_learning_fits_the_read_orientation_artifact_overlap_and_leaves_the_read_orientation_model_unchanged():
    read_orientation_model = make_read_orientation_model()
    model = PosteriorModel(-10, -10, device="cpu", read_orientation_model=read_orientation_model)
    before = {name: buffer.clone() for name, buffer in read_orientation_model.named_buffers()}
    # read orientation artifacts whose reads look like artifacts' to the artifact model
    loader = [make_snv_batch([("ACG", "T", 6, 0)] * 4, SNV_VALUES | {Data.CACHED_ARTIFACT_LOGIT: 3})]
    model.learn_priors_and_spectra(loader, num_iterations=2, ignored_to_non_ignored_ratio=10, learning_rate=0.1)
    assert model.read_orientation_artifact_overlap.item() > 0.5
    assert all(torch.equal(before[name], buffer) for name, buffer in read_orientation_model.named_buffers())


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


def test_known_artifacts_rule_out_slippage():
    model = make_posterior_model("cpu")
    batch = make_batch([1e-8], STR_INDEL_VALUES | {Data.CACHED_ARTIFACT_LOGIT: math.inf})
    assert model.log_relative_posteriors_bc(batch)[0, Call.SLIPPAGE].item() == constants.IMPOSSIBLE_LOG_PROB
