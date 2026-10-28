import math

import numpy as np
import pytest
import torch

from permutect import constants
from permutect.architecture.posterior_model import PosteriorModel
from permutect.architecture.posterior_model import best_error_probability_threshold
from permutect.architecture.posterior_model_priors import PosteriorModelPriors
from permutect.data.datum import Data
from permutect.test.architecture.test_posterior_model_priors import make_batch
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
    for call in (Call.SOMATIC, Call.SEQ_ERROR, Call.GERMLINE):
        assert log_posteriors_bc[0, call].item() == impossible

    negative_artifact_logit = make_batch([0.01], values | {Data.CACHED_ARTIFACT_LOGIT: -1})
    assert model.log_relative_posteriors_bc(negative_artifact_logit)[0, Call.ARTIFACT].item() == impossible

    model.priors.disable_context_dependent_snv_priors()
    totals_rrra = PosteriorModelPriors.initialize_snv_context_totals_rrra()
    model.priors.update_priors_m_step(torch.ones(len(Variation), len(Call)), totals_rrra, totals_rrra, 10)
    assert torch.all(model.priors.log_priors_vc[:, Call.GERMLINE] == impossible)


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
    loader = [
        make_batch([1e-3] * 200, {Data.VARIANT_TYPE: var_type, Data.ALT_COUNT: 1 + var_type}) for var_type in Variation
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
