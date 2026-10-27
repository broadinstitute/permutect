import pytest
import torch

from permutect.architecture.posterior_model import PosteriorModel
from permutect.data.datum import Data
from permutect.test.architecture.test_posterior_model_priors import make_batch

# a device other than the default one, which on a machine without a GPU is the CPU
META = torch.device("meta")


def make_posterior_model(device) -> PosteriorModel:
    return PosteriorModel(variant_log_prior=-10, artifact_log_prior=-10, device=device)


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
