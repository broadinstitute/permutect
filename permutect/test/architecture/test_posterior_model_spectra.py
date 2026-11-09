import math

import torch

from permutect.architecture.posterior_model import PosteriorModel
from permutect.data.datum import Data
from permutect.segmentation.segmentation_model import SegmentationModel
from permutect.test.architecture.test_posterior_model_priors import make_batch
from permutect.test.segmentation.test_segmentation_model import sample_segmentation
from permutect.utils.enums import Call


def test_germline_log_likelihoods_come_from_the_segmentation_model():
    # the tumor's and the normal's, each with its own parameters
    segmentation_model = SegmentationModel(sample_segmentation("tumor"), sample_segmentation("normal"), 0.03, 0.002)
    with torch.no_grad():
        segmentation_model.normal.emission_model.parametrizations.het_concentration.original.fill_(math.log(20.0))
    model = PosteriorModel(-10, -10, device="cpu", segmentation_model=segmentation_model)

    values = {
        Data.ORIGINAL_DEPTH: 40,
        Data.ORIGINAL_ALT_COUNT: 15,
        Data.ORIGINAL_NORMAL_DEPTH: 30,
        Data.ORIGINAL_NORMAL_ALT_COUNT: 12,
        Data.MAF: 0.3,
        Data.NORMAL_MAF: 0.45,
    }
    batch = make_batch([0.01, 0.2], values)
    spectra_log_lks_bc, normal_log_lks_bc = model.spectra.spectra_log_likelihoods_bc(batch)

    log_afs_b = batch.get(Data.LOG_ALLELE_FREQUENCY)
    tumor_counts = (batch.get(Data.ORIGINAL_ALT_COUNT), batch.get(Data.ORIGINAL_DEPTH))
    normal_counts = (batch.get(Data.ORIGINAL_NORMAL_ALT_COUNT), batch.get(Data.ORIGINAL_NORMAL_DEPTH))
    tumor_b = segmentation_model.tumor.germline_log_likelihoods(*tumor_counts, log_afs_b, batch.get(Data.MAF))
    normal_b = segmentation_model.normal.germline_log_likelihoods(*normal_counts, log_afs_b, batch.get(Data.NORMAL_MAF))
    assert torch.allclose(spectra_log_lks_bc[:, Call.GERMLINE], tumor_b)
    assert torch.allclose(normal_log_lks_bc[:, Call.GERMLINE], normal_b)
    # the normal's own parameters, not the tumor's
    tumor_model_on_normal_b = segmentation_model.tumor.germline_log_likelihoods(
        *normal_counts, log_afs_b, batch.get(Data.NORMAL_MAF)
    )
    assert not torch.allclose(normal_b, tumor_model_on_normal_b)


def test_posterior_model_without_a_segmentation_model_uses_default_parameters():
    model = PosteriorModel(-10, -10, device="cpu")
    default = SegmentationModel()
    for sample in ("tumor", "normal"):
        segmentation = getattr(model.spectra.segmentation_model, sample)
        assert segmentation.segments == []
        state, default_state = (
            segmentation.emission_model.state_dict(),
            getattr(default, sample).emission_model.state_dict(),
        )
        assert all(torch.equal(state[key], default_state[key].to(state[key].dtype)) for key in state)
