import math

import torch

from permutect.architecture.posterior_model import PosteriorModel
from permutect.data.datum import Data
from permutect.segmentation.germline_model import GermlineModel
from permutect.test.architecture.test_posterior_model_priors import make_batch
from permutect.test.segmentation.test_germline_model import sample_segmentation
from permutect.utils.enums import Call
from permutect.utils.enums import Variation


def test_germline_log_likelihoods_come_from_the_germline_model():
    # one likelihood of the tumor's and the normal's counts jointly, with each sample's own parameters
    germline_model = GermlineModel(sample_segmentation("tumor"), sample_segmentation("normal"), 0.03, 0.002)
    with torch.no_grad():
        germline_model.normal.emission_model.parametrizations.het_concentration.original.fill_(math.log(20.0))
    model = PosteriorModel(-10, -10, device="cpu", germline_model=germline_model)

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
    mafs = (batch.get(Data.MAF), batch.get(Data.NORMAL_MAF))
    joint_b = germline_model.germline_log_likelihoods(*tumor_counts, *normal_counts, log_afs_b, *mafs)
    assert torch.allclose(spectra_log_lks_bc[:, Call.GERMLINE], joint_b)
    # the joint likelihood includes the normal, whose own germline column is 0
    assert torch.all(normal_log_lks_bc[:, Call.GERMLINE] == 0)
    # the normal's counts get the normal's own parameters, not the tumor's
    tumor_parameters_for_both = GermlineModel(germline_model.tumor, germline_model.tumor, 0.03, 0.002)
    assert not torch.allclose(
        joint_b, tumor_parameters_for_both.germline_log_likelihoods(*tumor_counts, *normal_counts, log_afs_b, *mafs)
    )


def test_indels_use_the_indel_parameters_for_their_repeat_lengths():
    germline_model = GermlineModel(sample_segmentation("tumor"), sample_segmentation("normal"), 0.03, 0.002)
    model = PosteriorModel(-10, -10, device="cpu", germline_model=germline_model)
    # a germline deletion in a 16-base homopolymer, in LOH in the tumor, whose alt fraction falls short of 1
    values = {
        Data.VARIANT_TYPE: Variation.DELETION,
        Data.ORIGINAL_DEPTH: 40,
        Data.ORIGINAL_ALT_COUNT: 31,
        Data.ORIGINAL_NORMAL_DEPTH: 30,
        Data.ORIGINAL_NORMAL_ALT_COUNT: 13,
        Data.MAF: 0.0,
        Data.NORMAL_MAF: 0.5,
        Data.REPEAT_UNIT_LENGTH: 1,
        Data.REPEAT_LENGTH: 16,
    }
    batch = make_batch([0.3], values)
    spectra_log_lks_bc, _ = model.spectra.spectra_log_likelihoods_bc(batch)

    counts_and_mafs = (
        batch.get(Data.ORIGINAL_ALT_COUNT),
        batch.get(Data.ORIGINAL_DEPTH),
        batch.get(Data.ORIGINAL_NORMAL_ALT_COUNT),
        batch.get(Data.ORIGINAL_NORMAL_DEPTH),
        batch.get(Data.LOG_ALLELE_FREQUENCY),
        batch.get(Data.MAF),
        batch.get(Data.NORMAL_MAF),
    )
    as_indel = germline_model.germline_log_likelihoods(
        *counts_and_mafs,
        indels=torch.tensor([True]),
        repeat_unit_lengths=batch.get(Data.REPEAT_UNIT_LENGTH),
        repeat_lengths=batch.get(Data.REPEAT_LENGTH),
    )
    assert torch.allclose(spectra_log_lks_bc[:, Call.GERMLINE], as_indel)
    # which is far likelier than with the SNV parameters
    assert spectra_log_lks_bc[0, Call.GERMLINE] > germline_model.germline_log_likelihoods(*counts_and_mafs)[0] + 5


def test_contamination_log_likelihoods_come_from_the_tumor_segmentation():
    germline_model = GermlineModel(sample_segmentation("tumor"), sample_segmentation("normal"), 0.03, 0.002)
    model = PosteriorModel(-10, -10, device="cpu", germline_model=germline_model)
    values = {
        Data.ORIGINAL_DEPTH: 100,
        Data.ORIGINAL_ALT_COUNT: 2,
        Data.ORIGINAL_NORMAL_DEPTH: 30,
        Data.ORIGINAL_NORMAL_ALT_COUNT: 1,
        Data.NORMAL_SEQ_ERROR_LOG_LK: -3,
    }
    batch = make_batch([0.01, 0.2], values)
    spectra_log_lks_bc, normal_log_lks_bc = model.spectra.spectra_log_likelihoods_bc(batch)

    tumor_counts = (batch.get(Data.ORIGINAL_ALT_COUNT), batch.get(Data.ORIGINAL_DEPTH))
    expected_b = germline_model.tumor.contamination_log_likelihoods(*tumor_counts, batch.get(Data.LOG_ALLELE_FREQUENCY))
    assert torch.allclose(spectra_log_lks_bc[:, Call.CONTAMINATION], expected_b)
    # the contaminant's reads are in the tumor alone, so the normal's alt reads are sequencing errors
    assert torch.allclose(normal_log_lks_bc[:, Call.CONTAMINATION], batch.get(Data.NORMAL_SEQ_ERROR_LOG_LK))


def test_posterior_model_without_a_germline_model_uses_default_parameters():
    model = PosteriorModel(-10, -10, device="cpu")
    default = GermlineModel()
    for sample in ("tumor", "normal"):
        segmentation = getattr(model.spectra.germline_model, sample)
        assert segmentation.segments == []
        state, default_state = (
            segmentation.emission_model.state_dict(),
            getattr(default, sample).emission_model.state_dict(),
        )
        assert all(torch.equal(state[key], default_state[key].to(state[key].dtype)) for key in state)
