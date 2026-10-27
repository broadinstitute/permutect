import math

import numpy as np
import torch

from permutect.architecture.spectra.posterior_model_spectra import germline_log_likelihood
from permutect.utils.stats_utils import beta_binomial_log_lk


def linear_space_germline_log_likelihood(afs, mafs, alt_counts, depths):
    # the original implementation in terms of allele frequencies rather than log allele frequencies
    het_probs = 2 * afs * (1 - afs)
    hom_probs = afs * afs
    het_proportion = het_probs / (het_probs + hom_probs)
    hom_proportion = 1 - het_proportion
    log_half_het_prop = torch.log(het_proportion / 2)
    ref_counts = depths - alt_counts
    combinatorial_term = torch.lgamma(depths + 1) - torch.lgamma(alt_counts + 1) - torch.lgamma(ref_counts + 1)
    alt_minor_ll = (
        log_half_het_prop + combinatorial_term + alt_counts * torch.log(mafs) + ref_counts * torch.log(1 - mafs)
    )
    alt_major_ll = (
        log_half_het_prop + combinatorial_term + ref_counts * torch.log(mafs) + alt_counts * torch.log(1 - mafs)
    )
    hom_ll = torch.log(hom_proportion) + beta_binomial_log_lk(
        depths, alt_counts, torch.tensor([98.0]), torch.tensor([2.0])
    )
    return torch.logsumexp(torch.vstack((alt_minor_ll, alt_major_ll, hom_ll)), dim=0)


def test_zero_counts_give_zero_log_likelihood():
    log_afs = torch.log(torch.tensor([1e-12, 1e-6, 0.01, 0.3, 0.9, 1.0]))
    mafs = torch.full_like(log_afs, 0.4)
    zeros = torch.zeros_like(log_afs)
    for het_beta in (None, 10.0):
        result = germline_log_likelihood(log_afs, mafs, zeros, zeros, het_beta)
        # float32 lgamma cancellation in the hom alt beta binomial leaves errors of order 1e-5
        assert torch.allclose(result, torch.zeros_like(result), atol=1e-4)


def test_agrees_with_linear_space_for_moderate_afs():
    afs = torch.tensor([1e-4, 1e-3, 0.01, 0.1, 0.3, 0.5, 0.8, 0.99])
    mafs = torch.tensor([0.5, 0.45, 0.3, 0.5, 0.2, 0.5, 0.4, 0.5])
    depths = torch.tensor([30.0, 50.0, 10.0, 100.0, 40.0, 25.0, 60.0, 35.0])
    alt_counts = torch.tensor([15.0, 2.0, 5.0, 98.0, 8.0, 12.0, 30.0, 34.0])
    expected = linear_space_germline_log_likelihood(afs, mafs, alt_counts, depths)
    calc = germline_log_likelihood(torch.log(afs), mafs, alt_counts, depths)
    assert torch.allclose(calc, expected, rtol=1e-5)


def test_tiny_afs_survive_float16_storage():
    # allele frequencies below ~6e-8 round to zero in float16, but their logs are stored with small relative error
    afs = np.array([1e-9, 1e-15, 1e-30], dtype=np.float64)
    assert np.all(afs.astype(np.float16) == 0)
    log_afs = torch.from_numpy(np.log(afs).astype(np.float16)).to(torch.float)
    assert torch.allclose(torch.exp(log_afs).double(), torch.from_numpy(afs), rtol=0.02)

    mafs = torch.full_like(log_afs, 0.5)
    depths = torch.full_like(log_afs, 40.0)
    for alt_count in (0.0, 20.0, 40.0):
        result = germline_log_likelihood(log_afs, mafs, torch.full_like(log_afs, alt_count), depths)
        assert torch.all(torch.isfinite(result))

    # with zero AF the het proportion is 1, so the likelihood is the het binomial
    het_only = germline_log_likelihood(log_afs, mafs, torch.full_like(log_afs, 20.0), depths)
    het_binomial = math.lgamma(41) - 2 * math.lgamma(21) + 40 * math.log(0.5)
    assert torch.allclose(het_only, torch.full_like(het_only, het_binomial), atol=1e-5)
