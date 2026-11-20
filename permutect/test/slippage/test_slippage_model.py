import math

import numpy as np
import pytest
import torch
from scipy.special import betainc
from scipy.stats import betabinom

from permutect.slippage.slippage_model import SLIPPAGE_ALLELE_FRACTION_PSEUDOCOUNTS
from permutect.slippage.slippage_model import SlippageModel
from permutect.utils.enums import Variation


def test_slippage_is_a_one_unit_indel_of_a_long_str():
    # each variant's type, reference STR length and change in repeat count, and whether it can be slippage
    cases = [
        (Variation.DELETION, 8, -1, True),  # a homopolymer of the minimum length loses a base
        (Variation.INSERTION, 12, 1, True),
        (Variation.BIG_DELETION, 12, -1, True),  # eg a dinucleotide repeat loses a unit
        (Variation.BIG_INSERTION, 20, 1, True),
        (Variation.DELETION, 7, -1, False),  # too short an STR
        (Variation.BIG_DELETION, 12, -2, False),  # two repeat units
        (Variation.INSERTION, 0, 0, False),  # outside an STR
        (Variation.SNV, 12, 0, False),
        (Variation.MNV, 12, 0, False),
        (Variation.DELETION, math.nan, math.nan, False),  # not annotated from the VCF
    ]
    types, lengths, changes, expected = zip(*cases)
    result_b = SlippageModel().possible_b(torch.tensor(types), torch.tensor(lengths), torch.tensor(changes))
    assert result_b.tolist() == list(expected)


def test_tumor_likelihoods_are_a_normalized_beta_binomial():
    depth = 50
    alt_counts_b = torch.arange(depth + 1, dtype=torch.float64)
    result_b = SlippageModel().tumor_log_likelihoods_b(torch.full_like(alt_counts_b, depth), alt_counts_b)
    alpha, beta = SLIPPAGE_ALLELE_FRACTION_PSEUDOCOUNTS
    assert np.allclose(result_b.numpy(), betabinom.logpmf(alt_counts_b.numpy(), depth, alpha, beta))
    assert torch.exp(result_b).sum().item() == pytest.approx(1)


def passing_allele_fraction(log_slippage_k: np.ndarray, depth: int) -> float:
    """
    The allele fraction above which a somatic indel beats slippage at every alt count, with a somatic prior of 1e-6, a
    flat somatic allele fraction and slippage's log prior of 0
    """
    log_somatic = math.log(1e-6) - math.log(depth + 1)
    slippage_wins_k = log_slippage_k >= log_somatic
    return (np.flatnonzero(slippage_wins_k).max() + 1) / depth


@pytest.mark.parametrize("depth", [20, 1000])
def test_decisions_approximate_filter_mutect_calls(depth):
    # FilterMutectCalls' slippage likelihood is the regularized incomplete beta function I_r(k + 1, n - k + 1) with r =
    # 0.1.  The beta-binomial's passing allele fraction is lower, by at most 0.07.
    alt_counts_k = np.arange(depth + 1)
    with np.errstate(divide="ignore"):  # far above r, I_r underflows to 0
        gatk_k = np.log(betainc(alt_counts_k + 1, depth - alt_counts_k + 1, 0.1))
    depths_k = torch.full((depth + 1,), float(depth), dtype=torch.float64)
    model_k = SlippageModel().tumor_log_likelihoods_b(depths_k, torch.from_numpy(alt_counts_k).double()).numpy()
    gatk_fraction, model_fraction = passing_allele_fraction(gatk_k, depth), passing_allele_fraction(model_k, depth)
    assert gatk_fraction - 0.07 <= model_fraction <= gatk_fraction


def test_save_and_load(tmp_path):
    path = tmp_path / "slippage-model.pt"
    slip_log_odds_jul = torch.randn(4, 5, 40, dtype=torch.float64)
    # pseudocounts that float32 would round
    SlippageModel((67.13, 1055.24), 10, slip_log_odds_jul).save(path)
    model = SlippageModel.load(path)
    assert model.allele_fraction_pseudocounts_2.tolist() == [67.13, 1055.24]
    assert model.allele_fraction_pseudocounts_2.dtype == torch.float64
    assert model.min_str_length == 10
    assert torch.equal(model.slip_log_odds_jul, slip_log_odds_jul)


def test_save_and_load_without_slip_rates(tmp_path):
    path = tmp_path / "slippage-model.pt"
    SlippageModel().save(path)
    assert SlippageModel.load(path).slip_log_odds_jul.numel() == 0
