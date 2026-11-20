import math

import numpy as np
import pytest
import torch
from torch import Tensor

from permutect.slippage.slip_rates import DEFAULT_ALLELE_FREQUENCY
from permutect.slippage.slip_rates import MAX_ALLELE_LENGTH
from permutect.slippage.slip_rates import MAX_UNIT_LENGTH
from permutect.slippage.slip_rates import OTHER_ERROR_RATE
from permutect.slippage.slip_rates import SLIPS
from permutect.slippage.slip_rates import AlleleBins
from permutect.slippage.slip_rates import MonotoneSlipLogOdds
from permutect.slippage.slip_rates import SlippageData
from permutect.slippage.slip_rates import fit_slip_log_odds
from permutect.slippage.slip_rates import slip_probabilities
from permutect.slippage.slippage_counts import COUNT_COLUMNS
from permutect.slippage.slippage_counts import SlippageCounts

EPSILON = OTHER_ERROR_RATE
# log odds of slips by -2, -1, +1 and +2 units, and the probabilities of no slip and of those slips
LOG_ODDS_J = (-6.0, -3.0, -4.0, -7.0)
P_O = np.exp((0, *LOG_ODDS_J)) / np.sum(np.exp((0, *LOG_ODDS_J)))


def constant_log_odds(log_odds_j=LOG_ODDS_J) -> Tensor:
    """a grid of log odds that are the same at every unit length and allele length"""
    log_odds_j = torch.as_tensor(log_odds_j, dtype=torch.float64)
    return log_odds_j[:, None, None].expand(len(SLIPS), MAX_UNIT_LENGTH, MAX_ALLELE_LENGTH)


def allele_probabilities(unit_length: int, ref_length: int, change: int, log_odds_jul: Tensor) -> np.ndarray:
    bins = AlleleBins.of(torch.tensor([unit_length]), torch.tensor([ref_length]), torch.tensor([change]))
    return bins.probabilities_sk(log_odds_jul)[0].numpy()


def counts_of_one_str(counts_k, alt_change: int, allele_frequency: float, unit_length=1, ref_length=10):
    return SlippageCounts(
        sample="SAMPLE",
        contigs=["1"],
        starts_s=torch.tensor([1000]),
        unit_lengths_s=torch.tensor([unit_length]),
        repeat_lengths_s=torch.tensor([ref_length]),
        counts_sk=torch.tensor([counts_k]),
        alt_changes_sa=torch.tensor([[alt_change, 0]]),
        alt_allele_frequencies_sa=torch.tensor([[allele_frequency, math.nan]], dtype=torch.float64),
    )


@pytest.mark.parametrize("seed", range(5))
def test_log_odds_rise_with_length_and_fall_with_unit_length(seed):
    model = MonotoneSlipLogOdds()
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.copy_(3 * torch.randn(parameter.shape, generator=generator, dtype=torch.float64))
    log_odds_jul = model()
    assert log_odds_jul.shape == (len(SLIPS), MAX_UNIT_LENGTH, MAX_ALLELE_LENGTH)
    assert torch.all(torch.diff(log_odds_jul, dim=-1) >= 0)
    assert torch.all(torch.diff(log_odds_jul, dim=1) < 0)
    # each longer unit length's log odds rise at least as fast as the next shorter one's
    assert torch.all(torch.diff(torch.diff(log_odds_jul, dim=-1), dim=1) >= -1e-9)


def test_fragments_of_the_reference_allele():
    # the outcomes reach changes of -2 to +2 units, which leaves five bins for other errors: -4 or more, -3, +3, +4 or
    # more, and part of a unit
    reached = 1 - 5 * EPSILON
    no_slip, minus_2, minus_1, plus_1, plus_2 = reached * P_O
    expected_k = [EPSILON, EPSILON, minus_2, minus_1, no_slip, plus_1, plus_2, EPSILON, EPSILON, EPSILON]
    result_k = allele_probabilities(1, 10, 0, constant_log_odds())
    assert result_k == pytest.approx(expected_k, rel=1e-12)
    assert result_k.sum() == pytest.approx(1, rel=1e-12)


def test_slips_past_four_units_share_the_last_bin():
    # an allele with 4 fewer units, whose slips by -1 and -2 units also land in the -4 or more bin
    reached = 1 - 7 * EPSILON
    no_slip, minus_2, minus_1, plus_1, plus_2 = reached * P_O
    expected_k = [no_slip + minus_1 + minus_2, plus_1, plus_2] + [EPSILON] * 7
    result_k = allele_probabilities(2, 20, -4, constant_log_odds())
    assert result_k == pytest.approx(expected_k, rel=1e-12)
    assert result_k.sum() == pytest.approx(1, rel=1e-12)


@pytest.mark.parametrize(
    "unit_length, ref_length, change, allele_length",
    [
        (2, 20, -4, 12),  # a dinucleotide allele with 4 fewer units
        (1, 7, 2, 9),
        (4, 8, -4, 1),  # a deletion longer than the STR, which gets the rates of the grid's shortest length
        (1, 45, 1, MAX_ALLELE_LENGTH),  # alleles longer than the grid slip at the rates of its longest
    ],
)
def test_alleles_slip_at_the_rates_of_their_unit_length_and_length(unit_length, ref_length, change, allele_length):
    log_odds_jul = torch.randn(len(SLIPS), MAX_UNIT_LENGTH, MAX_ALLELE_LENGTH, dtype=torch.float64)
    expected_k = allele_probabilities(
        unit_length, ref_length, change, constant_log_odds(log_odds_jul[:, unit_length - 1, allele_length - 1])
    )
    assert allele_probabilities(unit_length, ref_length, change, log_odds_jul) == pytest.approx(expected_k, rel=1e-12)


def test_log_likelihood_sums_over_genotypes():
    # a homopolymer of 10 bases with an alt allele of one fewer base at population allele frequency q, in a segment of
    # MAF f
    counts_k = [0, 1, 2, 30, 25, 3, 0, 0, 0, 1]
    q, f = 0.1, 0.3
    data = SlippageData.of(counts_of_one_str(counts_k, -1, q), torch.tensor([f], dtype=torch.float64))
    result = data.log_likelihoods_s(constant_log_odds()).item()

    reached = 1 - 5 * EPSILON
    ref_k, alt_k = np.full(len(COUNT_COLUMNS), EPSILON), np.full(len(COUNT_COLUMNS), EPSILON)
    # no slip, -2, -1, +1, +2 in the order of the bins
    ref_k[2:7] = reached * P_O[[1, 2, 0, 3, 4]]
    alt_k[1:6] = reached * P_O[[1, 2, 0, 3, 4]]
    priors_g = [(1 - q) ** 2, q * (1 - q), q * (1 - q), q**2]
    ref_fractions_g = [1, f, 1 - f, 0]
    expected = math.log(
        sum(
            prior * np.prod((w * ref_k + (1 - w) * alt_k) ** np.array(counts_k))
            for prior, w in zip(priors_g, ref_fractions_g)
        )
    )
    assert result == pytest.approx(expected, rel=1e-12)


def test_strs_without_an_alt_allele_are_hom_ref():
    counts_k = [0, 0, 0, 1, 40, 0, 0, 0, 0, 0]
    data = SlippageData.of(counts_of_one_str(counts_k, 0, math.nan), torch.tensor([0.5]))
    expected = np.sum(np.array(counts_k) * np.log(allele_probabilities(1, 10, 0, constant_log_odds())))
    assert data.log_likelihoods_s(constant_log_odds()).item() == pytest.approx(expected, rel=1e-12)


def test_alt_alleles_without_an_allele_frequency_get_the_default():
    counts_k = [0, 0, 0, 20, 20, 0, 0, 0, 0, 0]
    for default in (DEFAULT_ALLELE_FREQUENCY, 0.01):
        unknown = SlippageData.of(counts_of_one_str(counts_k, -1, math.nan), torch.tensor([0.5]), default)
        known = SlippageData.of(counts_of_one_str(counts_k, -1, default), torch.tensor([0.5]))
        assert unknown.log_likelihoods_s(constant_log_odds()).item() == known.log_likelihoods_s(constant_log_odds())


def test_units_longer_than_the_grid_are_an_error():
    with pytest.raises(ValueError, match="at most 5 bases"):
        SlippageData.of(counts_of_one_str([0] * 4 + [10] + [0] * 5, 0, math.nan, 6, 12), torch.tensor([0.5]))


def test_fit_recovers_the_slip_rates_of_simulated_strs():
    generator = torch.Generator().manual_seed(0)
    truth = MonotoneSlipLogOdds()
    with torch.no_grad():
        for parameter in truth.parameters():
            parameter.add_(0.3 * torch.randn(parameter.shape, generator=generator, dtype=torch.float64))
        true_log_odds_jul = truth()

    # STRs of several unit lengths and lengths, half with a germline allele of one fewer unit at allele frequency 0.3,
    # whose genotypes are in Hardy-Weinberg equilibrium
    classes = [(1, 8), (1, 10), (1, 12), (1, 14), (2, 10), (2, 14), (3, 12), (4, 16)]
    strs_per_class, depth, q = 1000, 100, 0.3
    unit_lengths_s = torch.tensor([u for u, _ in classes]).repeat_interleave(strs_per_class)
    ref_lengths_s = torch.tensor([length for _, length in classes]).repeat_interleave(strs_per_class)
    num_strs = len(unit_lengths_s)
    alt_changes_s = torch.where(torch.arange(num_strs) % 2 == 0, -1, 0)
    ref_haplotypes_s = torch.sum(torch.rand(num_strs, 2, generator=generator) > q, dim=-1)
    ref_fractions_s = torch.where(alt_changes_s == 0, 1.0, ref_haplotypes_s / 2).double()
    ref_sk = AlleleBins.of(unit_lengths_s, ref_lengths_s, torch.zeros_like(alt_changes_s))
    alt_sk = AlleleBins.of(unit_lengths_s, ref_lengths_s, alt_changes_s)
    probabilities_sk = ref_fractions_s[:, None] * ref_sk.probabilities_sk(true_log_odds_jul)
    probabilities_sk += (1 - ref_fractions_s[:, None]) * alt_sk.probabilities_sk(true_log_odds_jul)
    bins_sd = torch.multinomial(probabilities_sk, depth, replacement=True, generator=generator)
    counts_sk = torch.nn.functional.one_hot(bins_sd, len(COUNT_COLUMNS)).sum(dim=1)
    counts = SlippageCounts(
        sample="SAMPLE",
        contigs=["1"] * num_strs,
        starts_s=torch.arange(num_strs),
        unit_lengths_s=unit_lengths_s,
        repeat_lengths_s=ref_lengths_s,
        counts_sk=counts_sk,
        alt_changes_sa=torch.stack((alt_changes_s, torch.zeros_like(alt_changes_s)), dim=-1),
        alt_allele_frequencies_sa=torch.tensor([[q, math.nan]], dtype=torch.float64).expand(num_strs, 2),
    )

    log_odds_jul, _, _ = fit_slip_log_odds(SlippageData.of(counts, torch.full((num_strs,), 0.5)), max_iterations=300)

    # the rates of slips by one unit of the reference alleles, where there are enough slips to estimate them well
    true_rates_jul, rates_jul = slip_probabilities(true_log_odds_jul), slip_probabilities(log_odds_jul)
    fragments, compared = strs_per_class * depth, 0
    for u, length in classes:
        for slip in (-1, 1):
            true_rate = true_rates_jul[SLIPS.index(slip), u - 1, length - 1].item()
            if true_rate * fragments > 300:
                assert rates_jul[SLIPS.index(slip), u - 1, length - 1].item() == pytest.approx(true_rate, rel=0.2)
                compared += 1
    assert compared >= 6
