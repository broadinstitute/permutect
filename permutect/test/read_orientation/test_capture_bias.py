import numpy as np
import pytest
import torch

from permutect.read_orientation.capture_bias import MAX_CONCENTRATION
from permutect.read_orientation.capture_bias import MIN_CONCENTRATION
from permutect.read_orientation.capture_bias import fit_capture_bias
from permutect.read_orientation.capture_bias import moment_estimates
from permutect.read_orientation.f1r2_counts_reader import CaptureCounts


def counts_of(depths, f1r2_counts) -> CaptureCounts:
    pairs, counts = np.unique(np.column_stack([depths, f1r2_counts]), axis=0, return_counts=True)
    return CaptureCounts(*(torch.tensor(x, dtype=torch.int64) for x in (pairs[:, 0], pairs[:, 1], counts)))


def simulated_counts(f1r2_fraction: float, concentration: float, num_sites: int = 20_000) -> CaptureCounts:
    """the reads of sites with depths from 2 to 200 and F1R2 fractions drawn from a beta distribution"""
    rng = np.random.default_rng(1)
    depths = rng.integers(2, 201, num_sites)
    fractions = rng.beta(f1r2_fraction * concentration, (1 - f1r2_fraction) * concentration, num_sites)
    return counts_of(depths, rng.binomial(depths, fractions))


# like a skewed FFPE exome, a fresh one, and a broader spread than either
@pytest.mark.parametrize("f1r2_fraction, concentration", [(0.15, 25.0), (0.5, 600.0), (0.3, 8.0)])
def test_fit_recovers_the_capture_bias(f1r2_fraction, concentration):
    counts = simulated_counts(f1r2_fraction, concentration)
    capture = fit_capture_bias(counts)
    assert capture.f1r2_fraction.item() == pytest.approx(f1r2_fraction, abs=0.005)
    assert capture.concentration.item() == pytest.approx(concentration, rel=0.2)
    alpha, beta = capture.pseudocounts()
    assert alpha / (alpha + beta) == pytest.approx(capture.f1r2_fraction.item())
    assert alpha + beta == pytest.approx(capture.concentration.item())

    # the method of moments, which the fit starts from, is close too
    moment_fraction, moment_concentration = moment_estimates(counts)
    assert moment_fraction == pytest.approx(f1r2_fraction, abs=0.005)
    assert moment_concentration == pytest.approx(concentration, rel=0.5)


def test_binomial_sites_have_a_large_concentration():
    # every site's F1R2 fraction is 0.4
    rng = np.random.default_rng(2)
    depths = rng.integers(2, 201, 20_000)
    capture = fit_capture_bias(counts_of(depths, rng.binomial(depths, 0.4)))
    assert capture.f1r2_fraction.item() == pytest.approx(0.4, abs=0.005)
    assert 1000 < capture.concentration.item() <= MAX_CONCENTRATION


def test_concentration_has_a_minimum():
    # sites with all their reads in one orientation or the other
    capture = fit_capture_bias(counts_of([50, 50], [0, 50]))
    assert capture.concentration.item() == pytest.approx(MIN_CONCENTRATION, rel=0.01)


def test_sites_of_one_read():
    # they tell the fraction but nothing of its spread
    capture = fit_capture_bias(counts_of([1, 1, 1, 1], [0, 1, 0, 0]))
    assert capture.f1r2_fraction.item() == pytest.approx(0.25, abs=0.01)


def test_without_sites():
    assert fit_capture_bias(CaptureCounts.empty()) is None
