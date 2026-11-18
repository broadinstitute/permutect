import numpy as np
import pytest
import torch

from permutect.read_orientation.baseline_orientation import MAX_CONCENTRATION
from permutect.read_orientation.baseline_orientation import MIN_CONCENTRATION
from permutect.read_orientation.baseline_orientation import fit_baseline_orientation
from permutect.read_orientation.baseline_orientation import moment_estimates
from permutect.read_orientation.f1r2_counts_reader import RefOrientations


def orientations_of(depths, f1r2_counts) -> RefOrientations:
    pairs, counts = np.unique(np.column_stack([depths, f1r2_counts]), axis=0, return_counts=True)
    return RefOrientations(*(torch.tensor(x, dtype=torch.int64) for x in (pairs[:, 0], pairs[:, 1], counts)))


def simulated_orientations(f1r2_fraction: float, concentration: float, num_sites: int = 20_000) -> RefOrientations:
    """the ref reads of sites with depths from 2 to 200 and F1R2 fractions drawn from a beta distribution"""
    rng = np.random.default_rng(1)
    depths = rng.integers(2, 201, num_sites)
    fractions = rng.beta(f1r2_fraction * concentration, (1 - f1r2_fraction) * concentration, num_sites)
    return orientations_of(depths, rng.binomial(depths, fractions))


# like a skewed FFPE exome, a fresh one, and a broader spread than either
@pytest.mark.parametrize("f1r2_fraction, concentration", [(0.15, 25.0), (0.5, 600.0), (0.3, 8.0)])
def test_fit_recovers_the_baseline(f1r2_fraction, concentration):
    orientations = simulated_orientations(f1r2_fraction, concentration)
    baseline = fit_baseline_orientation(orientations)
    assert baseline.f1r2_fraction.item() == pytest.approx(f1r2_fraction, abs=0.005)
    assert baseline.concentration.item() == pytest.approx(concentration, rel=0.2)
    alpha, beta = baseline.pseudocounts()
    assert alpha / (alpha + beta) == pytest.approx(baseline.f1r2_fraction.item())
    assert alpha + beta == pytest.approx(baseline.concentration.item())

    # the method of moments, which the fit starts from, is close too
    moment_fraction, moment_concentration = moment_estimates(orientations)
    assert moment_fraction == pytest.approx(f1r2_fraction, abs=0.005)
    assert moment_concentration == pytest.approx(concentration, rel=0.5)


def test_binomial_sites_have_a_large_concentration():
    # every site's F1R2 fraction is 0.4
    rng = np.random.default_rng(2)
    depths = rng.integers(2, 201, 20_000)
    baseline = fit_baseline_orientation(orientations_of(depths, rng.binomial(depths, 0.4)))
    assert baseline.f1r2_fraction.item() == pytest.approx(0.4, abs=0.005)
    assert 1000 < baseline.concentration.item() <= MAX_CONCENTRATION


def test_concentration_has_a_minimum():
    # sites with all their ref reads in one orientation or the other
    baseline = fit_baseline_orientation(orientations_of([50, 50], [0, 50]))
    assert baseline.concentration.item() == pytest.approx(MIN_CONCENTRATION, rel=0.01)


def test_sites_of_one_ref_read():
    # they tell the fraction but nothing of its spread
    baseline = fit_baseline_orientation(orientations_of([1, 1, 1, 1], [0, 1, 0, 0]))
    assert baseline.f1r2_fraction.item() == pytest.approx(0.25, abs=0.01)


def test_without_ref_reads():
    assert fit_baseline_orientation(RefOrientations.empty()) is None
