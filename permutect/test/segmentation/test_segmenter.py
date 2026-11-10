import dataclasses

import numpy as np
import pytest
import torch

from permutect.segmentation import segmenter
from permutect.segmentation.emissions import EmissionModel
from permutect.segmentation.segmenter import Segment
from permutect.segmentation.segmenter import find_runs
from permutect.segmentation.segmenter import log_emissions
from permutect.segmentation.segmenter import make_segments
from permutect.segmentation.segmenter import run_mafs
from permutect.segmentation.segmenter import segment
from permutect.test.segmentation import simulation
from permutect.test.segmentation.simulation import SimulatedSample
from permutect.test.segmentation.simulation import simulate_sample
from permutect.test.segmentation.test_emissions import simulate_pileups

CONTAMINATION = 0.02


def segment_simulated_sample(sample: SimulatedSample):
    return segment(
        sample.pileups,
        sample.positions_n,
        sample.lengths_c,
        num_maf_states=51,
        contamination=CONTAMINATION,
        max_jump_rate=1e-7,
        prior_concentration=0.1,
        max_iterations=30,
    )


def runs(values_n: np.ndarray, lengths_c: np.ndarray) -> list[tuple[int, int]]:
    """the first site and one past the last site of each run of consecutive sites of a contig with equal values"""
    result = []
    for start, length in zip(np.cumsum(lengths_c) - lengths_c, lengths_c):
        values = values_n[start : start + length]
        run_starts = np.concatenate(([0], 1 + np.flatnonzero(values[1:] != values[:-1])))
        run_ends = np.append(run_starts[1:], length)
        result.extend(zip(start + run_starts, start + run_ends))
    return result


def test_simulated_tumor():
    sample = simulate_sample(
        np.random.default_rng(0), purity=0.7, cell_fractions=[1.0, 0.5], contaminations=[CONTAMINATION]
    )
    segmentation = segment_simulated_sample(sample)

    true_runs, called_runs = runs(sample.true_mafs_n, sample.lengths_c), runs(segmentation.states_n, sample.lengths_c)
    assert len(called_runs) == len(true_runs)
    assert np.array_equal(segmentation.run_starts_r, [start for start, _ in called_runs])
    for (true_start, _), (start, end), (low, maf, high) in zip(true_runs, called_runs, segmentation.run_mafs_r3):
        # by chance, sites near a breakpoint between similar MAFs can look like the other side's; in one simulation the
        # 132 sites before a breakpoint from MAF 0.5 to 0.39 favored 0.39
        assert abs(start - true_start) <= 200
        true_maf = sample.true_mafs_n[(start + end) // 2]
        assert maf == pytest.approx(true_maf, abs=0.01)
        # the intervals take the emission model as exact, so with thousands of sites they can be too narrow by a little
        assert low - 0.01 <= true_maf <= high + 0.01

    model = segmentation.emission_model
    assert model.error_rate.item() == pytest.approx(simulation.ERROR_RATE, rel=0.1)
    assert model.reference_bias.item() == pytest.approx(simulation.REFERENCE_BIAS, rel=0.02)


def test_normal_has_one_segment_per_contig():
    # with every MAF 1/2, any breakpoint is a false positive
    rng = np.random.default_rng(1)
    sample = simulate_sample(rng, purity=0.0, cell_fractions=[1.0], contaminations=[CONTAMINATION], num_contigs=2)
    segmentation = segment_simulated_sample(sample)
    assert np.all(segmentation.mafs_k[segmentation.states_n] == 0.5)


def test_find_runs():
    # a later run of the same state is a separate run, as is a run of the same state on the next contig
    assert np.array_equal(find_runs(np.array([3, 3, 7, 7, 3, 3, 4]), np.array([5, 2])), [0, 2, 4, 5, 6])


RUN_MODEL = EmissionModel(contamination=0.01, error_rate=0.005, hom_concentration=50.0, het_concentration=100.0)
# runs with MAFs 0.3, 1/2 and 0.1, the last of only 50 sites
RUN_SIZES_R = np.array([2000, 2000, 50])
RUN_PILEUPS = simulate_pileups(RUN_MODEL, np.repeat([0.3, 0.5, 0.1], RUN_SIZES_R), np.random.default_rng(0))


def test_run_mafs():
    lows_r, mafs_r, highs_r = run_mafs(RUN_MODEL, RUN_PILEUPS, np.cumsum(RUN_SIZES_R) - RUN_SIZES_R).T
    assert np.all((lows_r <= mafs_r) & (mafs_r <= highs_r))
    assert mafs_r[0] == pytest.approx(0.3, abs=0.01)
    # a balanced segment's MAF is weakly identified, since extra spread among its hets looks like a MAF below 1/2
    assert 0.46 <= mafs_r[1] <= 0.5 and highs_r[1] >= 0.48
    # few sites give a wide interval
    assert highs_r[2] - lows_r[2] > 3 * (highs_r[0] - lows_r[0])


def test_run_mafs_in_chunks(monkeypatch):
    run_starts_r = np.cumsum(RUN_SIZES_R) - RUN_SIZES_R
    expected_r3 = run_mafs(RUN_MODEL, RUN_PILEUPS, run_starts_r)
    # chunks of 700 sites split runs
    monkeypatch.setattr(segmenter, "CHUNK_SIZE", 700)
    assert np.allclose(run_mafs(RUN_MODEL, RUN_PILEUPS, run_starts_r), expected_r3)


def test_log_emissions_are_float32_and_match_the_emission_model():
    mafs_k = np.linspace(0, 0.5, 11)
    with torch.no_grad():
        expected_nk = RUN_MODEL.log_likelihoods(RUN_PILEUPS, torch.from_numpy(mafs_k)).numpy()
    log_emissions_nk = log_emissions(RUN_MODEL, RUN_PILEUPS, mafs_k)
    assert log_emissions_nk.dtype == np.float32
    assert np.allclose(log_emissions_nk, expected_nk, rtol=1e-6, atol=0)
    assert np.allclose(log_emissions(RUN_MODEL, RUN_PILEUPS, mafs_k, dtype=np.float64), expected_nk, rtol=1e-12, atol=0)


def test_make_segments():
    contigs_c, lengths_c = ["chr1", "chr2"], np.array([5, 2])
    positions_n = np.array([10, 20, 21, 30, 40, 5, 9])
    run_starts_r = np.array([0, 2, 4, 5, 6])
    # the lower bound of each run's MAF, the MAF and the upper bound
    run_mafs_r3 = np.array(
        [[0.2, 0.25, 0.3], [0.3, 0.3125, 0.35], [0.45, 0.5, 0.5], [0, 0.0625, 0.1], [0.4, 0.4375, 0.45]]
    )
    segments = make_segments(contigs_c, lengths_c, positions_n, run_starts_r, run_mafs_r3)
    expected = [
        Segment("chr1", 10, 20, 0.25, 0.25, 0.2, 0.3),
        # sites 20 and 21 are adjacent, so there's no gap segment between them
        Segment("chr1", 21, 30, 0.3125, 0.3125, 0.3, 0.35),
        # a gap segment has no interval
        Segment("chr1", 31, 39, 0.3125, 0.5),
        Segment("chr1", 40, 40, 0.5, 0.5, 0.45, 0.5),
        # nor is there a gap segment between contigs
        Segment("chr2", 5, 5, 0.0625, 0.0625, 0, 0.1),
        Segment("chr2", 6, 8, 0.0625, 0.4375),
        Segment("chr2", 9, 9, 0.4375, 0.4375, 0.4, 0.45),
    ]
    # Gap segments have NaN interval bounds, and since Python 3.13 dataclasses compare field by field, so NaN != NaN
    # makes equal segments unequal.  Numpy's assert_equal treats NaNs in the same place as equal.
    np.testing.assert_equal([dataclasses.astuple(s) for s in segments], [dataclasses.astuple(s) for s in expected])
