import numpy as np
import pytest

from permutect.segmentation.segmenter import Segment
from permutect.segmentation.segmenter import make_segments
from permutect.segmentation.segmenter import segment
from permutect.test.segmentation import simulation
from permutect.test.segmentation.simulation import SimulatedSample
from permutect.test.segmentation.simulation import simulate_sample

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
        np.random.default_rng(0), purity=0.7, cell_fractions=[1.0, 0.5], contamination=CONTAMINATION
    )
    segmentation = segment_simulated_sample(sample)

    true_runs, called_runs = runs(sample.true_mafs_n, sample.lengths_c), runs(segmentation.states_n, sample.lengths_c)
    assert len(called_runs) == len(true_runs)
    for (true_start, _), (start, end) in zip(true_runs, called_runs):
        # by chance, sites near a breakpoint between similar MAFs can look like the other side's; in one simulation the
        # 132 sites before a breakpoint from MAF 0.5 to 0.39 favored 0.39
        assert abs(start - true_start) <= 200
        mean_maf = np.mean(segmentation.posterior_mafs_n[start:end])
        assert mean_maf == pytest.approx(sample.true_mafs_n[true_start], abs=0.02)

    model = segmentation.emission_model
    assert model.error_rate.item() == pytest.approx(simulation.ERROR_RATE, rel=0.1)
    assert model.reference_bias.item() == pytest.approx(simulation.REFERENCE_BIAS, rel=0.02)


def test_normal_has_one_segment_per_contig():
    # with every MAF 1/2, any breakpoint is a false positive
    rng = np.random.default_rng(1)
    sample = simulate_sample(rng, purity=0.0, cell_fractions=[1.0], contamination=CONTAMINATION, num_contigs=2)
    segmentation = segment_simulated_sample(sample)
    assert np.all(segmentation.mafs_k[segmentation.states_n] == 0.5)


def test_make_segments():
    contigs_c, lengths_c = ["chr1", "chr2"], np.array([5, 2])
    positions_n = np.array([10, 20, 21, 30, 40, 5, 9])
    states_n = np.array([3, 3, 7, 7, 3, 3, 4])
    mafs_n = np.array([0.125, 0.375, 0.25, 0.375, 0.5, 0.0625, 0.4375])
    assert make_segments(contigs_c, lengths_c, positions_n, states_n, mafs_n) == [
        Segment("chr1", 10, 20, 0.25, 0.25),
        # sites 20 and 21 are adjacent, so there's no gap segment between them
        Segment("chr1", 21, 30, 0.3125, 0.3125),
        Segment("chr1", 31, 39, 0.3125, 0.5),
        # a later run of the same state is a separate segment
        Segment("chr1", 40, 40, 0.5, 0.5),
        # as is a run of the same state on the next contig
        Segment("chr2", 5, 5, 0.0625, 0.0625),
        Segment("chr2", 6, 8, 0.0625, 0.4375),
        Segment("chr2", 9, 9, 0.4375, 0.4375),
    ]
