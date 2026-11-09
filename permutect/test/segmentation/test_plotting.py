import math

import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.collections import PatchCollection
from matplotlib.collections import PathCollection

from permutect.segmentation import plotting
from permutect.segmentation.plotting import plot_segmentation
from permutect.segmentation.segmenter import find_runs
from permutect.segmentation.segmenter import make_segments
from permutect.test.segmentation.simulation import SimulatedSample
from permutect.test.segmentation.simulation import simulate_sample

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def simulated_tumor() -> SimulatedSample:
    return simulate_sample(np.random.default_rng(0), purity=0.7, cell_fractions=[1.0, 0.5], contamination=0.0)


def plot_true_segments(sample: SimulatedSample, path):
    """
    plot the segments of the true MAFs, in which every run of sites of a contig with equal MAFs is a segment, with
    intervals of 0.02 on either side of the MAFs
    """
    _, states_n = np.unique(sample.true_mafs_n, return_inverse=True)
    run_starts_r = find_runs(states_n, sample.lengths_c)
    mafs_r = sample.true_mafs_n[run_starts_r]
    run_mafs_r3 = np.stack((np.maximum(mafs_r - 0.02, 0), mafs_r, np.minimum(mafs_r + 0.02, 0.5)), axis=1)
    segments = make_segments(sample.contigs_c, sample.lengths_c, sample.positions_n, run_starts_r, run_mafs_r3)
    args = (sample.contigs_c, sample.lengths_c, sample.positions_n, sample.pileups, segments)
    return plot_segmentation(path, "tumor", *args), segments


def test_plot_segmentation(tmp_path):
    sample = simulated_tumor()
    fig, segments = plot_true_segments(sample, tmp_path / "plot.png")
    with open(tmp_path / "plot.png", "rb") as file:
        assert file.read(8) == PNG_SIGNATURE

    ax = fig.axes[0]
    # every segment of sites, but no gap segment between them, has lines and boxes at its MAF and 1 - MAF
    num_segments_of_sites = sum(not math.isnan(s.maf_low) for s in segments)
    assert num_segments_of_sites < len(segments)
    (lines,) = [collection for collection in ax.collections if isinstance(collection, LineCollection)]
    assert len(lines.get_segments()) == 2 * num_segments_of_sites
    (boxes,) = [collection for collection in ax.collections if isinstance(collection, PatchCollection)]
    assert len(boxes.get_paths()) == 2 * num_segments_of_sites

    # the color of consecutive sites changes exactly where a new segment starts, including at the start of a contig
    (sites,) = [collection for collection in ax.collections if isinstance(collection, PathCollection)]
    colors_n = sites.get_facecolors()[:, :3]
    new_segment_n = np.diff(sample.true_mafs_n) != 0
    new_segment_n[np.cumsum(sample.lengths_c)[:-1] - 1] = True
    assert np.array_equal(np.any(colors_n[1:] != colors_n[:-1], axis=1), new_segment_n)


def test_plot_segmentation_subsamples_sites(tmp_path, monkeypatch):
    monkeypatch.setattr(plotting, "MAX_PLOTTED_SITES", 1000)
    fig, _ = plot_true_segments(simulated_tumor(), tmp_path / "plot.png")
    (sites,) = [collection for collection in fig.axes[0].collections if isinstance(collection, PathCollection)]
    assert len(sites.get_offsets()) == 1000
