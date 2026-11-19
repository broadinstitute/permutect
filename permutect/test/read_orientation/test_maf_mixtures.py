import math

import numpy as np
import torch

from permutect.read_orientation.maf_mixtures import BALANCED_MIXTURE
from permutect.read_orientation.maf_mixtures import MAF_GRID
from permutect.read_orientation.maf_mixtures import MafMixtures
from permutect.read_orientation.maf_mixtures import nearest_grid_mafs
from permutect.segmentation.germline_model import SampleSegmentation
from permutect.segmentation.segmenter import Segment


def segmentation(*segments) -> SampleSegmentation:
    """a segmentation of (contig, start, end, MAF) segments"""
    return SampleSegmentation("SAMPLE", [Segment(contig, start, end, maf, maf) for contig, start, end, maf in segments])


def grid(maf: float) -> int:
    return MAF_GRID.index(maf)


def test_nearest_grid_mafs():
    # 0 rounds up to 0.01, 0.07 down to 0.05 and 0.08 up to 0.1
    mafs = [0.0, 0.012, 0.016, 0.07, 0.08, 0.26, 0.5]
    assert [MAF_GRID[g] for g in nearest_grid_mafs(mafs)] == [0.01, 0.01, 0.02, 0.05, 0.1, 0.25, 0.5]


def test_mixtures_of_positions():
    # segments out of order, and one without a MAF
    mixtures = MafMixtures(
        segmentation(("chr1", 1001, 2000, 0.3), ("chr1", 1, 1000, 0.0), ("chr1", 3001, 4000, math.nan))
    )
    contigs = np.array(["chr1"] * 6 + ["chr2"])
    positions = np.array([1, 1000, 1001, 2000, 2500, 3500, 1500])
    # in a segment, its grid MAF, and between segments, in the segment without a MAF and on a contig without segments, 1/2
    expected = [grid(0.01), grid(0.01), grid(0.3), grid(0.3), BALANCED_MIXTURE, BALANCED_MIXTURE, BALANCED_MIXTURE]
    assert mixtures.of_positions(contigs, positions).tolist() == expected


def test_mixtures_of_blocks():
    mixtures = MafMixtures(
        segmentation(
            # chr1's first block is 30% MAF 0.1 and 70% MAF 0.5, and its second all MAF 0.5
            ("chr1", 1, 300_000, 0.1),
            ("chr1", 300_001, 1_500_000, 0.5),
            # chr2's first block has one segment, which covers 40% of it
            ("chr2", 200_001, 600_000, 0.02),
            # chr3's first block is like chr1's
            ("chr3", 1, 300_000, 0.1),
            ("chr3", 300_001, 1_000_000, 0.5),
        )
    )
    contigs = np.array(["chr1", "chr1", "chr1", "chr2", "chr3", "chr4", "chr1"])
    block_starts = np.array([1, 1_000_001, 2_000_001, 1, 1, 1, 1])
    # one new mixture, of the blocks of chr1 and chr3 with the same weights.  A block that one segment covers in part
    # has its MAF, and a block without segments MAF 1/2.
    straddling = len(MAF_GRID)
    expected = [straddling, grid(0.5), BALANCED_MIXTURE, grid(0.02), straddling, BALANCED_MIXTURE, straddling]
    assert mixtures.of_blocks(contigs, block_starts).tolist() == expected

    weights_mg = mixtures.weights_mg()
    assert torch.equal(weights_mg[: len(MAF_GRID)], torch.eye(len(MAF_GRID), dtype=torch.float64))
    expected_weights = torch.zeros(len(MAF_GRID), dtype=torch.float64)
    expected_weights[[grid(0.1), grid(0.5)]] = torch.tensor([0.3, 0.7], dtype=torch.float64)
    assert weights_mg.shape == (len(MAF_GRID) + 1, len(MAF_GRID))
    assert torch.allclose(weights_mg[straddling], expected_weights)
    assert mixtures.summary() == "4 of 6 blocks overlap segments"


def test_without_segmentation():
    mixtures = MafMixtures()
    assert not mixtures.has_segments()
    assert mixtures.of_positions(np.array(["chr1"]), np.array([5])).tolist() == [BALANCED_MIXTURE]
    assert mixtures.of_blocks(np.array(["chr1"]), np.array([1])).tolist() == [BALANCED_MIXTURE]
    assert torch.equal(mixtures.weights_mg(), torch.eye(len(MAF_GRID), dtype=torch.float64))
