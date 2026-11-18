"""
The minor allele fractions (MAFs) of the sites of the F1R2 counts, from segment_pileups' segmentation of the sample, for
the likelihoods of germline hets in learning the read orientation model.  For efficiency, MAFs are rounded to the nearest
MAF of MAF_GRID.

A site of an alt table has a position, and so a single MAF, but the sites of a histogram row could be anywhere in its
block of the genome, so they have a mixture of MAFs, weighted by how much of the block each segment covers.  Every site
gets the index of its MAF mixture, a weight for each MAF of the grid.  The first len(MAF_GRID) mixtures are the MAFs of
the grid, each alone, and the rest are the mixtures of blocks that overlap segments with different MAFs.
"""

import math
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from torch import Tensor

from permutect.segmentation.germline_model import SampleSegmentation

# the MAFs that segments' MAFs are rounded to, finer near 0, where loss of heterozygosity puts them
MAF_GRID = (0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5)
# the mixture of the sites outside the segments, and of all sites without a segmentation: MAF 1/2
BALANCED_MIXTURE = MAF_GRID.index(0.5)
# the length of the blocks of the genome in the histograms of the F1R2 counts, GATK's F1R2FilterConstants.BLOCK_SIZE.
# A histogram row's block_start is 1-based.
BLOCK_SIZE = 1_000_000


def nearest_grid_mafs(mafs) -> np.ndarray:
    """the index in MAF_GRID of the grid MAF nearest each MAF"""
    return np.abs(np.asarray(mafs, dtype=np.float64)[:, None] - np.array(MAF_GRID)).argmin(axis=1)


class MafMixtures:
    """
    The MAF mixtures of sites given a segmentation, or, without one, MAF 1/2 everywhere.  The table of mixtures grows as
    blocks that overlap segments with different MAFs turn up.
    """

    def __init__(self, segmentation: SampleSegmentation | None = None):
        # The segments with MAFs, by contig: their sorted 1-based starts, their inclusive ends, and their grid MAFs.  As
        # filter_variants has it, a gap between segments is a segment with the MAF of the segment before it.
        segments_by_contig = defaultdict(list)
        for segment in () if segmentation is None else segmentation.segments:
            if not math.isnan(segment.maf):
                segments_by_contig[segment.contig].append(segment)
        self.segments = {}
        for contig, segments in segments_by_contig.items():
            segments.sort(key=lambda segment: segment.start)
            starts, ends = (
                np.array([getattr(s, field) for s in segments], dtype=np.int64) for field in ("start", "end")
            )
            self.segments[contig] = (starts, ends, nearest_grid_mafs([segment.maf for segment in segments]))

        self.weights = list(np.eye(len(MAF_GRID)))  # of each mixture, by index
        self._index_of_weights = {}  # of the mixtures after the grid's
        self._block_mixtures = {}  # by (contig, block start)
        self._num_blocks_in_segments = 0

    def has_segments(self) -> bool:
        return bool(self.segments)

    def of_positions(self, contigs: np.ndarray, positions: np.ndarray) -> np.ndarray:
        """the mixture of the site at each position: the grid MAF of its segment, or 1/2 outside the segments"""
        result = np.full(len(positions), BALANCED_MIXTURE, dtype=np.int64)
        for contig in np.unique(contigs):
            if contig not in self.segments:
                continue
            starts, ends, grid_mafs = self.segments[contig]
            in_contig = contigs == contig
            contig_positions = positions[in_contig]
            # the last segment to start at or before each position, which contains it unless it ends before it
            segment = np.maximum(np.searchsorted(starts, contig_positions, side="right") - 1, 0)
            inside = (starts[segment] <= contig_positions) & (contig_positions <= ends[segment])
            result[in_contig] = np.where(inside, grid_mafs[segment], BALANCED_MIXTURE)
        return result

    def of_blocks(self, contigs: np.ndarray, block_starts: np.ndarray) -> np.ndarray:
        """the mixture of the sites of each histogram row's block"""
        if not self.segments:
            return np.full(len(contigs), BALANCED_MIXTURE, dtype=np.int64)
        codes, blocks = pd.MultiIndex.from_arrays([contigs, block_starts]).factorize()
        mixtures = [self._block_mixture(contig, int(block_start)) for contig, block_start in blocks]
        return np.array(mixtures, dtype=np.int64)[codes]

    def _block_mixture(self, contig: str, block_start: int) -> int:
        """
        The mixture of a block: the grid MAFs of the segments that overlap it, weighted by the overlaps and normalized
        over the part of the block that they cover, since no sites lie between segments.  A block that no segment
        overlaps has MAF 1/2.
        """
        if (contig, block_start) not in self._block_mixtures:
            mixture = BALANCED_MIXTURE
            if contig in self.segments:
                starts, ends, grid_mafs = self.segments[contig]
                overlaps = np.minimum(ends, block_start + BLOCK_SIZE - 1) - np.maximum(starts, block_start) + 1
                overlapping = overlaps > 0
                self._num_blocks_in_segments += int(overlapping.any())
                weights = np.bincount(grid_mafs[overlapping], overlaps[overlapping], minlength=len(MAF_GRID))
                if np.count_nonzero(weights) == 1:
                    mixture = int(weights.argmax())
                elif np.count_nonzero(weights) > 1:
                    mixture = self._mixture_index(weights / weights.sum())
            self._block_mixtures[contig, block_start] = mixture
        return self._block_mixtures[contig, block_start]

    def _mixture_index(self, weights: np.ndarray) -> int:
        """the index of a mixture of more than one grid MAF, added to the table unless it's there already"""
        key = tuple(weights.tolist())
        if key not in self._index_of_weights:
            self._index_of_weights[key] = len(self.weights)
            self.weights.append(weights)
        return self._index_of_weights[key]

    def weights_mg(self) -> Tensor:
        """the weight of each grid MAF in each mixture, by mixture index"""
        return torch.tensor(np.array(self.weights), dtype=torch.float64)

    def summary(self) -> str:
        """how many of the blocks seen so far overlap segments, eg to catch a segmentation of another reference"""
        return f"{self._num_blocks_in_segments} of {len(self._block_mixtures)} blocks overlap segments"


def balanced_mixtures_mg() -> Tensor:
    """the table of mixtures without a segmentation, the grid MAFs alone"""
    return MafMixtures().weights_mg()
