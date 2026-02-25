from __future__ import annotations

import copy
from random import randint
from typing import List

import numpy as np
import torch
from torch import Tensor
from torch_scatter import segment_csr

from permutect.data.batch import Batch
from permutect.data.datum import Data, Datum


class ReadsBatch(Batch):
    """
    Read sets have different sizes so we can't form a batch by naively stacking tensors.  We need a custom way
    to collate a list of Datum into a Batch

    collated batch contains:
    2D tensors of ALL ref (alt) reads, not separated by set.
    number of reads in ref (alt) read sets, in same order as read tensors
    info: 2D tensor of info fields, one row per variant
    labels: 1D tensor of 0 if non-artifact, 1 if artifact
    lists of original mutect2_data and site info

    Example: if we have two input data, one with alt reads [[0,1,2], [3,4,5] and the other with
    alt reads [[6,7,8], [9,10,11], [12,13,14] then the output alt reads tensor is
    [[0,1,2], [3,4,5], [6,7,8], [9,10,11], [12,13,14]] and the output counts are [2,3]
    inside the model, the counts will be used to separate the reads into sets
    """

    def __init__(self, data: List[Datum]):
        super().__init__(data)



class DownsampledReadsBatch(ReadsBatch):
    """
    wrapper class that downsamples reads on the fly without copying data
    This lets us produce multiple count augmentations from a single batch very efficiently
    """
    def __init__(self, original_batch: ReadsBatch, ref_fracs_b: Tensor, alt_fracs_b: Tensor):
        """
        This is delicate.  We're constructing it without calling super().__init__
        """
        self.int_tensor = original_batch.int_tensor # note: no copy -- we never modify it!!!
        self.float_tensor = original_batch.float_tensor # note: no copy -- we never modify it!!!
        self.device = self.int_tensor.device
        self.reads_re = original_batch.reads_re
        self._finish_initializiation_from_arrays()
        # at this point all member variables needed by the parent class are available

        old_ref_counts, old_alt_counts = original_batch.get(Data.REF_COUNT), original_batch.get(Data.ALT_COUNT)
        old_total_ref, old_total_alt = torch.sum(old_ref_counts), torch.sum(old_alt_counts)

        ref_probs_r = torch.repeat_interleave(ref_fracs_b, dim=0, repeats=old_ref_counts)
        alt_probs_r = torch.repeat_interleave(alt_fracs_b, dim=0, repeats=old_alt_counts)
        keep_ref_mask = torch.zeros(old_total_ref, device=self.device, dtype=torch.int64)
        keep_ref_mask.bernoulli_(p=ref_probs_r)    # fills in-place with Bernoulli samples
        keep_alt_mask = torch.zeros(old_total_alt, device=self.device, dtype=torch.int64)
        keep_alt_mask.bernoulli_(p=alt_probs_r)    # fills in-place with Bernoulli samples

        # unlike ref, we need to ensure at least one alt read.  One way to do that is to set one random element from each range of alts
        # to be masked to keep.  If eg we have alt counts of 3, 4, 7, 2 in the batch, the cumsums starting from zero are
        # 0, 3, 7, 14.  If we simply set indices 0, 3, 7, 14 of the mask to 1, we non-randomly guarantee that at least one alt read
        # (the first) is kept.  If we do torch.remainder(torch.tensor([random integer]), alt counts) we get offsets within each group of
        # alts.  For example if the random integer is 11 the offsets are [2,3,4,1].  Adding these offsets to the zero-based cumsums
        # gives mask indices 2, 6, 11, 15 to set to 1
        random_int = randint(0, 100)

        prepend_zero = torch.tensor([0], device=self.device, dtype=torch.int64)
        ref_bounds = torch.cumsum(torch.hstack((prepend_zero, old_ref_counts)), dim=0)
        alt_bounds = torch.cumsum(torch.hstack((prepend_zero, old_alt_counts)), dim=0)
        alt_cumsums = alt_bounds[:-1]

        alt_override_idx = alt_cumsums + torch.remainder(torch.tensor([random_int], device=self.device, dtype=torch.int64), old_alt_counts)
        keep_alt_mask[alt_override_idx] = 1

        # the alt counts are the sums of the mask within the ranges of each datum
        self.ref_counts = segment_csr(keep_ref_mask, ref_bounds, reduce="sum")
        self.alt_counts = segment_csr(keep_alt_mask, alt_bounds, reduce="sum")
        # randomly assign ref reads to keep

        kept_ref_indices = torch.nonzero(keep_ref_mask).view(-1)
        kept_alt_indices = torch.nonzero(keep_alt_mask).view(-1)

        self.read_indices = torch.hstack((kept_ref_indices, kept_alt_indices))

    #override for downsampled counts
    def get(self, data_field: Data):
        if data_field == Data.REF_COUNT:
            return self.ref_counts
        elif data_field == Data.ALT_COUNT:
            return self.alt_counts
        else:
            return super().get(data_field)

    # override
    def get_int_array_be(self) -> np.ndarray:
        result = self.int_tensor.cpu().numpy(force=True)  # force it to make a copy because we modify it
        result[:, Data.REF_COUNT.idx] = self.ref_counts.cpu().numpy()
        result[:, Data.ALT_COUNT.idx] = self.alt_counts.cpu().numpy()
        return result

    # override
    def get_reads_re(self) -> Tensor:
        return self.reads_re[self.read_indices]



