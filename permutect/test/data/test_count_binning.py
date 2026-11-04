import numpy as np
import torch

from permutect.data.count_binning import MAX_ALT_COUNT
from permutect.data.count_binning import MAX_REF_COUNT
from permutect.data.count_binning import NUM_ALT_COUNT_BINS
from permutect.data.count_binning import NUM_REF_COUNT_BINS
from permutect.data.count_binning import alt_count_bin_index
from permutect.data.count_binning import alt_count_bin_name
from permutect.data.count_binning import count_from_alt_bin_index
from permutect.data.count_binning import count_from_ref_bin_index
from permutect.data.count_binning import ref_count_bin_index
from permutect.data.count_binning import ref_count_bin_name


def test_counts_outside_the_range_go_in_the_end_bins():
    assert ref_count_bin_index(MAX_REF_COUNT + 2) == NUM_REF_COUNT_BINS - 1
    assert alt_count_bin_index(MAX_ALT_COUNT + 1) == NUM_ALT_COUNT_BINS - 1
    # eg posterior data, whose ALT_COUNT is 0
    assert alt_count_bin_index(0) == 0


def test_int_and_tensor_bin_indices_agree():
    counts = list(range(3 * max(MAX_REF_COUNT, MAX_ALT_COUNT)))
    ref_bins, alt_bins = ref_count_bin_index(torch.tensor(counts)), alt_count_bin_index(torch.tensor(counts))
    for count, ref_bin, alt_bin in zip(counts, ref_bins.tolist(), alt_bins.tolist()):
        assert ref_count_bin_index(count) == ref_bin and 0 <= ref_bin < NUM_REF_COUNT_BINS
        assert alt_count_bin_index(count) == alt_bin and 0 <= alt_bin < NUM_ALT_COUNT_BINS
        # the 16-bit integers stored in a Datum
        assert ref_count_bin_index(np.int16(count)) == ref_bin
        assert alt_count_bin_index(np.int16(count)) == alt_bin


def test_bin_names_are_bin_centers():
    for bin_idx in range(NUM_REF_COUNT_BINS):
        center = count_from_ref_bin_index(bin_idx)
        assert ref_count_bin_index(center) == bin_idx
        assert ref_count_bin_name(bin_idx) == str(center)
    for bin_idx in range(NUM_ALT_COUNT_BINS):
        center = count_from_alt_bin_index(bin_idx)
        assert alt_count_bin_index(center) == bin_idx
        assert alt_count_bin_name(bin_idx) == str(center)

    # embedding metrics label each datum with the name of its alt count bin and group data by name
    names = [alt_count_bin_name(bin_idx) for bin_idx in range(NUM_ALT_COUNT_BINS)]
    assert all(alt_count_bin_name(alt_count_bin_index(count)) in names for count in range(3 * MAX_ALT_COUNT))

    bin_indices = torch.arange(NUM_ALT_COUNT_BINS)
    assert count_from_alt_bin_index(bin_indices).tolist() == [count_from_alt_bin_index(n) for n in bin_indices.tolist()]
