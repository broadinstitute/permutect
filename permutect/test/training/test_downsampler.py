import itertools

import pytest
import torch
from torch.distributions import Beta

from permutect.data.batch import Batch
from permutect.data.batch import DownsampledBatch
from permutect.data.count_binning import MAX_ALT_COUNT
from permutect.data.count_binning import MAX_REF_COUNT
from permutect.data.count_binning import MIN_ALT_COUNT
from permutect.data.count_binning import NUM_ALT_COUNT_BINS
from permutect.data.count_binning import NUM_REF_COUNT_BINS
from permutect.data.count_binning import alt_count_bin_index
from permutect.data.count_binning import ref_count_bin_index
from permutect.data.datum import Data
from permutect.data.reads_dataset import count_totals_shape
from permutect.misc_utils import backpropagate
from permutect.test.data.test_batch import make_datum
from permutect.training.downsampler import FITTING_LEARNING_RATE
from permutect.training.downsampler import MAX_FITTING_STEPS
from permutect.training.downsampler import Downsampler
from permutect.utils.enums import Label
from permutect.utils.enums import Variation


def test_downsampling_weights_stay_finite_when_a_bin_is_empty():
    torch.manual_seed(0)
    count_totals_slvcd = torch.randint(1, 50, count_totals_shape(num_sources=1)).float()
    count_totals_slvcd[0, Label.VARIANT, Variation.BIG_INSERTION] = 0

    downsampler = Downsampler(num_sources=1)
    downsampler.optimize_downsampling_balance(count_totals_slvcd)

    assert torch.isfinite(downsampler.log_ref_weights_slvrak).all()
    assert torch.isfinite(downsampler.log_alt_weights_slvrah).all()


def test_fitting_stops_once_converged():
    torch.manual_seed(0)
    count_totals_slvcd = torch.randint(0, 50, count_totals_shape(num_sources=1)).float()
    downsampler = Downsampler(num_sources=1)
    num_steps = downsampler.optimize_downsampling_balance(count_totals_slvcd)
    assert num_steps < MAX_FITTING_STEPS
    converged_loss = downsampler.balance_loss(count_totals_slvcd).item()

    # fitting for twice as long again barely improves the balance
    optimizer = torch.optim.Adam(downsampler.parameters(), lr=FITTING_LEARNING_RATE)
    for _ in range(2 * num_steps):
        backpropagate(optimizer, downsampler.balance_loss(count_totals_slvcd))
    assert downsampler.balance_loss(count_totals_slvcd).item() > (1 - 1e-3) * converged_loss


def assert_transitions_match(original_b, downsampled_b, counts, bin_index, num_bins: int, transitions_cb):
    for count, transitions_b in zip(counts, transitions_cb):
        bins = bin_index(downsampled_b[original_b == count]).long()
        frequencies_b = torch.bincount(bins, minlength=num_bins) / len(bins)
        assert torch.allclose(frequencies_b, transitions_b, atol=0.03), f"count {count}"


@pytest.mark.parametrize("basis", range(len(Downsampler.BETA_BASIS_SHAPES)))
def test_transitions_are_those_of_downsampled_batches(basis):
    # the downsampler balances the count bins that it expects DownsampledBatch to give, so its transitions from each ref
    # and alt count must be the distributions of the bins of DownsampledBatch's downsampled counts
    torch.manual_seed(0)
    downsampler = Downsampler(num_sources=1)
    ref_counts, alt_counts = range(MAX_REF_COUNT + 1), range(MIN_ALT_COUNT, MAX_ALT_COUNT + 1)
    # every combination of ref and alt count, many times
    counts = list(itertools.product(ref_counts, alt_counts)) * 1000
    parent = Batch([make_datum(0, ref_count, alt_count) for ref_count, alt_count in counts])
    fractions = Beta(*Downsampler.BETA_BASIS_SHAPES[basis])
    downsampled = DownsampledBatch(
        parent, ref_fracs_b=fractions.sample((parent.size(),)), alt_fracs_b=fractions.sample((parent.size(),))
    )
    assert_transitions_match(
        parent.get(Data.REF_COUNT),
        downsampled.get(Data.REF_COUNT),
        ref_counts,
        ref_count_bin_index,
        NUM_REF_COUNT_BINS,
        downsampler.ref_trans_kcy[basis],
    )
    assert_transitions_match(
        parent.get(Data.ALT_COUNT),
        downsampled.get(Data.ALT_COUNT),
        alt_counts,
        alt_count_bin_index,
        NUM_ALT_COUNT_BINS,
        downsampler.alt_trans_hdz[basis],
    )


def test_expected_downsampled_counts_keep_the_numbers_of_data():
    # downsampling changes the counts of data, not how many data there are, so whatever the weights the expected
    # downsampled counts of each source, label and variant type add up to its number of data
    torch.manual_seed(0)
    count_totals_slvcd = torch.randint(0, 50, count_totals_shape(num_sources=2)).float()
    downsampler = Downsampler(num_sources=2)
    with torch.no_grad():
        for weights in downsampler.weights_parameters():
            weights.normal_()
    expected_slvyz = downsampler.calculate_expected_downsampled_counts(count_totals_slvcd)
    assert torch.allclose(expected_slvyz.sum(dim=(-2, -1)), count_totals_slvcd.sum(dim=(-2, -1)))
