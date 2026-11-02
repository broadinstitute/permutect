import torch

from permutect.data.batch import BatchIndexedTensor
from permutect.training.downsampler import Downsampler
from permutect.utils.enums import Label
from permutect.utils.enums import Variation


def test_downsampling_weights_stay_finite_when_a_bin_is_empty():
    torch.manual_seed(0)
    counts = BatchIndexedTensor(torch.randint(1, 50, BatchIndexedTensor.shape_without_logits(num_sources=1)).float())
    counts[0, Label.VARIANT, Variation.BIG_INSERTION] = 0

    downsampler = Downsampler(num_sources=1)
    downsampler.optimize_downsampling_balance(counts)

    assert torch.isfinite(downsampler.log_ref_weights_slvrak).all()
    assert torch.isfinite(downsampler.log_alt_weights_slvrah).all()
