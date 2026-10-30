import torch

from permutect.data.batches_on_device import batches_on_device
from permutect.data.datum import DEFAULT_GPU_FLOAT
from permutect.test.architecture.test_posterior_model_priors import make_batch


def test_batches_on_device(capsys):
    # PyTorch's meta device stands in for a GPU
    batches = [make_batch([0.1, 0.2]), make_batch([0.3])]
    on_device = list(batches_on_device(batches, device=torch.device("meta")))
    assert [batch.size() for batch in on_device] == [2, 1]
    for batch in on_device:
        assert batch.reads_re.device.type == batch.int_tensor.device.type == batch.float_tensor.device.type == "meta"
        assert batch.reads_re.dtype == DEFAULT_GPU_FLOAT

    # nothing is printed, as "Entering prefetch generator..." used to be in every epoch
    assert capsys.readouterr().out == ""
