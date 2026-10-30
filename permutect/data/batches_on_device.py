from torch.utils.data import DataLoader

from permutect.data.batch import Batch
from permutect.misc_utils import float_dtype
from permutect.misc_utils import resolve_device


def batches_on_device(dataloader: DataLoader, device=None):
    """
    Yield the batches of a DataLoader copied to a device, which should be the device of the model that uses them.
    Nothing is fetched ahead here: the DataLoader's workers do that, and on a GPU the loaders pin memory, so that the
    copies don't block.
    """
    device = resolve_device(device)
    dtype = float_dtype(device)
    batch_cpu: Batch
    for batch_cpu in dataloader:
        yield batch_cpu.copy_to(device, dtype=dtype)
