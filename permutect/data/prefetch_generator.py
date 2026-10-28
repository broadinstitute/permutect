from torch.utils.data import DataLoader

from permutect.data.batch import Batch
from permutect.misc_utils import float_dtype
from permutect.misc_utils import resolve_device


def prefetch_generator(dataloader: DataLoader, device=None):
    """
    prefetch and send batches to GPU in the background
    dataloader must yield batches that have a copy_to method
    device should be the device of the model that uses the batches
    """
    print("Entering prefetch generator...")
    device = resolve_device(device)
    dtype = float_dtype(device)
    batch_cpu: Batch
    for batch_cpu in dataloader:
        batch_gpu = batch_cpu.copy_to(device, dtype=dtype)
        yield batch_gpu

    """
    loader_iter = iter(dataloader)
    print("about to get the first batch for the prefetch generator...")
    next_batch_cpu = next(loader_iter, None)
    next_batch = None if next_batch_cpu is None else next_batch_cpu.copy_to(device=device, dtype=dtype)
    print("Iterating over the rest of the loader in the prefetch generator...")
    for _ in range(len(dataloader)):
        # the prefetched + sent-to-GPU batch is processed
        batch = next_batch
        # but in the background we'll fetch the next batch and send to GPU
        # the default None will come up on the final batch where there is no next batch to prefetch
        next_batch_cpu = next(loader_iter, None)
        next_batch = None if next_batch_cpu is None else next_batch_cpu.copy_to(device, dtype=dtype)
        yield batch
    """
