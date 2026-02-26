from torch.utils.data import DataLoader

from permutect.data.batch import Batch
from permutect.data.datum import DEFAULT_GPU_FLOAT, DEFAULT_CPU_FLOAT
from permutect.misc_utils import gpu_if_available


def prefetch_generator(dataloader: DataLoader, device=gpu_if_available()):
    """
    prefetch and send batches to GPU in the background
    dataloader must yield batches that have a copy_to method
    """
    print("Entering prefetch generator...")
    is_cuda = device.type == 'cuda'
    dtype = DEFAULT_GPU_FLOAT if is_cuda else DEFAULT_CPU_FLOAT
    batch_cpu: Batch
    for batch_cpu in dataloader:
        batch_gpu = batch_cpu.copy_to(device, dtype=dtype)
        yield batch_gpu