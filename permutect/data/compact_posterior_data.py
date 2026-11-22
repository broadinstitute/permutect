"""
The posterior data that the posterior model reads, held as tensors, on the GPU when they fit, so that fitting the spectra
and finding the error probability thresholds don't build a Datum and a Batch of every variant in every pass.
"""

import math

import numpy as np
import torch

from permutect.data.batch import Batch
from permutect.data.datum import Data
from permutect.data.memory_mapped_data import MemoryMappedData
from permutect.misc_utils import resolve_device

# the most of a GPU's free memory that the data may take, leaving the rest for fitting the spectra
MAX_FRACTION_OF_FREE_GPU_MEMORY = 0.25


def central_haplotype_columns(num_int_columns: int) -> list[int]:
    """
    The columns of a datum's int array that hold the three central bases of its ref haplotype and then of its alt
    haplotype, which are all of the haplotypes that the posterior model reads: an SNV's reference context and alt base.
    Kept in place of the whole haplotypes, they are haplotypes of length 3 with the variant in the middle, so that
    Batch.get_haplotypes_bs and get_ref_contexts_and_alt_bases work on them unchanged.  Haplotypes shorter than 3 bases
    are kept whole.
    """
    seq_length = (num_int_columns - Data.HAPLOTYPES_START_IDX) // 2
    if seq_length < 3:
        return list(range(Data.HAPLOTYPES_START_IDX, num_int_columns))
    center = (seq_length - 1) // 2
    ref_columns = [Data.HAPLOTYPES_START_IDX + center + offset for offset in (-1, 0, 1)]
    return ref_columns + [column + seq_length for column in ref_columns]


class CompactPosteriorData:
    """
    The fields of posterior data that the posterior model reads: the scalar int and float fields and the three central
    bases of each haplotype, without the rest of the haplotypes or the INFO vector of embeddings.  They're int16 and
    float16, as in the memory-mapped data, 70 bytes per variant, eg 70 MB for a million variants.  They're kept on a GPU
    if they take at most MAX_FRACTION_OF_FREE_GPU_MEMORY of its free memory, and otherwise in pinned memory on the host,
    from which each batch is copied.

    Iterating gives batches on the device in a new random order each time, as a DataLoader of the posterior ReadsDataset
    does, but without building a Datum and a Batch of every variant on the host.  The order comes from torch's global
    random number generator, which filter_variants seeds, so that it's the same in every run.
    """

    def __init__(
        self, int_array_ve: np.ndarray, float_array_ve: np.ndarray, batch_size: int, device=None, max_device_bytes=None
    ):
        """
        :param int_array_ve, float_array_ve: the compact int and float arrays, one row per variant
        :param max_device_bytes: the most memory that the data may take on the device, by default
            MAX_FRACTION_OF_FREE_GPU_MEMORY of a GPU's free memory and unlimited on other devices
        """
        self.device = resolve_device(device)
        self.batch_size = batch_size
        self.size = len(int_array_ve)
        self.nbytes = int_array_ve.nbytes + float_array_ve.nbytes
        if max_device_bytes is None and self.device.type == "cuda":
            max_device_bytes = MAX_FRACTION_OF_FREE_GPU_MEMORY * torch.cuda.mem_get_info(self.device)[0]
        on_device = max_device_bytes is None or self.nbytes <= max_device_bytes
        # pinned memory lets each batch's copy to a GPU run while the GPU works on the previous batch
        self._pinned = not on_device and self.device.type == "cuda"

        int_tensor_ve, float_tensor_ve = torch.from_numpy(int_array_ve), torch.from_numpy(float_array_ve)
        if on_device:
            int_tensor_ve, float_tensor_ve = int_tensor_ve.to(self.device), float_tensor_ve.to(self.device)
        elif self._pinned:
            int_tensor_ve, float_tensor_ve = int_tensor_ve.pin_memory(), float_tensor_ve.pin_memory()
        self.int_tensor_ve, self.float_tensor_ve = int_tensor_ve, float_tensor_ve
        print(f"Posterior data for the spectra: {self.nbytes / 1e6:.0f} MB on {int_tensor_ve.device}")

    @classmethod
    def from_memory_mapped_data(
        cls, memory_mapped_data: MemoryMappedData, batch_size: int, device=None, max_device_bytes=None
    ) -> "CompactPosteriorData":
        int_mmap, float_mmap = memory_mapped_data.int_mmap, memory_mapped_data.float_mmap
        int_columns = list(range(Data.NUM_SCALAR_INT_ELEMENTS)) + central_haplotype_columns(int_mmap.shape[1])
        # plain arrays in RAM, copied in one pass over the memory-mapped files
        int_array_ve = np.array(int_mmap[:, int_columns])
        float_array_ve = np.array(float_mmap[:, : Data.NUM_SCALAR_FLOAT_ELEMENTS])
        return cls(int_array_ve, float_array_ve, batch_size, device, max_device_bytes)

    def __len__(self) -> int:
        return math.ceil(self.size / self.batch_size)

    def __iter__(self):
        # a permutation from torch's global generator on the CPU, so that the order doesn't depend on the device
        order = torch.randperm(self.size).to(self.int_tensor_ve.device)
        for start in range(0, self.size, self.batch_size):
            indices = order[start : start + self.batch_size]
            int_be, float_be = self.int_tensor_ve[indices], self.float_tensor_ve[indices]
            if self._pinned:
                int_be, float_be = int_be.pin_memory(), float_be.pin_memory()
            # copied as int16 and float16 and widened on the device, as a Batch widens them on the host
            int_be = int_be.to(self.device, non_blocking=True).long()
            float_be = float_be.to(self.device, non_blocking=True).float()
            yield Batch.without_reads(int_be, float_be)
