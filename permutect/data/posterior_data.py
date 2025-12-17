from __future__ import annotations

import copy
import gc
import random
from typing import List

import psutil
import torch
from torch import IntTensor, Tensor
from torch.utils.data import Dataset, DataLoader, IterableDataset

from permutect.data.batch import Batch
from permutect.data.datum import Datum
from permutect.data.memory_mapped_posterior_data import MemoryMappedPosteriorData
from permutect.misc_utils import Timer, report_memory_usage


class PosteriorDatum(Datum):

    ALLELE_FREQUENCY = 0
    ARTIFACT_LOGIT = 1
    MAF = 2
    NORMAL_MAF = 3

    def __init__(self, datum_array, float_array, embedding: Tensor):
        super().__init__(datum_array)
        self.embedding = embedding
        self.float_array = float_array


    @classmethod
    def create(cls, datum_array, allele_frequency: float, artifact_logit: float, maf: float, normal_maf: float, embedding: Tensor):
        float_array = torch.zeros(4, dtype=torch.float16)
        float_array[PosteriorDatum.ALLELE_FREQUENCY] = allele_frequency
        float_array[PosteriorDatum.ARTIFACT_LOGIT] = artifact_logit
        float_array[PosteriorDatum.MAF] = maf
        float_array[PosteriorDatum.NORMAL_MAF] = normal_maf
        return cls(datum_array, float_array, embedding)

    def get_artifact_logit(self) -> float:
        return self.float_array[self.__class__.ARTIFACT_LOGIT]


class PosteriorBatch(Batch):

    def __init__(self, data: List[PosteriorDatum]):
        super().__init__(data)
        self.embeddings = torch.vstack([item.embedding for item in data]).float()
        self.float_tensor = torch.vstack([item.float_array for item in data]).float()

    def pin_memory(self):
        super().pin_memory()
        self.embeddings = self.embeddings.pin_memory()
        self.float_tensor = self.float_tensor.pin_memory()
        return self

    # dtype is just for floats!!! Better not convert the int tensor to a float accidentally!
    def copy_to(self, device, dtype):
        is_cuda = device.type == 'cuda'
        new_batch = copy.copy(self)
        new_batch.data = self.data.to(device, non_blocking=is_cuda)  # don't cast dtype -- needs to stay integral!
        new_batch.embeddings = self.embeddings.to(device=device, dtype=dtype, non_blocking=is_cuda)
        new_batch.float_tensor = self.float_tensor.to(device=device, dtype=dtype, non_blocking=is_cuda)
        return new_batch

    def get_allele_frequencies(self) -> Tensor:
        return self.float_tensor[:, PosteriorDatum.ALLELE_FREQUENCY]

    def get_artifact_logits(self) -> Tensor:
        return self.float_tensor[:, PosteriorDatum.ARTIFACT_LOGIT]

    def get_mafs(self) -> Tensor:
        return self.float_tensor[:, PosteriorDatum.MAF]

    def get_normal_mafs(self) -> Tensor:
        return self.float_tensor[:, PosteriorDatum.NORMAL_MAF]

    def get_original_normal_ref_counts(self) -> IntTensor:
        return self.get_original_normal_depths() - self.get_original_normal_alt_counts()


class PosteriorDataset(IterableDataset):
    def __init__(self, posterior_mmap: MemoryMappedPosteriorData):
        super(PosteriorDataset, self).__init__()
        self.memory_map = posterior_mmap
        self._size = len(posterior_mmap)

        available_memory = psutil.virtual_memory().available
        print(f"Posterior data occupy {posterior_mmap.size_in_bytes() // 1000000} Mb and the system has {available_memory // 1000000} Mb of RAM available.")
        self._stacked_data_ve = self.memory_map.data_mmap
        self._stacked_floats_ve = self.memory_map.float_mmap
        self._stacked_embeddings_ve = self.memory_map.embedding_mmap

    def __len__(self) -> int:
        return len(self._size)


    def __iter__(self):
        """
        See documentation for ReadsDataset's __iter__ function for details on how multiple workers work
        """
        print("Inside a PosteriorDataset's .__iter__ function")
        worker_info = torch.utils.data.get_worker_info()
        worker_id = 0 if worker_info is None else worker_info.id
        num_workers = 1 if worker_info is None else worker_info.num_workers

        num_data_per_worker = self._size // num_workers
        num_bytes_per_worker = self.memory_map.size_in_bytes() // num_workers

        # note: this is the total available system memory, not per process
        total_available_memory_in_bytes = psutil.virtual_memory().available
        available_memory_per_worker = total_available_memory_in_bytes // num_workers

        # we want the amount of memory loaded to RAM at any given time to be well below the total available memory
        # thus we multiply by a cautious fudge factor that accounts for: 1) maybe space in RAM is less efficient than space in disk,
        # 2) one chunk might still be in memory, not yet garbage-collected, when the next is loaded
        fudge_factor = 8
        chunks_per_worker = 1 + ((fudge_factor * num_bytes_per_worker) // available_memory_per_worker)
        print(f"Worker {worker_id} will divide its portion of the data into {chunks_per_worker} chunks.")

        worker_start_idx = worker_id * num_data_per_worker
        worker_end_idx = (worker_id + 1) * num_data_per_worker if worker_id < num_workers - 1 else self._size

        num_data_for_this_worker = worker_end_idx - worker_start_idx
        data_per_chunk = num_data_for_this_worker // chunks_per_worker
        chunks = list(range(chunks_per_worker))
        random.shuffle(chunks)

        if worker_info is None:
            print("Iterating over the whole posterior dataset in a single process.")
        else:
            print(f"Iterating over posterior data with worker {worker_id} out of {num_workers}.")
            print(f"This worker is responsible for data range [{worker_start_idx}, {worker_end_idx}).")

        for chunk in chunks:
            chunk_start_idx = worker_start_idx + chunk * data_per_chunk
            chunk_end_idx = (worker_start_idx + (chunk + 1) * data_per_chunk) if (chunk < chunks_per_worker - 1) else worker_end_idx


            ram_timer = Timer(f"Worker {worker_id} loading posterior data chunk [{chunk_start_idx}, {chunk_end_idx}) into RAM.")
            # TODO: I think the .copy() is necessary to copy the slice of the memory-map from disk into RAM
            # these operations should be really fast because it's all sequential access
            chunk_data_ve = self._stacked_data_ve[chunk_start_idx:chunk_end_idx].copy()
            chunk_floats_ve = self._stacked_floats_ve[chunk_start_idx:chunk_end_idx].copy()
            chunk_embeddings_ve = self._stacked_embeddings_ve[chunk_start_idx:chunk_end_idx]

            ram_timer.report("Time to load chunk data into RAM")
            report_memory_usage("Chunk data loaded into RAM.")

            # now that it's all in RAM, we can yield in randomly-accessed order
            indices = list(range(len(chunk_data_ve)))
            random.shuffle(indices)

            for idx in indices:
                datum = PosteriorDatum(datum_array=chunk_data_ve[idx], float_array=chunk_floats_ve[idx], embedding=chunk_embeddings_ve[idx])
                yield datum

            # we have finished yielding all the data in this chunk.  Because this is such a large amount of data,
            # we explicitly free memory (delete objects and garbage collect) before loading the next chunk
            del chunk_data_ve
            del chunk_floats_ve
            del chunk_embeddings_ve
            del indices
            gc.collect()


    def make_data_loader(self, batch_size: int, pin_memory=False, num_workers: int = 0):
        return DataLoader(dataset=self, batch_size=batch_size, collate_fn=PosteriorBatch, pin_memory=pin_memory,
                          num_workers=num_workers, prefetch_factor=2 if num_workers > 0 else None, persistent_workers=num_workers > 0)
