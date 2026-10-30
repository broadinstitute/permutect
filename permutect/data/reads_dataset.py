import random
from typing import List

import numpy as np
import psutil
import torch
from torch.utils.data import DataLoader
from torch.utils.data import IterableDataset

from permutect.data.batch import Batch
from permutect.data.batch import BatchIndexedTensor
from permutect.data.batch import BatchProperty
from permutect.data.datum import COMPRESSED_READS_ARRAY_DTYPE
from permutect.data.datum import Datum
from permutect.data.memory_mapped_data import MemoryMappedData
from permutect.misc_utils import ConsistentValue
from permutect.misc_utils import Timer
from permutect.misc_utils import report_memory_usage
from permutect.utils.enums import Label
from permutect.utils.enums import Variation

WEIGHT_PSEUDOCOUNT = 10


# it is often convenient to arbitrarily use the last fold for validation
def last_fold_only(num_folds: int):
    return [num_folds - 1]  # use the last fold for validation


def all_but_last_fold(num_folds: int):
    return list(range(num_folds - 1))


def all_but_one_fold(num_folds: int, fold_to_exclude: int):
    return list(range(fold_to_exclude)) + list(range(fold_to_exclude + 1, num_folds))


def all_folds(num_folds: int):
    return list(range(num_folds))


def ratio_with_pseudocount(a, b):
    return (a + WEIGHT_PSEUDOCOUNT) / (b + WEIGHT_PSEUDOCOUNT)


class ReadsDataset(IterableDataset):
    def __init__(
        self,
        memory_mapped_data: MemoryMappedData,
        num_folds: int = 1,
        folds_to_use: List[int] = None,
        keep_probs_by_label_l=None,
    ):
        """
        :param num_folds:
        """
        super(ReadsDataset, self).__init__()
        self.totals_slvra = BatchIndexedTensor.zeros(num_sources=1, include_logits=False, device=torch.device("cpu"))
        # if no folds, no copying is done; otherwise this creates a new file on disk
        self.memory_mapped_data = memory_mapped_data.restrict_to_folds(num_folds, folds_to_use, keep_probs_by_label_l)
        self._size = self.memory_mapped_data.num_data
        self._read_offsets = self.memory_mapped_data.read_offsets

        available_memory = psutil.virtual_memory().available
        print(
            f"Data occupy {memory_mapped_data.num_bytes() // 1000000} Mb and the system has {available_memory // 1000000} Mb of RAM available."
        )

        self._num_read_features, self._num_info_features, self._haplotypes_length = (
            ConsistentValue(),
            ConsistentValue(),
            ConsistentValue(),
        )
        data_recording_timer = Timer("Recording data counts. . .")
        for datum in self.memory_mapped_data.generate():
            self.totals_slvra.record_datum(datum)
            self._num_read_features.check(datum.num_read_features())
            self._num_info_features.check(len(datum.get_info_1d()))
            self._haplotypes_length.check(datum.get_haplotypes_array_size())
        data_recording_timer.report("Time to record data counts")

        self.totals_by_label_l = self.totals_slvra.get_marginal(BatchProperty.LABEL)  # totals by label

    def totals_by_label(self):
        return self.totals_by_label_l

    def num_read_features(self) -> int:
        return self._num_read_features.value

    def num_info_features(self) -> int:
        return self._num_info_features.value

    def haplotypes_length(self) -> int:
        return self._haplotypes_length.value

    # this is not required for an IterableDataset, but it can't hurt!
    def __len__(self):
        return self._size

    def __iter__(self):
        print("Inside a ReadsDataset's .__iter__ function")
        worker_info = torch.utils.data.get_worker_info()
        worker_id = 0 if worker_info is None else worker_info.id
        num_workers = 1 if worker_info is None else worker_info.num_workers

        num_data_per_worker = self._size // num_workers
        num_bytes_per_worker = self.memory_mapped_data.num_bytes() // num_workers

        # note: this is the total available system memory, not per process
        total_available_memory_in_bytes = psutil.virtual_memory().available
        available_memory_per_worker = total_available_memory_in_bytes // num_workers

        # we want the amount of memory loaded to RAM at any given time to be well below the total available memory
        # thus we multiply by a cautious fudge factor that accounts for: 1) maybe space in RAM is less efficient than space in disk,
        # 2) one chunk might still be in memory, not yet garbage-collected, when the next is loaded
        fudge_factor = 8
        chunks_per_worker = 1 + ((fudge_factor * num_bytes_per_worker) // available_memory_per_worker)
        print(f"Worker {worker_id} will divide its portion of the data into {chunks_per_worker} chunks.")

        # The way multiple DataLoader workers work in PyTorch is not so obvious.
        # 1) There is an original Dataset
        # 2) The DataLoader gives each worker process its own copy of the Dataset.  With fork (the Linux default through
        #    Python 3.13) a worker shares the original's memory copy-on-write.  With spawn (the macOS default) or
        #    forkserver (the Linux default from Python 3.14) it gets a pickled copy, in which numpy arrays are copied,
        #    tensors are moved to shared memory, and memory maps are reopened from their files rather than copied.
        # 3) With num_workers > 0 the DataLoader calls iter() in each worker on its copy, which is responsible for only
        #    part of the memory-mapped data.  With num_workers = 0, the default, iter() runs on the original Dataset in
        #    the main process, which is responsible for all of it.
        # 4) this part of the memory-mapped data might be too big for available RAM, so we load one contiguous chunk
        # (this is a chunk within the subset of data that this worker is responsible for) into RAM at a time.

        # example of specific number: num_data = 21, num_workers = 4, then data_per_worker is 5 and the index ranges
        # are [0,5), [5,10), [10,15), [15,21)
        worker_start_idx = worker_id * num_data_per_worker
        worker_end_idx = (worker_id + 1) * num_data_per_worker if worker_id < num_workers - 1 else self._size

        num_data_for_this_worker = worker_end_idx - worker_start_idx
        data_per_chunk = num_data_for_this_worker // chunks_per_worker
        chunks = list(range(chunks_per_worker))
        random.shuffle(chunks)

        if worker_info is None:
            print("Iterating over the whole dataset in a single process.")
        else:
            print(f"Iterating with worker {worker_id} out of {num_workers}.")
            print(f"This worker is responsible for data range [{worker_start_idx}, {worker_end_idx}).")

        # from memory_mapped_data rather than attributes of their own, which would be pickled as copies of all the data
        int_array_ve, float_array_ve = self.memory_mapped_data.int_mmap, self.memory_mapped_data.float_mmap
        reads_mmap = self.memory_mapped_data.reads_mmap
        stacked_reads_re = np.zeros((0, 0), dtype=COMPRESSED_READS_ARRAY_DTYPE) if reads_mmap is None else reads_mmap

        for chunk in chunks:
            chunk_start_idx = worker_start_idx + chunk * data_per_chunk
            chunk_end_idx = (
                (worker_start_idx + (chunk + 1) * data_per_chunk) if (chunk < chunks_per_worker - 1) else worker_end_idx
            )

            chunk_read_start_idx = self._read_offsets[chunk_start_idx]
            chunk_read_end_idx = self._read_offsets[chunk_end_idx]

            ram_timer = Timer(f"Worker {worker_id} loading chunk [{chunk_start_idx}, {chunk_end_idx}) into RAM.")
            # copying reads each chunk from the memory-mapped files in one sequential pass, so that the shuffled access
            # below is to RAM rather than to the files.  The copies aren't needed for correctness: a slice of a memory
            # map is read from its file when used, and Datum copies its int and float arrays anyway.
            chunk_int_data_ve = int_array_ve[chunk_start_idx:chunk_end_idx].copy()
            chunk_float_data_ve = float_array_ve[chunk_start_idx:chunk_end_idx].copy()
            chunk_reads_re = stacked_reads_re[chunk_read_start_idx:chunk_read_end_idx].copy()
            chunk_read_offsets = self._read_offsets[chunk_start_idx : chunk_end_idx + 1] - chunk_read_start_idx
            ram_timer.report("Time to load chunk data into RAM")
            report_memory_usage("Chunk data loaded into RAM.")

            # now that it's all in RAM, we can yield in randomly-accessed order
            indices = list(range(len(chunk_int_data_ve)))
            random.shuffle(indices)

            for idx in indices:
                datum = Datum(
                    int_array=chunk_int_data_ve[idx],
                    float_array=chunk_float_data_ve[idx],
                    reads_re=chunk_reads_re[chunk_read_offsets[idx] : chunk_read_offsets[idx + 1]],
                    compressed=True,
                )
                # assert datum.get_ref_count() + datum.get_alt_count() == len(datum.get_reads_array_re())
                yield datum

            # free this chunk's arrays before the next chunk is copied into RAM, rather than only when the names are
            # reassigned after that copy.  Each Datum's reads are views of chunk_reads_re, which therefore stays in
            # memory until this chunk's data have all been batched.
            del chunk_int_data_ve
            del chunk_float_data_ve
            del chunk_reads_re

    def num_sources(self) -> int:
        return self.totals_slvra.num_sources()

    def report_totals(self):
        totals_slv = self.totals_slvra.get_marginal(
            BatchProperty.SOURCE, BatchProperty.LABEL, BatchProperty.VARIANT_TYPE
        )
        for source in range(len(totals_slv)):
            print(f"Data counts for source {source}:")
            for var_type in Variation:
                print(f"Data counts for variant type {var_type.name}:")
                for label in Label:
                    print(f"{label.name}: {int(totals_slv[source, label, var_type].item())}")

    def validate_sources(self) -> int:
        num_sources = self.num_sources()
        totals_by_source_s = self.totals_slvra.get_marginal(BatchProperty.SOURCE)
        if num_sources == 1:
            print("Data come from a single source")
        else:
            for source in range(self.num_sources()):
                assert totals_by_source_s[source].item() >= 1, f"No data for source {source}."
            print(f"Data come from multiple sources, with counts {totals_by_source_s.cpu().tolist()}.")
        return num_sources

    def make_data_loader(self, batch_size: int, pin_memory=False, num_workers: int = 0):
        return DataLoader(
            dataset=self,
            batch_size=batch_size,
            collate_fn=Batch,
            pin_memory=pin_memory,
            num_workers=num_workers,
            prefetch_factor=2 if num_workers > 0 else None,
            persistent_workers=num_workers > 0,
        )


# ex: chunk([a,b,c,d,e], 3) = [[a,b,c], [d,e]]
def chunk(lis, chunk_size):
    return [lis[i : i + chunk_size] for i in range(0, len(lis), chunk_size)]
