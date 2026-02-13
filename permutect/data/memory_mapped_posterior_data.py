from __future__ import annotations
from tempfile import NamedTemporaryFile

import numpy as np
from permutect.data.posterior_data import PosteriorDatum

SUFFIX_FOR_INT16_MMAP = ".int16_data_mmap.npy"
SUFFIX_FOR_FLOAT16_MMAP = ".float16_data_mmap.npy"
SUFFIX_FOR_EXTRA_FLOAT_MMAP = ".extra_float_mmap.npy"
SUFFIX_FOR_EMBEDDING_MMAP = ".embedding_mmap.npy"


class MemoryMappedPosteriorData:
    """
    wrapper for
        1) memory-mapped numpy file for stacked 1D data arrays.  The nth row of this array is the 1D array for the nth Datum.
        2) likewise, memory-mapped numpy file for stacked 1D float arrays.
        3) likewise, memory-mapped numpy file for stacked 1D embeddings.

    This class is quite a bit simpler than the corresponding one for ReadsDatum / ReadsDataset because every memory map
    is a stack of 1D arrays, hence the nth datum is simpler the nth row.

    Also, it never needs to be saved/loaded.  It only exists during filtering and does not need to persist afterwards.
    """

    def __init__(self, int16_mmap, float16_mmap, extra_float_mmap, embedding_mmap, num_data):
        self.int16_mmap = int16_mmap
        self.float16_mmap = float16_mmap
        self.extra_float_mmap = extra_float_mmap
        self.embedding_mmap = embedding_mmap
        self.num_data = num_data

    def __len__(self):
        return self.num_data

    def size_in_bytes(self):
        return self.int16_mmap.nbytes + self.float16_mmap.nbytes + self.extra_float_mmap.nbytes + self.embedding_mmap.nbytes

    @classmethod
    def from_generator(cls, posterior_datum_source, estimated_num_data) -> MemoryMappedPosteriorData:
        """
        Write PosteriorDatum data to memory maps.  We set the file sizes to initial guesses but if these are outgrown we copy
        data to larger files, just like the amortized O(N) append operation on lists.
        """
        num_data, data_capacity = 0, 0
        int16_mmap, float16_mmap, extra_float_mmap, embedding_mmap = None, None, None, None

        datum: PosteriorDatum
        for datum in posterior_datum_source:
            int16_array = datum.get_int16_array()
            float16_array = datum.get_float16_array()
            extract_float_array = datum.extra_float_array
            embedding_array = datum.embedding

            num_data += 1

            # double capacity or set to initial estimate, create new file and mmap, copy old data
            if num_data > data_capacity:
                old_capacity = data_capacity
                data_capacity = estimated_num_data if data_capacity == 0 else data_capacity*2

                int16_file = NamedTemporaryFile(suffix=SUFFIX_FOR_INT16_MMAP)
                float16_file = NamedTemporaryFile(suffix=SUFFIX_FOR_FLOAT16_MMAP)
                extra_float_file = NamedTemporaryFile(suffix=SUFFIX_FOR_EXTRA_FLOAT_MMAP)
                embedding_file = NamedTemporaryFile(suffix=SUFFIX_FOR_EMBEDDING_MMAP)

                old_int16_mmap, old_float16_mmap, old_extra_float_mmap, old_embedding_mmap = int16_mmap, float16_mmap,  extra_float_mmap, embedding_mmap

                # allocate new memory maps
                int16_mmap = np.memmap(int16_file.name, dtype=int16_array.dtype, mode='w+', shape=(data_capacity, int16_array.shape[-1]))
                float16_mmap = np.memmap(float16_file.name, dtype=float16_array.dtype, mode='w+',
                                       shape=(data_capacity, float16_array.shape[-1]))
                extra_float_mmap = np.memmap(extra_float_file.name, dtype=extract_float_array.dtype, mode='w+', shape=(data_capacity, extract_float_array.shape[-1]))
                embedding_mmap = np.memmap(embedding_file.name, dtype=embedding_array.dtype, mode='w+', shape=(data_capacity, embedding_array.shape[-1]))

                # copy the existing data into the new memory maps
                if old_int16_mmap is not None:
                    int16_mmap[:old_capacity] = old_int16_mmap
                    float16_mmap[:old_capacity] = old_float16_mmap
                    extra_float_mmap[:old_capacity] = old_extra_float_mmap
                    int16_mmap[:old_capacity] = old_embedding_mmap


            # write new data
            int16_mmap[num_data - 1] = int16_array
            float16_mmap[num_data - 1] = float16_array
            extra_float_mmap[num_data - 1] = extract_float_array
            embedding_mmap[num_data - 1] = embedding_array

        int16_mmap.flush()
        float16_mmap.flush()
        extra_float_mmap.flush()
        embedding_mmap.flush()

        return cls(int16_mmap=int16_mmap, float16_mmap=float16_mmap, extra_float_mmap=extra_float_mmap, embedding_mmap=embedding_mmap, num_data=num_data)