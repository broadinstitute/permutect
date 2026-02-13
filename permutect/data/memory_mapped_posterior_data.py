from __future__ import annotations
from tempfile import NamedTemporaryFile
import numpy as np
from permutect.data.posterior_data import PosteriorDatum

SUFFIX_FOR_DATA_MMAP = ".data_mmap.npy"
SUFFIX_FOR_FLOAT_MMAP = ".float_mmap.npy"
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

    def __init__(self, data_mmap, float_mmap, embedding_mmap, num_data):
        self.data_mmap = data_mmap
        self.float_mmap = float_mmap
        self.embedding_mmap = embedding_mmap
        self.num_data = num_data

    def __len__(self):
        return self.num_data

    def size_in_bytes(self):
        return self.data_mmap.nbytes + self.float_mmap.nbytes + self.embedding_mmap.nbytes

    @classmethod
    def from_generator(cls, posterior_datum_source, estimated_num_data) -> MemoryMappedPosteriorData:
        """
        Write PosteriorDatum data to memory maps.  We set the file sizes to initial guesses but if these are outgrown we copy
        data to larger files, just like the amortized O(N) append operation on lists.
        """
        num_data, data_capacity = 0, 0
        data_mmap, float_mmap, embedding_mmap = None, None, None

        datum: PosteriorDatum
        for datum in posterior_datum_source:
            data_array = datum.get_int16_array()
            float_array = datum.extra_float_array
            embedding_array = datum.embedding

            num_data += 1

            # double capacity or set to initial estimate, create new file and mmap, copy old data
            if num_data > data_capacity:
                old_capacity = data_capacity
                data_capacity = estimated_num_data if data_capacity == 0 else data_capacity*2

                data_file = NamedTemporaryFile(suffix=SUFFIX_FOR_DATA_MMAP)
                float_file = NamedTemporaryFile(suffix=SUFFIX_FOR_FLOAT_MMAP)
                embedding_file = NamedTemporaryFile(suffix=SUFFIX_FOR_EMBEDDING_MMAP)

                old_data_mmap, old_float_mmap, old_embedding_mmap = data_mmap, float_mmap, embedding_mmap

                # allocate new memory maps
                data_mmap = np.memmap(data_file.name, dtype=data_array.dtype, mode='w+', shape=(data_capacity, data_array.shape[-1]))
                float_mmap = np.memmap(float_file.name, dtype=float_array.dtype, mode='w+', shape=(data_capacity, float_array.shape[-1]))
                embedding_mmap = np.memmap(embedding_file.name, dtype=embedding_array.dtype, mode='w+', shape=(data_capacity, embedding_array.shape[-1]))

                # copy the existing data into the new memory maps
                if old_data_mmap is not None:
                    data_mmap[:old_capacity] = old_data_mmap
                    float_mmap[:old_capacity] = old_float_mmap
                    data_mmap[:old_capacity] = old_embedding_mmap


            # write new data
            data_mmap[num_data - 1] = data_array
            float_mmap[num_data - 1] = float_array
            embedding_mmap[num_data - 1] = embedding_array

        data_mmap.flush()
        float_mmap.flush()
        embedding_mmap.flush()

        return cls(data_mmap=data_mmap, float_mmap=float_mmap, embedding_mmap=embedding_mmap, num_data=num_data)