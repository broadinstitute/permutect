from __future__ import annotations

import numpy as np
import torch

from permutect.data.datum import Datum, FLOAT_DTYPE, Data

# base strings longer than this when encoding data


COMPRESSED_READS_ARRAY_DTYPE = np.uint8


# quantile-normalized data is generally some small number of standard deviations from 0.  We can store as uint8 by
# mapping x --> 32x + 128 and restricting to the range [0,255], which maps -4 and +4 standard deviations to the limits
# of uint8
def convert_quantile_normalized_to_uint8(data: np.ndarray):
    return np.ndarray.astype(np.clip(data * 32 + 128, 0, 255), np.uint8)


# the inverse of the above
def convert_uint8_to_quantile_normalized(data: np.ndarray):
    return np.ndarray.astype((data - 128) / 32, FLOAT_DTYPE)


def make_sequence_tensor(sequence_string: str) -> np.ndarray:
    """
    convert string of form ACCGTA into 4-channel one-hot tensor
    [ [1, 0, 0, 0, 0, 1],   # A channel
      [0, 1, 1, 0, 0, 0],   # C channel
      [0, 0, 0, 1, 0, 0],   # G channel
      [0, 0, 0, 0, 1, 0] ]  # T channel
    """
    result = np.zeros([4, len(sequence_string)])
    for n, char in enumerate(sequence_string):
        channel = 0 if char == 'A' else (1 if char == 'C' else (2 if char == 'G' else 3))
        result[channel, n] = 1
    return result



class ReadsDatum(Datum):
    # TODO: switch all constructor invocations to Datum
    # TODO: make sure that shape=(0,0) and dtype = COMPRESSED (which we use for posterior model) is okay
    def __init__(self, int_array: np.ndarray, float_array: np.ndarray, reads_re: np.ndarray):
        super().__init__(int_array, float_array)
        assert reads_re.dtype == COMPRESSED_READS_ARRAY_DTYPE

        # Reads are in a compressed, unusable form.  Binary columns must be unpacked and float
        # columns must be transformed back from uint8
        self.reads_re = reads_re







