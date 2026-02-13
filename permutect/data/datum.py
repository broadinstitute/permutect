from __future__ import annotations

import enum
import numpy as np
import torch

from permutect.utils.allele_utils import bases_as_base5_int, bases5_as_base_string, get_ref_and_alt_sequences
from permutect.utils.enums import Label, Variation

# the range is -32,768 to 32,767
# this is sufficient for the count, length, and enum variables, as well as the floats (multiplied by something like 100
# and rounded to the nearest integer)
# haplotypes are represented as A = 0, C = 1, G = 2, T = 3 so 16 bits are easily enough (and we could compress further)
# the position needs 32 bits (to get up to 2 billion or so) so we give it two int16s
# the ref and alt alleles also need 32 bits to handle up to 13 bases
DATUM_ARRAY_DTYPE = np.int16
BIGGEST_UINT16 = 65535
BIGGEST_INT16 = 32767
FLOAT_TO_LONG_MULTIPLIER = 30
MAX_FLOAT = BIGGEST_INT16 / FLOAT_TO_LONG_MULTIPLIER

DEFAULT_NUMPY_FLOAT = np.float16
DEFAULT_GPU_FLOAT = torch.float32
DEFAULT_CPU_FLOAT = torch.float32

def float_to_clipped_int16(float_number: float) -> int:
    unbounded_int = round(float_number * FLOAT_TO_LONG_MULTIPLIER)
    return max(min(unbounded_int, BIGGEST_INT16), -BIGGEST_INT16)


def int16_to_float(int16_number_or_tensor):
    return int16_number_or_tensor / FLOAT_TO_LONG_MULTIPLIER


def uint32_to_two_int16s(num: int):
    uint16_1, uint16_2 = num // BIGGEST_UINT16, num % BIGGEST_UINT16
    return uint16_1 - (BIGGEST_INT16 + 1), uint16_2 - (BIGGEST_INT16 + 1)


def uint32_from_two_int16s(int16_1, int16_2):
    shifted1, shifted2 = int16_1 + (BIGGEST_INT16 + 1), int16_2 + (BIGGEST_INT16 + 1)
    return BIGGEST_UINT16 * shifted1 + shifted2

class Data(enum.Enum):
    # int16 array elements
    REF_COUNT = (np.uint16, 0)
    ALT_COUNT = (np.uint16, 1)
    HAPLOTYPES_LENGTH = (np.uint16, 2)      # TODO: this will be obsolete
    INFO_LENGTH = (np.uint16, 3)            # TODO: this will be obsolete
    LABEL = (np.uint16, 4)
    VARIANT_TYPE = (np.uint16, 5)
    SOURCE = (np.uint16, 6)
    ORIGINAL_DEPTH = (np.uint16, 7)
    ORIGINAL_ALT_COUNT = (np.uint16, 8)
    ORIGINAL_NORMAL_DEPTH = (np.uint16, 9)
    ORIGINAL_NORMAL_ALT_COUNT = (np.uint16, 10)
    CONTIG = (np.uint16, 11)
    POSITION = (np.uint32, 12)              # NOTE: uint32 takes TWO uint16s!
    REF_ALLELE_AS_BASE_5 = (np.uint32, 14)  # NOTE: uint32 takes TWO uint16s!
    ALT_ALLELE_AS_BASE_5 = (np.uint32, 16)  # NOTE: uint32 takes TWO uint16s!
    # after this, at the end of the int16 array comes the sub-array containing the ref sequence haplotype

    # float16 array elements
    SEQ_ERROR_LOG_LK = (np.float16, 0)
    NORMAL_SEQ_ERROR_LOG_LK = (np.float16, 1)
    # after this, at the end of the float16 array comes the sub-array containing the INFO vector

    def __init__(self, dtype: np.dtype, idx: int):
        self.dtype = dtype
        self.idx = idx

Data.NUM_SCALAR_INT16_ELEMENTS = 18    # in Python 3.11+ can use enum.nonmember
Data.HAPLOTYPES_START_IDX = 18          # in Python 3.11+ can use enum.nonmember
Data.NUM_SCALAR_FLOAT16_ELEMENTS = 2    # in Python 3.11+ can use enum.nonmember
Data.INFO_START_IDX = 2          # in Python 3.11+ can use enum.nonmember
class Datum:
    """
    contains data that apply to a candidate mutation as a whole i.e. not the read sets.  These are organized into a single
    LongTensor, containing some quantities that are inherently integral and some that are cast as longs by multiplying
    with a large number and rounding.
    """
    
    def __init__(self, int16_array: np.ndarray, float16_array: np.ndarray):
        # note: this constructor does no checking eg of whether the arrays are consistent with their purported lengths
        # or of whether ref, alt alleles have been trimmed
        assert int16_array.ndim == 1 and len(int16_array) >= Data.NUM_SCALAR_INT16_ELEMENTS
        self.int16_array: np.ndarray = np.ndarray.astype(int16_array, np.int16)
        assert float16_array.ndim == 1 and len(float16_array) >= Data.NUM_SCALAR_FLOAT16_ELEMENTS
        self.float16_array: np.ndarray = np.ndarray.astype(float16_array, np.float16)

    @classmethod
    def make_datum_without_reads(cls, label: Label, variant_type: Variation, source: int,
        original_depth: int, original_alt_count: int, original_normal_depth: int, original_normal_alt_count: int,
        contig: int, position: int, ref_allele: str, alt_allele: str,
        seq_error_log_lk: float, normal_seq_error_log_lk: float, ref_seq_array: np.ndarray, info_array: np.ndarray) -> Datum:
        """
        We are careful about our float to long conversions here and in the getters!
        """
        ref_hap, alt_hap = get_ref_and_alt_sequences(ref_seq_array, ref_allele, alt_allele)
        assert len(ref_hap) == len(ref_seq_array) and len(alt_hap) == len(ref_seq_array)
        haplotypes = np.hstack((ref_hap, alt_hap))

        haplotypes_length, info_length = len(haplotypes), len(info_array)
        zeroed_int16_array = np.zeros(Data.NUM_SCALAR_INT16_ELEMENTS + haplotypes_length, dtype=np.int16)
        zeroed_float16_array = np.zeros(Data.NUM_SCALAR_FLOAT16_ELEMENTS + info_length, dtype=np.float16)
        result = cls(zeroed_int16_array, zeroed_float16_array)
        # ref count and alt count remain zero
        result.set(Data.HAPLOTYPES_LENGTH, haplotypes_length)
        result.set(Data.INFO_LENGTH, info_length)
        result.set(Data.LABEL, label)
        result.set(Data.VARIANT_TYPE, variant_type)
        result.set(Data.SOURCE, source)
        result.set(Data.ORIGINAL_DEPTH, original_depth)
        result.set(Data.ORIGINAL_ALT_COUNT, original_alt_count)
        result.set(Data.ORIGINAL_NORMAL_DEPTH, original_normal_depth)
        result.set(Data.ORIGINAL_NORMAL_ALT_COUNT, original_normal_alt_count)
        result.set(Data.CONTIG, contig)
        result.set(Data.POSITION, position)
        result.set(Data.REF_ALLELE_AS_BASE_5, bases_as_base5_int(ref_allele))
        result.set(Data.ALT_ALLELE_AS_BASE_5, bases_as_base5_int(alt_allele))
        result.int16_array[Data.HAPLOTYPES_START_IDX:] = haplotypes

        result.set(Data.SEQ_ERROR_LOG_LK, seq_error_log_lk)    # this is -log10ToLog(TLOD) - log(tumorDepth + 1)
        result.set(Data.NORMAL_SEQ_ERROR_LOG_LK, normal_seq_error_log_lk)       # this is -log10ToLog(NALOD) - log(normalDepth + 1)
        result.float16_array[Data.INFO_START_IDX:] = info_array

        return result

    def get(self, data_field: Data):
        index = data_field.idx
        if data_field.dtype == np.uint16:
            return self.int16_array[index]
        elif data_field.dtype == np.uint32:
            return uint32_from_two_int16s(self.int16_array[index], self.int16_array[index + 1])
        elif data_field.dtype == np.float16:
            return self.float16_array[index]
        else:
            assert False, "Unsupported data type"

    def set(self, data_field: Data, value):
        index = data_field.idx
        if data_field.dtype == np.uint16:
            self.int16_array[index] = value
        elif data_field.dtype == np.uint32:
            int16_1, int16_2 = uint32_to_two_int16s(value)
            self.int16_array[index] = int16_1
            self.int16_array[index + 1] = int16_2
        elif data_field.dtype == np.float16:
            self.float16_array[index] = value
        else:
            assert False, "Unsupported data type"

    def get_read_count(self) -> int:
        return self.get(Data.ALT_COUNT) + self.get(Data.REF_COUNT)

    def is_labeled(self):
        return self.get(Data.LABEL) != Label.UNLABELED

    def get_ref_allele(self) -> str:
        return bases5_as_base_string(self.get(Data.REF_ALLELE_AS_BASE_5))

    def get_alt_allele(self) -> str:
        return bases5_as_base_string(self.get(Data.ALT_ALLELE_AS_BASE_5))

    def get_haplotypes_1d(self) -> np.ndarray:
        # 1D array of integer array reference and alt haplotypes concatenated -- A, C, G, T, deletion = 0, 1, 2, 3, 4
        assert len(self.int16_array) > Data.NUM_SCALAR_INT16_ELEMENTS, "trying to get ref seq array when none exists"
        return self.int16_array[Data.HAPLOTYPES_START_IDX:]

    def get_info_1d(self) -> np.ndarray:
        assert len(self.float16_array) > Data.NUM_SCALAR_FLOAT16_ELEMENTS, "trying to get info array when none exists"
        return self.float16_array[Data.INFO_START_IDX:]

    # note: this potentially resizes the array and requires the leading info tensor size element to be modified
    # we do this in preprocessing when adding extra info to the info from GATK.
    # this method should not otherwise be used!!!
    def set_info_1d(self, new_info: np.ndarray):
        self.float16_array = np.hstack((self.float16_array[:Data.INFO_START_IDX], new_info))
        self.set(Data.INFO_LENGTH, len(new_info))

    def get_int16_array(self) -> np.ndarray:
        return self.int16_array

    def get_float16_array(self) -> np.ndarray:
        return self.float16_array

    def get_nbytes(self) -> int:
        return self.int16_array.nbytes + self.float16_array.nbytes