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
    REF_COUNT = (np.uint16, 0)
    ALT_COUNT = (np.uint16, 1)
    HAPLOTYPES_LENGTH = (np.uint16, 2)
    INFO_LENGTH = (np.uint16, 3)
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

    # FloatTensor indices
    SEQ_ERROR_LOG_LK = (np.float16, 18)         # float stored as int=
    NORMAL_SEQ_ERROR_LOG_LK = (np.float16, 19)  # float stored as int=

    # after these come the variable-length sub-arrays (not within a single dataset, but in principle variable length fo
    # different versions of Permutect or different sequencing) for the reference sequence context and the info tensor

    def __init__(self, dtype: np.dtype, idx: int):
        self.dtype = dtype
        self.idx = idx

Data.NUM_SCALAR_ELEMENTS = 20    # in Python 3.11+ can use enum.nonmember
Data.HAPLOTYPES_START_IDX = 20   # in Python 3.11+ can use enum.nonmember

class Datum:
    """
    contains data that apply to a candidate mutation as a whole i.e. not the read sets.  These are organized into a single
    LongTensor, containing some quantities that are inherently integral and some that are cast as longs by multiplying
    with a large number and rounding.
    """
    
    def __init__(self, array: np.ndarray):
        # note: this constructor does no checking eg of whether the arrays are consistent with their purported lengths
        # or of whether ref, alt alleles have been trimmed
        assert array.ndim == 1 and len(array) >= Data.NUM_SCALAR_ELEMENTS
        self.array: np.ndarray = np.ndarray.astype(array, DATUM_ARRAY_DTYPE)

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
        result = cls(np.zeros(Data.NUM_SCALAR_ELEMENTS + haplotypes_length + info_length, dtype=DATUM_ARRAY_DTYPE))
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
        result.set(Data.SEQ_ERROR_LOG_LK, seq_error_log_lk)    # this is -log10ToLog(TLOD) - log(tumorDepth + 1)
        result.set(Data.NORMAL_SEQ_ERROR_LOG_LK, normal_seq_error_log_lk)       # this is -log10ToLog(NALOD) - log(normalDepth + 1)

        haplotypes_start = Data.HAPLOTYPES_START_IDX
        haplotypes_end = haplotypes_start + haplotypes_length
        info_end = haplotypes_end + info_length
        result.array[haplotypes_start:haplotypes_end] = haplotypes  # haplotypes array is uint8
        result.array[haplotypes_end:info_end] = np.ndarray.astype(info_array * FLOAT_TO_LONG_MULTIPLIER, DATUM_ARRAY_DTYPE)

        return result

    def get(self, data_field: Data):
        index = data_field.idx
        if data_field.dtype == np.uint16:
            return self.array[index]
        elif data_field.dtype == np.uint32:
            return uint32_from_two_int16s(self.array[index], self.array[index + 1])
        elif data_field.dtype == np.float16:
            return int16_to_float(self.array[index])
        else:
            assert False, "Unsupported data type"

    def set(self, data_field: Data, value):
        index = data_field.idx
        if data_field.dtype == np.uint16:
            self.array[index] = value
        elif data_field.dtype == np.uint32:
            int16_1, int16_2 = uint32_to_two_int16s(value)
            self.array[index] = int16_1
            self.array[index + 1] = int16_2
        elif data_field.dtype == np.float16:
            self.array[index] = float_to_clipped_int16(value)
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
        start = Data.HAPLOTYPES_START_IDX
        haplotypes_length = self.get(Data.HAPLOTYPES_LENGTH)
        assert haplotypes_length > 0, "trying to get ref seq array when none exists"
        return self.array[start:start + haplotypes_length]

    def get_info_1d(self) -> np.ndarray:
        start = Data.HAPLOTYPES_START_IDX + self.get(Data.HAPLOTYPES_LENGTH)
        info_length = self.get(Data.INFO_LENGTH)
        assert info_length > 0, "trying to get info array when none exists"
        return self.array[start:start + info_length] / FLOAT_TO_LONG_MULTIPLIER

    # note: this potentially resizes the array and requires the leading info tensor size element to be modified
    # we do this in preprocessing when adding extra info to the info from GATK.
    # this method should not otherwise be used!!!
    def set_info_1d(self, new_info: np.ndarray):
        new_info_as_long = np.ndarray.astype(new_info * FLOAT_TO_LONG_MULTIPLIER, DATUM_ARRAY_DTYPE)
        old_info_start = Data.HAPLOTYPES_START_IDX + self.get(Data.HAPLOTYPES_LENGTH)
        self.array = np.hstack((self.array[:old_info_start], new_info_as_long))
        self.set(Data.INFO_LENGTH, len(new_info))

    def get_array_1d(self) -> np.ndarray:
        return self.array

    def get_nbytes(self) -> int:
        return self.array.nbytes


DEFAULT_NUMPY_FLOAT = np.float16
DEFAULT_GPU_FLOAT = torch.float32
DEFAULT_CPU_FLOAT = torch.float32
MAX_FLOAT_16 = torch.finfo(torch.float16).max
MIN_FLOAT_16 = torch.finfo(torch.float16).min