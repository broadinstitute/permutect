from __future__ import annotations

import enum
from tkinter.tix import INTEGER

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
INTEGER_DTYPE = np.int16
LARGE_INTEGER_DTYPE = np.uint32
FLOAT_DTYPE = np.float16
BIGGEST_UINT16 = 65535
BIGGEST_INT16 = 32767

DEFAULT_GPU_FLOAT = torch.float32
DEFAULT_CPU_FLOAT = torch.float32

def uint32_to_two_int16s(num: int):
    uint16_1, uint16_2 = num // BIGGEST_UINT16, num % BIGGEST_UINT16
    return uint16_1 - (BIGGEST_INT16 + 1), uint16_2 - (BIGGEST_INT16 + 1)


def uint32_from_two_int16s(int16_1, int16_2):
    shifted1, shifted2 = int16_1 + (BIGGEST_INT16 + 1), int16_2 + (BIGGEST_INT16 + 1)
    return BIGGEST_UINT16 * shifted1 + shifted2

class Data(enum.Enum):
    # int array elements
    REF_COUNT = (INTEGER_DTYPE, 0)
    ALT_COUNT = (INTEGER_DTYPE, 1)
    LABEL = (INTEGER_DTYPE, 2)
    VARIANT_TYPE = (INTEGER_DTYPE, 3)
    SOURCE = (INTEGER_DTYPE, 4)
    ORIGINAL_DEPTH = (INTEGER_DTYPE, 5)
    ORIGINAL_ALT_COUNT = (INTEGER_DTYPE, 6)
    ORIGINAL_NORMAL_DEPTH = (INTEGER_DTYPE, 7)
    ORIGINAL_NORMAL_ALT_COUNT = (INTEGER_DTYPE, 8)
    CONTIG = (INTEGER_DTYPE, 9)
    POSITION = (LARGE_INTEGER_DTYPE, 10)              # NOTE: uint32 takes TWO uint16s!
    REF_ALLELE_AS_BASE_5 = (LARGE_INTEGER_DTYPE, 12)  # NOTE: uint32 takes TWO uint16s!
    ALT_ALLELE_AS_BASE_5 = (LARGE_INTEGER_DTYPE, 14)  # NOTE: uint32 takes TWO uint16s!
    # after this, at the end of the int16 array comes the sub-array containing the ref sequence haplotype

    # float array elements
    # allele frequency, maf, normal maf, and cached artifact logit are not used until the posterior model
    # are are set to
    SEQ_ERROR_LOG_LK = (FLOAT_DTYPE, 0)
    NORMAL_SEQ_ERROR_LOG_LK = (FLOAT_DTYPE, 1)
    ALLELE_FREQUENCY = (FLOAT_DTYPE, 2)
    MAF = (FLOAT_DTYPE, 3)
    NORMAL_MAF = (FLOAT_DTYPE, 4)
    CACHED_ARTIFACT_LOGIT = (FLOAT_DTYPE, 5)
    # TODO: left off here -- I added these constants to the enum but haven't yet initialized them in the constructor
    # TODO: nor have I cleaned up how they interact with the posterior datum
    # after this, at the end of the float16 array comes the sub-array containing the INFO vector

    def __init__(self, dtype: np.dtype, idx: int):
        self.dtype = dtype
        self.idx = idx

Data.NUM_SCALAR_INT_ELEMENTS = 16    # in Python 3.11+ can use enum.nonmember
Data.HAPLOTYPES_START_IDX = 16          # in Python 3.11+ can use enum.nonmember
Data.NUM_SCALAR_FLOAT_ELEMENTS = 6    # in Python 3.11+ can use enum.nonmember
Data.INFO_START_IDX = 6          # in Python 3.11+ can use enum.nonmember

class Datum:
    """
    contains data that apply to a candidate mutation as a whole i.e. not the read sets.  These are organized into a single
    LongTensor, containing some quantities that are inherently integral and some that are cast as longs by multiplying
    with a large number and rounding.
    """
    
    def __init__(self, int_array: np.ndarray, float_array: np.ndarray):
        # note: this constructor does no checking eg of whether the arrays are consistent with their purported lengths
        # or of whether ref, alt alleles have been trimmed
        assert int_array.ndim == 1 and len(int_array) >= Data.NUM_SCALAR_INT_ELEMENTS
        self.int_array: np.ndarray = np.ndarray.astype(int_array, np.int16)
        assert float_array.ndim == 1 and len(float_array) >= Data.NUM_SCALAR_FLOAT_ELEMENTS
        self.float_array: np.ndarray = np.ndarray.astype(float_array, FLOAT_DTYPE)

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
        zeroed_int_array = np.zeros(Data.NUM_SCALAR_INT_ELEMENTS + haplotypes_length, dtype=INTEGER_DTYPE)
        zeroed_float_array = np.zeros(Data.NUM_SCALAR_FLOAT_ELEMENTS + info_length, dtype=FLOAT_DTYPE)
        result = cls(zeroed_int_array, zeroed_float_array)
        # ref count and alt count remain zero
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
        result.int_array[Data.HAPLOTYPES_START_IDX:] = haplotypes

        result.set(Data.SEQ_ERROR_LOG_LK, seq_error_log_lk)    # this is -log10ToLog(TLOD) - log(tumorDepth + 1)
        result.set(Data.NORMAL_SEQ_ERROR_LOG_LK, normal_seq_error_log_lk)       # this is -log10ToLog(NALOD) - log(normalDepth + 1)
        
        result.set(Data.ALLELE_FREQUENCY, np.nan)
        result.set(Data.MAF, np.nan)
        result.set(Data.NORMAL_MAF, np.nan)
        result.set(Data.CACHED_ARTIFACT_LOGIT, np.nan)

        result.float_array[Data.INFO_START_IDX:] = info_array
        return result

    def get(self, data_field: Data):
        index = data_field.idx
        if data_field.dtype == INTEGER_DTYPE:
            return self.int_array[index]
        elif data_field.dtype == LARGE_INTEGER_DTYPE:
            return uint32_from_two_int16s(self.int_array[index], self.int_array[index + 1])
        elif data_field.dtype == FLOAT_DTYPE:
            return self.float_array[index]
        else:
            assert False, "Unsupported data type"

    def set(self, data_field: Data, value):
        index = data_field.idx
        if data_field.dtype == INTEGER_DTYPE:
            self.int_array[index] = value
        elif data_field.dtype == LARGE_INTEGER_DTYPE:
            int_1, int_2 = uint32_to_two_int16s(value)
            self.int_array[index] = int_1
            self.int_array[index + 1] = int_2
        elif data_field.dtype == FLOAT_DTYPE:
            self.float_array[index] = value
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
        assert len(self.int_array) > Data.NUM_SCALAR_INT_ELEMENTS, "trying to get ref seq array when none exists"
        return self.int_array[Data.HAPLOTYPES_START_IDX:]

    def get_info_1d(self) -> np.ndarray:
        assert len(self.float_array) > Data.NUM_SCALAR_FLOAT_ELEMENTS, "trying to get info array when none exists"
        return self.float_array[Data.INFO_START_IDX:]

    # note: this potentially resizes the array
    # we do this in preprocessing when adding extra info to the info from GATK.
    # this method should not otherwise be used!!!
    def set_info_1d(self, new_info: np.ndarray):
        self.float_array = np.hstack((self.float_array[:Data.INFO_START_IDX], new_info))

    def set_haplotypes_1d(self, new_haplotypes: np.ndarray):
        self.int_array = np.hstack((self.int_array[:Data.HAPLOTYPES_START_IDX], new_haplotypes))

    def get_int_array(self) -> np.ndarray:
        return self.int_array

    def get_float_array(self) -> np.ndarray:
        return self.float_array

    def get_nbytes(self) -> int:
        return self.int_array.nbytes + self.float_array.nbytes