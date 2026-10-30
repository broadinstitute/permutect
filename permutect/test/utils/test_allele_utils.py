from types import SimpleNamespace

import numpy as np
import pytest

from permutect.utils.allele_utils import MAX_NUM_BASES_FOR_ENCODING
from permutect.utils.allele_utils import bases5_as_base_string
from permutect.utils.allele_utils import bases_as_base5_int
from permutect.utils.allele_utils import count_leading_repeats
from permutect.utils.allele_utils import count_trailing_repeats
from permutect.utils.allele_utils import decompose_str_unit
from permutect.utils.allele_utils import find_variant_type
from permutect.utils.allele_utils import get_ref_and_alt_sequences
from permutect.utils.allele_utils import get_str_info_array
from permutect.utils.allele_utils import is_repeat
from permutect.utils.allele_utils import make_1d_sequence_tensor
from permutect.utils.allele_utils import trim_alleles_on_right
from permutect.utils.allele_utils import truncate_bases_if_necessary
from permutect.utils.enums import Variation


def test_trim_alleles_on_right():
    assert trim_alleles_on_right("CGG", "TGG") == ("C", "T")
    assert trim_alleles_on_right("ACGT", "AT") == ("ACG", "A")
    assert trim_alleles_on_right("AT", "ACGT") == ("A", "ACG")

    # minimal alleles are unchanged
    assert trim_alleles_on_right("A", "T") == ("A", "T")
    assert trim_alleles_on_right("AC", "A") == ("AC", "A")

    # trimming stops at one base, so the anchor base of an indel is kept
    assert trim_alleles_on_right("CTT", "CT") == ("CT", "C")
    assert trim_alleles_on_right("TT", "TTT") == ("T", "TT")

    # matching bases on the left are not trimmed
    assert trim_alleles_on_right("GA", "GT") == ("GA", "GT")


def test_find_variant_type():
    # only the REF and ALT of the variant are used
    def variant(ref: str, *alts: str):
        return SimpleNamespace(REF=ref, ALT=list(alts))

    assert find_variant_type(variant("A", "T")) == Variation.SNV
    assert find_variant_type(variant("A", "AT")) == Variation.INSERTION
    assert find_variant_type(variant("AT", "A")) == Variation.DELETION
    assert find_variant_type(variant("A", "ATT")) == Variation.BIG_INSERTION
    assert find_variant_type(variant("ATT", "A")) == Variation.BIG_DELETION

    # only the first alt allele is used
    assert find_variant_type(variant("A", "AT", "T")) == Variation.INSERTION


def test_count_leading_repeats():
    assert count_leading_repeats("ATATGGG", "AT") == 2
    assert count_leading_repeats("AGGGGG", "G") == 0
    assert count_leading_repeats("TTATTATTAGTTA", "TTA") == 3

    # the whole sequence is repeats
    assert count_leading_repeats("ATATAT", "AT") == 3

    # a partial unit at the end doesn't count
    assert count_leading_repeats("ATATA", "AT") == 2

    assert count_leading_repeats("", "A") == 0
    assert count_leading_repeats("AT", "ATG") == 0


def test_count_trailing_repeats():
    assert count_trailing_repeats("ATATGGG", "G") == 3
    assert count_trailing_repeats("AGGGGG", "G") == 5
    assert count_trailing_repeats("TTATTATTAGTTA", "TTA") == 1
    assert count_trailing_repeats("GGATC", "TC") == 1

    # the whole sequence is repeats
    assert count_trailing_repeats("ATATAT", "AT") == 3

    # a partial unit at the start doesn't count
    assert count_trailing_repeats("TATAT", "AT") == 2

    assert count_trailing_repeats("", "A") == 0
    assert count_trailing_repeats("AT", "ATG") == 0


def test_truncate_bases_if_necessary():
    assert truncate_bases_if_necessary("") == ""
    assert truncate_bases_if_necessary("ACGT") == "ACGT"

    # 13 bases are kept whole, and longer ones are truncated to 13 bases
    assert MAX_NUM_BASES_FOR_ENCODING == 13
    assert truncate_bases_if_necessary("A" * 13) == "A" * 13
    assert truncate_bases_if_necessary("ACGT" * 4) == ("ACGT" * 4)[:13]


def test_bases_as_base5_int():
    # each base is a digit from 1 to 4, least significant first
    assert [bases_as_base5_int(base) for base in "ACGT"] == [1, 2, 3, 4]
    assert bases_as_base5_int("AC") == 1 + 2 * 5
    assert bases_as_base5_int("GTA") == 3 + 4 * 5 + 1 * 25
    assert bases_as_base5_int("") == 0

    # the longest encoding fits in the uint32 that a datum stores it in
    assert bases_as_base5_int("T" * 13) == 5**13 - 1 < 2**32

    # longer alleles are truncated to 13 bases
    assert bases_as_base5_int("A" * 13 + "C") == bases_as_base5_int("A" * 13)


def test_bases5_as_base_string():
    assert bases5_as_base_string(0) == ""
    assert bases5_as_base_string(1 + 2 * 5) == "AC"

    for bases in ["A", "C", "G", "T", "ACGT", "TTTT", "GATTACA", "T" * 13]:
        assert bases5_as_base_string(bases_as_base5_int(bases)) == bases

    # no digit is zero, so trailing As are not lost
    assert bases5_as_base_string(bases_as_base5_int("CAAA")) == "CAAA"


def test_is_repeat():
    assert is_repeat("ACGACGACG", "ACG")
    assert is_repeat("AAAA", "A")
    assert is_repeat("ACG", "ACG")

    assert not is_repeat("TTATTA", "TA")
    assert not is_repeat("ACGACT", "ACG")  # the last unit differs
    assert not is_repeat("ACGAC", "ACG")  # the length isn't a multiple of the unit length


def test_decompose_str_unit():
    assert decompose_str_unit("ATATAT") == ("AT", 3)
    assert decompose_str_unit("AAAAA") == ("A", 5)
    assert decompose_str_unit("TTGTTG") == ("TTG", 2)
    assert decompose_str_unit("ATGTG") == ("ATGTG", 1)
    assert decompose_str_unit("A") == ("A", 1)

    # the smallest repeated unit is used
    assert decompose_str_unit("ATATATAT") == ("AT", 4)
    assert decompose_str_unit("AA") == ("A", 2)


# In these reference contexts the ref allele starts at the middle base, as in GATK data.  The entries of the STR info
# array are insertion length, deletion length, unit length, number of units, repeats before and repeats after.
def test_get_str_info_array_snv():
    # CTAA G AAAC with G>A
    assert get_str_info_array("CTAAGAAAC", "G", "A").tolist() == [0, 0, 1, 1, 2, 3]


def test_get_str_info_array_insertion():
    # GGCGT T ACACG with an insertion of AC after the anchor T
    assert get_str_info_array("GGCGTTACACG", "T", "TAC").tolist() == [2, 0, 2, 1, 0, 2]

    # GGCGT T ACACG with an insertion of ACAC
    assert get_str_info_array("GGCGTTACACG", "T", "TACAC").tolist() == [4, 0, 2, 2, 0, 2]


def test_get_str_info_array_deletion():
    # ATTCT G CACAT with a deletion of CA after the anchor G.  The deleted bases count as repeats after.
    assert get_str_info_array("ATTCTGCACAT", "GCA", "G").tolist() == [0, 2, 2, 1, 0, 2]

    # the same context with a deletion of CACA
    assert get_str_info_array("ATTCTGCACAT", "GCACA", "G").tolist() == [0, 4, 2, 2, 0, 2]


def test_get_str_info_array_repeats_before_include_the_anchor_base():
    # an insertion of A that isn't left-aligned, anchored on the last of the three As before it
    assert get_str_info_array("GGCAAAAACTG", "A", "AA").tolist() == [1, 0, 1, 1, 3, 2]


def test_get_str_info_array_needs_odd_length():
    with pytest.raises(AssertionError):
        get_str_info_array("CTAAGAAA", "G", "A")


def test_make_1d_sequence_tensor():
    result = make_1d_sequence_tensor("ACCGTA")
    assert result.dtype == np.uint8
    assert result.tolist() == [0, 1, 1, 2, 3, 0]

    assert make_1d_sequence_tensor("").shape == (0,)


def ref_and_alt_sequences(ref_sequence: str, ref: str, alt: str) -> tuple[str, str]:
    # 4 is a gap: a deleted base in the alt sequence or an inserted base in the ref sequence
    ref_hap, alt_hap = get_ref_and_alt_sequences(make_1d_sequence_tensor(ref_sequence), ref, alt)
    return "".join("ACGT-"[i] for i in ref_hap), "".join("ACGT-"[i] for i in alt_hap)


def test_get_ref_and_alt_sequences():
    # substitutions replace bases in the alt sequence
    assert ref_and_alt_sequences("CTAAGAAAC", "G", "T") == ("CTAAGAAAC", "CTAATAAAC")
    assert ref_and_alt_sequences("CTAAGAAAC", "GA", "TC") == ("CTAAGAAAC", "CTAATCAAC")

    # deletions leave gaps in the alt sequence
    assert ref_and_alt_sequences("ATTCTGCACAT", "GCA", "G") == ("ATTCTGCACAT", "ATTCTG--CAT")

    # insertions leave gaps in the ref sequence and push the last bases out of both sequences
    assert ref_and_alt_sequences("GGCGTTACACG", "T", "TAC") == ("GGCGTT--ACA", "GGCGTTACACA")


def test_get_ref_and_alt_sequences_truncates_long_alleles():
    # alleles are truncated to the length of the flank after the middle base, here 5
    assert ref_and_alt_sequences("ATTCTGCACAT", "GCACAT", "G") == ("ATTCTGCACAT", "ATTCTG----T")
    assert ref_and_alt_sequences("GGCGTTACACG", "T", "TACGTAC") == ("GGCGTT----A", "GGCGTTACGTA")


def test_get_ref_and_alt_sequences_needs_odd_length():
    with pytest.raises(AssertionError):
        get_ref_and_alt_sequences(make_1d_sequence_tensor("CTAAGAAA"), "G", "T")
