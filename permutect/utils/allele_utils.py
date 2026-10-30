from __future__ import annotations

import cyvcf2
import numpy as np

from permutect.utils.enums import Variation
from permutect.utils.math_utils import find_factors


def trim_alleles_on_right(ref: str, alt: str):
    # if alt and ref alleles are not in minimal representation ie have redundant matching bases at the end, trim them
    trimmed_ref, trimmed_alt = ref, alt
    while len(trimmed_ref) > 1 and len(trimmed_alt) > 1 and trimmed_alt[-1] == trimmed_ref[-1]:
        trimmed_ref, trimmed_alt = trimmed_ref[:-1], trimmed_alt[:-1]
    return trimmed_ref, trimmed_alt


def find_variant_type(v: cyvcf2.Variant):
    alt = v.ALT[0]  # TODO: we're assuming biallelic
    ref = v.REF
    return Variation.get_type(ref, alt)


def count_leading_repeats(sequence: str, unit: str):
    # count how many times a unit string is repeated at the beginning of a larger string
    # eg 'ATATGGG', 'AT' -> 2; 'AGGGGG', 'G' -> 0; 'TTATTATTAGTTA', 'TTA' -> 3
    result = 0
    while sequence.startswith(unit, result * len(unit)):
        result += 1
    return result


def count_trailing_repeats(sequence: str, unit: str):
    # same, but at the end of a sequence
    # eg 'ATATGGG', 'G' -> 3; 'AGGGGG', 'G' -> 5; 'TTATTATTAGTTA', 'TTA' -> 1
    return count_leading_repeats(sequence[::-1], unit[::-1])


MAX_NUM_BASES_FOR_ENCODING = 13

# A, C, G and T as 0, 1, 2 and 3.  Anything else, such as N, is treated as T.
BASE_INDEX = {"A": 0, "C": 1, "G": 2}


def truncate_bases_if_necessary(bases: str):
    return bases if len(bases) <= MAX_NUM_BASES_FOR_ENCODING else bases[:MAX_NUM_BASES_FOR_ENCODING]


def bases_as_base5_int(bases: str) -> int:
    # here we just butcher variants longer than 13 bases and chop!!!
    return sum((BASE_INDEX.get(nuc, 3) + 1) * 5**n for n, nuc in enumerate(truncate_bases_if_necessary(bases)))


def bases5_as_base_string(base5: int) -> str:
    result = ""
    while base5 > 0:
        base5, digit = divmod(base5, 5)
        result += "TACGT"[digit]  # the digits 1 to 4 are A, C, G and T
    return result


def is_repeat(bases: str, unit: str):
    # eg ACGACGACG, ACG -> True; TTATTA, TA -> False
    return len(bases) % len(unit) == 0 and unit * (len(bases) // len(unit)) == bases


def decompose_str_unit(indel_bases: str):
    # decompose an indel into its most basic repeated unit
    # examples: "ATATAT" -> ("AT", 3); "AAAAA" -> ("A", 5); "TTGTTG" -> ("TTG", 2); "ATGTG" -> "ATGTG", 1
    for unit_length in find_factors(len(indel_bases)):  # note: these are sorted ascending
        unit = indel_bases[:unit_length]
        if is_repeat(indel_bases, unit):
            return unit, (len(indel_bases) // unit_length)
    return indel_bases, 1


def get_str_info_array(ref_sequence_string: str, ref_allele: str, alt_allele: str):
    assert len(ref_sequence_string) % 2 == 1, "must be odd length to have well-defined middle"
    middle_idx = (len(ref_sequence_string) - 1) // 2

    ref, alt = ref_allele, alt_allele

    insertion_length = max(len(alt) - len(ref), 0)
    deletion_length = max(len(ref) - len(alt), 0)

    if len(ref) == len(alt):
        unit, num_units, anchor = alt, 1, 0
    else:
        # the inserted or deleted sequence is everything after the anchor base, which repeats before the variant include
        unit, num_units = decompose_str_unit((alt if insertion_length > 0 else ref)[1:])
        anchor = 1
    # repeats after the substituted or anchor base.  For a deletion it's pretty arbitrary that the deleted bases
    # themselves count as after.
    repeats_after = count_leading_repeats(ref_sequence_string[middle_idx + min(len(ref), len(alt)) :], unit)
    repeats_before = count_trailing_repeats(ref_sequence_string[: middle_idx + anchor], unit)
    # note that if indels are left-aligned (as they should be from the GATK) repeats_before really ought to be zero!!
    return np.array([insertion_length, deletion_length, len(unit), num_units, repeats_before, repeats_after])


def make_1d_sequence_tensor(sequence_string: str) -> np.ndarray:
    """
    convert string of form ACCGTA into tensor [ 0, 1, 1, 2, 3, 0]
    """
    return np.array([BASE_INDEX.get(char, 3) for char in sequence_string], dtype=np.uint8)


# returns two length-L 1D arrays of ref stacked on top of alt, with '4' in alt(ref) for deletions(insertions)
def get_ref_and_alt_sequences(ref_seq_1d, ref_allele: str, alt_allele: str):
    """
    :param ref_seq_1d: 1D numpy integer array in form eg ATTTCGG -> [0,3,3,3,1,2,2]
    :return:
    """
    assert len(ref_seq_1d) % 2 == 1, "ref sequence length should be odd"
    middle_idx = (len(ref_seq_1d) - 1) // 2
    max_allele_length = middle_idx  # just kind of a coincidence
    ref, alt = ref_allele[:max_allele_length], alt_allele[:max_allele_length]

    if len(ref) >= len(alt):  # substitution or deletion
        ref_array = ref_seq_1d
        alt_array = np.copy(ref_array)
        deletion_length = len(ref) - len(alt)
        # add the deletion value '4' to make the alt allele array as long as the ref allele
        alt_allele_array = np.hstack((make_1d_sequence_tensor(alt), np.full(shape=deletion_length, fill_value=4)))
        alt_array[middle_idx : middle_idx + len(alt_allele_array)] = alt_allele_array
    else:  # insertion
        insertion_length = len(alt) - len(ref)
        before = ref_seq_1d[:middle_idx]
        after = ref_seq_1d[middle_idx + len(ref) : -insertion_length]

        alt_allele_array = make_1d_sequence_tensor(alt)
        ref_allele_array = np.hstack((make_1d_sequence_tensor(ref), np.full(shape=insertion_length, fill_value=4)))

        ref_array = np.hstack((before, ref_allele_array, after))
        alt_array = np.hstack((before, alt_allele_array, after))

    assert len(ref_array) == len(alt_array)
    if len(ref) == len(alt):  # SNV -- ref and alt ought to be different
        assert alt_array[middle_idx] != ref_array[middle_idx]
    else:  # indel -- ref and alt are the same at the anchor base, then are different
        assert alt_array[middle_idx + 1] != ref_array[middle_idx + 1]
    return ref_array[: len(ref_seq_1d)], alt_array[: len(ref_seq_1d)]  # this clipping may be redundant
