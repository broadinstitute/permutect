"""
Functions for reading from plain text dataset files of the format

UNLABELED                                       # label
1:13118,A->G                                    # locus and mutation
GAGGAAAGTGAGGTTGCCTGC                           # reference context
0.12 0.09 2.00 0.57 1.00 1.00 1.00 1.00 1.00    # variant-level info vector
5 4 0 0                                         # ref count, alt count, matched normal ref count, matched normal alt count
27 30 0 1 11 29 333 321 12 0 0                  # one ref read vector per line
50 22 1 0 30 19 342 70 272 0 0
27 31 0 1 32 17 236 203 33 0 0
27 20 0 1 32 17 141 72 69 1 0
21 28 1 0 49 0 232 49 183 1 0
23 29 1 1 40 9 335 294 41 0 0                   # one alt read vector per line
24 29 0 1 38 11 354 315 39 0 0
24 30 0 1 36 13 351 314 37 0 0
23 30 1 1 42 7 341 298 43 0 0
51 13 0 0                                       # original ref, alt, normal ref, normal alt counts before downsampling
-108.131                                        # sequencing error log likelihood
-0.000                                          # matched normal sequencing error log likelihood
VARIANT
1:13302,C->T
GTCCTGGACACGCTGTTGGCC
0.00 0.00 0.00 1.00 1.00 2.00 1.00 1.00 1.00
2 1 0 0
24 29 0 0 11 21 338 11 327 0 0
50 25 0 1 49 -8 355 303 52 0 0
23 33 1 0 13 21 312 87 225 0 0
69 4 0 0
-11.327
-0.000
"""

from __future__ import annotations

import itertools
import math
from typing import Generator
from typing import List

import numpy as np
import torch
from sklearn.preprocessing import QuantileTransformer
from tqdm import tqdm

from permutect.data.count_binning import cap_alt_count
from permutect.data.count_binning import cap_ref_count
from permutect.data.datum import FLOAT_DTYPE
from permutect.data.datum import NUMBER_OF_BYTES_IN_PACKED_READ
from permutect.data.datum import Data
from permutect.data.datum import Datum
from permutect.data.datum import get_data_field
from permutect.data.memory_mapped_data import MemoryMappedData
from permutect.data.memory_mapped_data import StackedData
from permutect.misc_utils import ConsistentValue
from permutect.utils.enums import Label
from permutect.utils.enums import Variation
from permutect.utils.math_utils import log_binomial_coefficient

MAX_VALUE = 10000
EPSILON = 0.00001
QUANTILE_DATA_COUNT = 10000
LOG10_TO_LN = 2.30258509299

NUM_RAW_DATA_TO_NORMALIZE_AT_ONCE = 10000

DISTANCE_FROM_END_SATURATION = 20


def binarize_str_length_column(str_length_column):
    result = np.full((len(str_length_column), 6), 0, dtype=str_length_column.dtype)
    result[:, 0] = str_length_column == 1
    result[:, 1] = (str_length_column > 1) & (str_length_column < 4)
    result[:, 2] = (str_length_column > 3) & (str_length_column < 6)
    result[:, 3] = (str_length_column > 5) & (str_length_column < 9)
    result[:, 4] = (str_length_column > 8) & (str_length_column < 13)
    result[:, 5] = str_length_column > 12
    return result


def count_number_of_data_and_reads_in_text_file(dataset_file):
    num_data, num_reads = 0, 0
    with open(dataset_file) as file:
        while label_line := file.readline():  # the label line is the first line of each datum
            next(file)  # skip the contig:position,ref->alt line
            next(file)  # skip the reference sequence line
            next(file)  # skip the info array line

            # get the read tensor sizes
            ref_tensor_size, alt_tensor_size, normal_ref_tensor_size, normal_alt_tensor_size = map(
                int, file.readline().strip().split()
            )

            # skip the read tensors except for getting the array size from the very first read
            for idx in range(ref_tensor_size + alt_tensor_size + normal_ref_tensor_size + normal_alt_tensor_size):
                next(file)

            next(file)  # skip the original depths line
            next(file)  # skip the seq error log likelihood line
            next(file)  # skip the normal seq error log likelihood line
            if alt_tensor_size > 0:  # data with no alts is skipped
                num_data += 1
                num_reads += ref_tensor_size + alt_tensor_size  # we don't use normal reads

    return num_data, num_reads


def read_raw_unnormalized_data(
    dataset_file, only_artifacts: bool = False, source: int = 0
) -> Generator[Datum, None, None]:
    """
    generator that yields data from a plain text dataset file.
    """
    with open(dataset_file) as file:
        while label_str := file.readline().strip():
            label = Label[label_str]
            passes_label_filter = label == Label.ARTIFACT or not only_artifacts

            # contig:position,ref->alt
            variant_line = file.readline().strip()
            locus, mutation = variant_line.split(",")
            contig, position = map(int, locus.split(":"))  # contig is an integer *index* from a sequence dictionary
            ref_allele, alt_allele = mutation.strip().split("->")

            ref_sequence_string = file.readline().strip()
            gatk_info_array = line_to_array(file.readline())
            ref_tensor_size, alt_tensor_size, normal_ref_tensor_size, normal_alt_tensor_size = map(
                int, file.readline().strip().split()
            )

            # the first column is read group index, which we currently discard
            # later we're going to want to use this
            ref_tensor = read_2d_array(file, ref_tensor_size)[:, 1:] if ref_tensor_size > 0 else None
            alt_tensor = read_2d_array(file, alt_tensor_size)[:, 1:] if alt_tensor_size > 0 else None

            # consume normal reads from the file even though we don't use them;
            # in the future if the model changes to use normal reads, not just
            # read counts, here is where we would read them with
            # normal_ref_tensor = read_2d_tensor(file, normal_ref_tensor_size) etc.
            for _ in range(normal_ref_tensor_size + normal_alt_tensor_size):
                next(file)

            original_depth, original_alt_count, original_normal_depth, original_normal_alt_count = read_integers(
                file.readline()
            )
            # this is -log10ToLog(tlod) - log(tumorDepth + 1);
            seq_error_log_lk = read_float(file.readline())
            # this is -log10ToLog(nalod) - log(normalDepth + 1)
            normal_seq_error_log_lk = read_float(file.readline())

            if alt_tensor_size > 0 and passes_label_filter:
                datum = Datum.from_gatk(
                    label=label,
                    variant_type=Variation.get_type(ref_allele, alt_allele),
                    source=source,
                    original_depth=original_depth,
                    original_alt_count=original_alt_count,
                    original_normal_depth=original_normal_depth,
                    original_normal_alt_count=original_normal_alt_count,
                    contig=contig,
                    position=position,
                    ref_allele=ref_allele,
                    alt_allele=alt_allele,
                    seq_error_log_lk=seq_error_log_lk,
                    normal_seq_error_log_lk=normal_seq_error_log_lk,
                    ref_sequence_string=ref_sequence_string,
                    gatk_info_array=gatk_info_array,
                    ref_reads_array_re=ref_tensor,
                    alt_reads_array_re=alt_tensor,
                )

                ref_count = cap_ref_count(datum.get(Data.REF_COUNT))
                alt_count = cap_alt_count(datum.get(Data.ALT_COUNT))
                yield datum.copy_with_downsampled_reads(ref_count, alt_count)


def generate_raw_data_from_text_files(dataset_files, sources: List[int] = None) -> Generator[Datum, None, None]:
    data_dim, reads_dim = ConsistentValue(), ConsistentValue()

    for n, dataset_file in enumerate(dataset_files):
        source = 0 if sources is None else (sources[0] if len(sources) == 1 else sources[n])
        reads_datum: Datum
        for reads_datum in read_raw_unnormalized_data(dataset_file, source=source):
            data_dim.check(len(reads_datum.int_array))
            reads_dim.check(reads_datum.reads_re.shape[-1])
            yield reads_datum


def normalized_data_generator(raw_data, num_data: int) -> Generator[StackedData, None, None]:
    """
    The normalized data of each chunk of num_data raw data from an iterator.  The raw data of each chunk are read and
    normalized together, without writing them to disk.
    """
    num_chunks = math.ceil(num_data / NUM_RAW_DATA_TO_NORMALIZE_AT_ONCE)
    data_per_chunk = num_data // num_chunks

    for chunk in range(num_chunks):
        chunk_size = (num_data - chunk * data_per_chunk) if (chunk == num_chunks - 1) else data_per_chunk
        raw = StackedData.of(list(itertools.islice(raw_data, chunk_size)))
        yield normalize_raw_data(raw, make_read_quantile_transform(raw))


def make_read_quantile_transform(raw: StackedData):
    # fit to the fragment lengths of the ref reads of every datum, which come first among its reads
    ref_counts, read_starts = get_data_field(raw.int_array_ve, None, Data.REF_COUNT), raw.read_offsets[:-1]
    ref_reads_re = np.vstack([raw.reads_re[start : start + count] for start, count in zip(read_starts, ref_counts)])
    return QuantileTransformer(n_quantiles=100, output_distribution="normal").fit(ref_reads_re[:, 6:7])


def make_normalized_mmap_data(dataset_files, sources: List[int] = None) -> MemoryMappedData:
    """
    given unnormalized plain text dataset files from Mutect2, normalize data and save as tarfile of memory mapped numpy arrays

    In addition to quantile-normalizing read tensors it also enlarges the info tensors
    :param dataset_files:
    :param sources if None, source is set to 0; if singleton list, all files are given that source; otherwise one source per file
    """
    num_data, num_reads = 0, 0
    for dataset_file in dataset_files:
        file_num_data, file_num_reads = count_number_of_data_and_reads_in_text_file(dataset_file)
        num_data += file_num_data
        num_reads += file_num_reads

    # the counts that divide the data into chunks and size the memory maps also give the progress bar its total
    raw_data = tqdm(generate_raw_data_from_text_files(dataset_files, sources), total=num_data, mininterval=60)
    normalized_chunks = normalized_data_generator(raw_data, num_data)
    return MemoryMappedData.from_chunks(normalized_chunks, estimated_num_data=num_data, estimated_num_reads=num_reads)


def alt_read_medians_and_means(float_columns_re, boolean_columns_re, read_offsets, ref_counts) -> tuple:
    """
    For each datum, the medians of the float columns and the means of the boolean columns over its alt reads, as
    np.median and np.mean give them.  Datum n's reads are rows read_offsets[n]:read_offsets[n + 1], ref reads first.
    """
    read_starts, ref_counts = read_offsets[:-1].astype(np.int64), np.asarray(ref_counts, dtype=np.int64)
    alt_counts = np.diff(read_offsets).astype(np.int64) - ref_counts
    assert np.all(alt_counts > 0), "every datum must have an alt read"

    # the rows of the alt reads, each datum's together, and where each datum's start among them
    segment_starts = np.cumsum(alt_counts) - alt_counts
    alt_rows = np.repeat(read_starts + ref_counts - segment_starts, alt_counts) + np.arange(np.sum(alt_counts))
    segment_ids = np.repeat(np.arange(len(alt_counts)), alt_counts)

    # each datum's middle value, or the mean of its two middle values, of each column sorted within each datum.  Adding
    # 0.0 turns -0.0 into 0.0, as np.median does by summing from 0.0.
    alt_floats_re = float_columns_re[alt_rows]
    lower_middles, upper_middles = segment_starts + (alt_counts - 1) // 2, segment_starts + alt_counts // 2
    medians = np.empty((len(alt_counts), alt_floats_re.shape[1]), dtype=alt_floats_re.dtype)
    for column in range(alt_floats_re.shape[1]):
        sorted_column = alt_floats_re[np.lexsort((alt_floats_re[:, column], segment_ids)), column]
        medians[:, column] = (sorted_column[lower_middles] + sorted_column[upper_middles]) / 2 + 0.0

    alt_booleans_re = boolean_columns_re[alt_rows].astype(np.float64)
    means = np.add.reduceat(alt_booleans_re, segment_starts, axis=0) / alt_counts[:, None]
    return medians, means


def normalize_raw_data(raw: StackedData, read_quantile_transform) -> StackedData:
    """
    Normalize a chunk of raw data, compressing their reads and replacing their info with new features
    """
    int_array_ve, float_array_ve = raw.int_array_ve, raw.float_array_ve

    # 2D array.  Rows are ref/alt reads, columns are read features
    all_reads_re = raw.reads_re

    # 2D array.  Rows are read sets, columns are info features
    all_info_ve = float_array_ve[:, Data.INFO_START_IDX :]

    # INFO features from GATK are
    # 0, 1 haplotype equivalence share of 2nd and 3rd most supported haplotype groups (among all haplotypes with
    # the alt allele)
    # 2 edit distance, excluding variant site, of best alt haplotype vs closest germline haplotype
    # 3 fraction of reads supporting alt allele that support the single best alt haplotype
    # 4 TLOD / alt count

    hap_equiv_columns = all_info_ve[:, 0:2]
    hap_equiv_binary_1 = hap_equiv_columns < 0.1
    hap_equiv_binary_2 = (hap_equiv_columns >= 0.1) & (hap_equiv_columns < 0.25)
    hap_equiv_binary_3 = hap_equiv_columns >= 0.25

    edit_dist_column = all_info_ve[:, 2]
    edit_dist_binary = np.column_stack([edit_dist_column == d for d in range(3)] + [edit_dist_column >= 3])

    hap_dom_column = all_info_ve[:, 3]
    hap_dom_binary = np.column_stack([hap_dom_column > 0.9, (hap_dom_column <= 0.9) & (hap_dom_column > 0.7)])

    # STR INFO features are
    # 5, 6 insertion_length, deletion_length of variant
    # 7 STR repeat unit length
    # 8 number of STR repeat units
    # 9 repeats before variant
    # 10 repeats after variant

    indel_lengths_ve = all_info_ve[:, 5:7] / 10

    str_unit_length = all_info_ve[:, 7]
    str_unit_length_binaries = np.column_stack([str_unit_length == n for n in range(1, 5)] + [str_unit_length >= 5])

    # total STR length, and STR lengths before and after the variant
    str_length_binaries = [
        binarize_str_length_column(str_unit_length * all_info_ve[:, column]) for column in (8, 9, 10)
    ]
    str_info_array_ve = np.hstack([indel_lengths_ve, str_unit_length_binaries, *str_length_binaries])

    binary_info_array_ve = np.hstack(
        (
            hap_equiv_binary_1,
            hap_equiv_binary_2,
            hap_equiv_binary_3,
            edit_dist_binary,
            hap_dom_binary,
            str_info_array_ve,
        )
    )

    tlod_over_nalt = all_info_ve[:, 4]
    orig_alt_counts = np.array(get_data_field(int_array_ve, None, Data.ORIGINAL_ALT_COUNT))
    orig_depths = np.array(get_data_field(int_array_ve, None, Data.ORIGINAL_DEPTH))

    natural_log_tlod = LOG10_TO_LN * tlod_over_nalt * orig_alt_counts
    tlod_correction = log_binomial_coefficient(torch.from_numpy(orig_depths), torch.from_numpy(orig_alt_counts)).numpy()
    average_qual_feature = ((natural_log_tlod + tlod_correction) / orig_alt_counts) / 10

    all_info_transformed_ve = np.hstack([binary_info_array_ve, average_qual_feature.reshape(-1, 1)])

    from_read_ends_columns_re = all_reads_re[:, 4:6]
    from_read_ends_transformed_re = np.tanh(from_read_ends_columns_re / DISTANCE_FROM_END_SATURATION)
    from_frag_ends_columns_re = all_reads_re[:, 7:9]
    from_frag_ends_transformed_re = np.tanh(from_frag_ends_columns_re / DISTANCE_FROM_END_SATURATION)
    distance_columns_re = all_reads_re[:, 6:7]
    distance_columns_transformed_re = read_quantile_transform.transform(distance_columns_re)
    float_read_columns_re = np.hstack(
        [
            from_read_ends_transformed_re,
            from_frag_ends_transformed_re,
            distance_columns_transformed_re,
        ]
    )

    # columns of raw read data are
    # 0 map qual -> 4 categorical columns
    # 1 base qual -> 4 categorical columns
    # 2,3 strand and orientation (binary) -> remain binary
    # 4,5 distance of variant from left and right end of read (including clipped bases)
    # 6 fragment length
    # 7,8 distance of variant from left and right end of fragment (including clipped bases)
    # TODO: combine these to get, on scale of -1 (extreme left of fragment) to +1 (extreme right of fragment),
    # TODO: the position of the variant within the fragment.
    # TODO: do the same for the read position.
    # 9 and higher -- SNV/indel error and low BQ counts
    # TODO: double-check in Mutect2 that these included soft-clipped bases

    map_qual_column = all_reads_re[:, 0]
    map_qual_boolean = np.full((len(all_reads_re), 4), True, dtype=bool)
    map_qual_boolean[:, 0] = map_qual_column > 59
    map_qual_boolean[:, 1] = (map_qual_column <= 59) & (map_qual_column >= 40)
    map_qual_boolean[:, 2] = (map_qual_column < 40) & (map_qual_column >= 20)
    map_qual_boolean[:, 3] = map_qual_column < 20

    base_qual_column = all_reads_re[:, 1]
    base_qual_boolean = np.full((len(all_reads_re), 4), True, dtype=bool)
    base_qual_boolean[:, 0] = base_qual_column >= 30
    base_qual_boolean[:, 1] = (base_qual_column < 30) & (base_qual_column >= 20)
    base_qual_boolean[:, 2] = (base_qual_column < 20) & (base_qual_column >= 10)
    base_qual_boolean[:, 3] = base_qual_column < 10

    strand_and_orientation_boolean = all_reads_re[:, 2:4] < 0.5
    error_counts_boolean_1 = all_reads_re[:, 9:] < 0.5
    error_counts_boolean_2 = (all_reads_re[:, 9:] > 0.5) & (all_reads_re[:, 9:] < 1.5)
    error_counts_boolean_3 = all_reads_re[:, 9:] > 1.5

    boolean_output_array_re = np.hstack(
        (
            map_qual_boolean,
            base_qual_boolean,
            strand_and_orientation_boolean,
            error_counts_boolean_1,
            error_counts_boolean_2,
            error_counts_boolean_3,
        )
    )

    # axis = 1 is essential so that each row (read) of the packed data corresponds to a row of the unpacked data
    packed_output_array = np.packbits(boolean_output_array_re, axis=1)

    assert packed_output_array.dtype == np.uint8
    assert packed_output_array.shape[1] == NUMBER_OF_BYTES_IN_PACKED_READ, (
        f"boolean array shape {boolean_output_array_re.shape}, packed shape {packed_output_array.shape}"
    )

    # the conversion to uint8 maps the range [-4,4] to [0,255].  Anything less than -4 is mapped to 0 and
    # anything greater than 4 is mapped to 255.
    distance_columns_output = convert_quantile_normalized_to_uint8(float_read_columns_re)
    assert packed_output_array.dtype == distance_columns_output.dtype
    output_uint8_reads_array = np.hstack((packed_output_array, distance_columns_output))

    # TODO: maybe we could also have columnwise nonparametric test statistics, like for example we record the
    # TODO: quantiles over all ref reads
    ref_counts = get_data_field(int_array_ve, None, Data.REF_COUNT)
    alt_distance_medians_ve, alt_boolean_means_ve = alt_read_medians_and_means(
        float_read_columns_re, boolean_output_array_re, raw.read_offsets, ref_counts
    )

    # the float arrays keep their scalar elements, and their info becomes the new features
    new_info_ve = np.hstack((all_info_transformed_ve, alt_distance_medians_ve, alt_boolean_means_ve))
    normalized_float_array_ve = np.hstack((float_array_ve[:, : Data.INFO_START_IDX], new_info_ve))
    # the compressed reads are the raw reads row for row, so they have the same offsets
    return StackedData(int_array_ve, normalized_float_array_ve, output_uint8_reads_array, raw.read_offsets)


def tokens_to_array(tokens) -> np.ndarray:
    """
    The numbers of a list of strings, or of a list of lists of strings with as many in each, clipped to
    [-MAX_VALUE, MAX_VALUE].  NaN becomes MAX_VALUE, as it did when each number was clipped with Python's min and max.
    """
    values = np.array(tokens, dtype=np.float64)
    return np.clip(np.nan_to_num(values, nan=MAX_VALUE), -MAX_VALUE, MAX_VALUE).astype(FLOAT_DTYPE)


def line_to_array(line: str) -> np.ndarray:
    return tokens_to_array(line.split())


def read_2d_array(file, num_lines: int) -> np.ndarray:
    if num_lines == 0:
        return None
    # numpy parses all the lines at once, rather than each number in Python
    return tokens_to_array([file.readline().split() for _ in range(num_lines)])


def read_integers(line: str):
    return map(int, line.strip().split())


def read_float(line: str):
    return float(line.strip().split()[0])


# quantile-normalized data is generally some small number of standard deviations from 0.  We can store as uint8 by
# mapping x --> 32x + 128 and restricting to the range [0,255], which maps -4 and +4 standard deviations to the limits
# of uint8
def convert_quantile_normalized_to_uint8(data: np.ndarray):
    return np.ndarray.astype(np.clip(data * 32 + 128, 0, 255), np.uint8)


def convert_uint8_to_quantile_normalized(data: np.ndarray):
    # cast before subtracting: uint8 - 128 stays uint8 and wraps around for every byte below 128
    return np.ndarray.astype((data.astype(np.float32) - 128) / 32, FLOAT_DTYPE)
