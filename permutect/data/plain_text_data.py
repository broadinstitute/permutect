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

import math
from queue import PriorityQueue
from typing import List, Generator

import numpy as np
from scipy.special import gammaln
from sklearn.preprocessing import QuantileTransformer

from permutect.data.count_binning import cap_ref_count, cap_alt_count
from permutect.data.memory_mapped_data import MemoryMappedData
from permutect.data.datum import FLOAT_DTYPE, Datum, Data, NUMBER_OF_BYTES_IN_PACKED_READ

from permutect.misc_utils import ConsistentValue
from permutect.utils.enums import Variation, Label

MAX_VALUE = 10000
EPSILON = 0.00001
QUANTILE_DATA_COUNT = 10000
LOG10_TO_LN = 2.30258509299

MIN_NUM_DATA_FOR_NORMALIZATION = 1000
MAX_NUM_DATA_FOR_NORMALIZATION = 100000
NUM_RAW_DATA_TO_NORMALIZE_AT_ONCE = 10000

DISTANCE_FROM_END_SATURATION = 20

def binarize_str_length_column(str_length_column):
    result = np.full((len(str_length_column), 6), 0, dtype=str_length_column.dtype)
    result[:, 0] = (str_length_column == 1)
    result[:, 1] = (str_length_column > 1) & (str_length_column < 4)
    result[:, 2] = (str_length_column > 3) & (str_length_column < 6)
    result[:, 3] = (str_length_column > 5) & (str_length_column < 9)
    result[:, 4] = (str_length_column > 8) & (str_length_column < 13)
    result[:, 5] = (str_length_column > 12)
    return result


def count_number_of_data_and_reads_in_text_file(dataset_file):
    num_data, num_reads = 0, 0
    with open(dataset_file) as file:
        while label_line := file.readline():    # the label line is the first line of each datum

            next(file)  # skip the contig:position,ref->alt line
            next(file)  # skip the reference sequence line
            next(file)  # skip the info array line

            # get the read tensor sizes
            ref_tensor_size, alt_tensor_size, normal_ref_tensor_size, normal_alt_tensor_size = map(int, file.readline().strip().split())

            # skip the read tensors except for getting the array size from the very first read
            for idx in range(ref_tensor_size + alt_tensor_size + normal_ref_tensor_size + normal_alt_tensor_size):
                next(file)

            next(file)  # skip the original depths line
            next(file)  # skip the seq error log likelihood line
            next(file)  # skip the normal seq error log likelihood line
            if alt_tensor_size > 0: # data with no alts is skipped
                num_data += 1
                num_reads += (ref_tensor_size + alt_tensor_size)    # we don't use normal reads

    return num_data, num_reads


def read_raw_unnormalized_data(dataset_file, only_artifacts: bool = False, source: int=0) -> Generator[Datum, None, None]:
    """
    generator that yields data from a plain text dataset file.
    """
    with open(dataset_file) as file:
        n = 0
        while label_str := file.readline().strip():
            label = Label.get_label(label_str)
            passes_label_filter = (label == Label.ARTIFACT or not only_artifacts)
            n += 1

            # contig:position,ref->alt
            variant_line = file.readline().strip()
            locus, mutation = variant_line.split(",")
            contig, position = map(int, locus.split(":"))   # contig is an integer *index* from a sequence dictionary
            # TODO: replace with tqdm progress bar by counting file in initial pass.  It can't be that expensive.
            if n % 100000 == 0:
                print(f"{contig}:{position}")
            ref_allele, alt_allele = mutation.strip().split("->")

            ref_sequence_string = file.readline().strip()
            gatk_info_array = line_to_array(file.readline())
            ref_tensor_size, alt_tensor_size, normal_ref_tensor_size, normal_alt_tensor_size = map(int, file.readline().strip().split())

            # the first column is read group index, which we currently discard
            # later we're going to want to use this
            ref_tensor = read_2d_array(file, ref_tensor_size)[:, 1:] if ref_tensor_size > 0 else None
            alt_tensor = read_2d_array(file, alt_tensor_size)[:, 1:] if alt_tensor_size > 0 else None

            # normal_ref_tensor = read_2d_tensor(file, normal_ref_tensor_size)  # not currently used
            # normal_alt_tensor = read_2d_tensor(file, normal_alt_tensor_size)  # not currently used
            # round down normal tensors as well

            original_depth, original_alt_count, original_normal_depth, original_normal_alt_count = read_integers(file.readline())
            # this is -log10ToLog(tlod) - log(tumorDepth + 1);
            seq_error_log_lk = read_float(file.readline())
            # this is -log10ToLog(nalod) - log(normalDepth + 1)
            normal_seq_error_log_lk = read_float(file.readline())

            if alt_tensor_size > 0 and passes_label_filter:
                datum = Datum.from_gatk(label=label, variant_type=Variation.get_type(ref_allele, alt_allele), source=source,
                                                            original_depth=original_depth, original_alt_count=original_alt_count,
                                                            original_normal_depth=original_normal_depth, original_normal_alt_count=original_normal_alt_count,
                                                            contig=contig, position=position, ref_allele=ref_allele, alt_allele=alt_allele,
                                                            seq_error_log_lk=seq_error_log_lk, normal_seq_error_log_lk=normal_seq_error_log_lk,
                                                            ref_sequence_string=ref_sequence_string, gatk_info_array=gatk_info_array,
                                                            ref_reads_array_re=ref_tensor, alt_reads_array_re=alt_tensor)

                ref_count = cap_ref_count(datum.get(Data.REF_COUNT))
                alt_count = cap_alt_count(datum.get(Data.ALT_COUNT))
                yield datum.copy_with_downsampled_reads(ref_count, alt_count)


def generate_raw_data_from_text_files(dataset_files, sources: List[int]=None) -> Generator[Datum, None, None]:
    data_dim, reads_dim = ConsistentValue(), ConsistentValue()

    for n, dataset_file in enumerate(dataset_files):
        source = 0 if sources is None else (sources[0] if len(sources) == 1 else sources[n])
        reads_datum: Datum
        for reads_datum in read_raw_unnormalized_data(dataset_file, source=source):
            data_dim.check(len(reads_datum.int_array))
            reads_dim.check(reads_datum.reads_re.shape[-1])
            yield reads_datum


def write_raw_unnormalized_data_to_memory_maps(dataset_files, sources: List[int]=None):
    total_num_data, total_num_reads = 0, 0

    for dataset_file in dataset_files:
        num_data, num_reads = count_number_of_data_and_reads_in_text_file(dataset_file)
        total_num_data += num_data
        total_num_reads += num_reads

    memory_mapped_data = MemoryMappedData.from_generator(reads_datum_source=generate_raw_data_from_text_files(dataset_files, sources),
                                                         estimated_num_data=num_data, estimated_num_reads=num_reads)
    return memory_mapped_data


def normalized_data_generator(raw_mmap_data: MemoryMappedData) -> Generator[Datum, None, None]:
    int_mmap_ve = raw_mmap_data.int_mmap
    float_mmap_ve = raw_mmap_data.float_mmap
    reads_mmap_re = raw_mmap_data.reads_mmap
    read_end_indices = raw_mmap_data.read_end_indices

    num_chunks = math.ceil(raw_mmap_data.num_data / NUM_RAW_DATA_TO_NORMALIZE_AT_ONCE)
    data_per_chunk = raw_mmap_data.num_data // num_chunks

    for chunk in range(num_chunks):
        start_idx = chunk * data_per_chunk
        end_idx = raw_mmap_data.num_data if (chunk == num_chunks - 1)  else (start_idx + data_per_chunk)

        read_start_idx = 0 if start_idx == 0 else read_end_indices[start_idx - 1]

        read_quantile_transform = make_read_quantile_transform(
            read_end_indices=(read_end_indices[start_idx:end_idx]-read_start_idx),
            int_array_ve=int_mmap_ve[start_idx:end_idx],
            float_array_ve=float_mmap_ve[start_idx:end_idx],
            reads_re=reads_mmap_re[read_start_idx:read_end_indices[end_idx - 1]],)

        raw_data_list = []
        for idx in range(start_idx, end_idx):
            reads = reads_mmap_re[0 if idx == 0 else read_end_indices[idx - 1]:read_end_indices[idx]]
            raw_datum = Datum(int_array=int_mmap_ve[idx], float_array=float_mmap_ve[idx], reads_re=reads)
            raw_data_list.append(raw_datum)

        normalized_data_list = normalize_raw_data_list(raw_data_list, read_quantile_transform)
        for datum in normalized_data_list:
            yield datum

def make_read_quantile_transform(read_end_indices, int_array_ve, float_array_ve, reads_re):
    read_end_indices = read_end_indices

    # indices_for_normalization = get_normalization_set(data_ve)
    # revert to old behavior: use all ref data for normalization
    indices_for_normalization = list(range(len(int_array_ve)))

    # define ref read ranges for each datum in the normalization set
    normalization_read_start_indices = [read_end_indices[max(idx - 1, 0)] for idx in indices_for_normalization]
    normalization_ref_counts = [Datum(int_array=int_array_ve[idx], float_array=float_array_ve[idx]).get(Data.REF_COUNT) for idx in indices_for_normalization]
    normalization_ref_end_indices = [(start + ref_count) for start, ref_count in zip(normalization_read_start_indices, normalization_ref_counts)]

    # for every index in the normalization set, get all the reads of the corresponding datum.  Stack all these reads to
    # obtain the reads normalization array
    reads_for_normalization_re = np.vstack([reads_re[start:end] for start, end in zip(normalization_read_start_indices, normalization_ref_end_indices)])
    reads_for_normalization_distance_columns_re = reads_for_normalization_re[:, 6:7]
    read_quantile_transform = QuantileTransformer(n_quantiles=100, output_distribution='normal')
    read_quantile_transform.fit(reads_for_normalization_distance_columns_re)

    return read_quantile_transform


def make_normalized_mmap_data(dataset_files, sources: List[int]=None) -> MemoryMappedData:
    """
    given unnormalized plain text dataset files from Mutect2, normalize data and save as tarfile of memory mapped numpy arrays

    In addition to quantile-normalizing read tensors it also enlarges the info tensors
    :param dataset_files:
    :param sources if None, source is set to 0; if singleton list, all files are given that source; otherwise one source per file
    """
    raw_memory_mapped_data = write_raw_unnormalized_data_to_memory_maps(dataset_files, sources)

    normalized_generator = normalized_data_generator(raw_memory_mapped_data)
    return MemoryMappedData.from_generator(reads_datum_source=normalized_generator,
        estimated_num_data=raw_memory_mapped_data.num_data, estimated_num_reads=raw_memory_mapped_data.num_reads)


def get_normalization_set(raw_int_array_ve, raw_float_array_ve) -> List[int]:
    """
    # we need a set of data that are pretty reliably not artifacts for the quantile normalization.  If we don't do this
    # and naively use the quantiles from the data as a whole we create a nasty domain shift where the artifact/non-artifact
    # balance of test data differs fromm that of the training data and thus the normalization is different, leading to a skew
    # of the input tensors *even* if the data re derived from the same sample prep and sequencing technology!

    # It is a good idea to inject a few tens of thousands of germline variants into test data to be able to make this normalization
    # set, but the following scheme has a back-up plan in case we don't have that (or if there's no information on
    # germline allele frequencies).
    """

    indices_for_normalization_queue = PriorityQueue(maxsize=MAX_NUM_DATA_FOR_NORMALIZATION)
    for n in range(len(raw_int_array_ve)):
        raw_datum = Datum(int_array=raw_int_array_ve[n], float_array=raw_float_array_ve[n])

        if indices_for_normalization_queue.full():
            indices_for_normalization_queue.get()  # pop the lowest-priority element i.e. the worst-suited for normalization

        # priority is negative squared difference between original allele fraction and 1/2
        # thus most germline het-like data have highest priority
        priority = -((raw_datum.get(Data.ORIGINAL_ALT_COUNT) / raw_datum.get(Data.ORIGINAL_DEPTH)) - 0.5) ** 2

        indices_for_normalization_queue.put((priority, n))
    all_indices_for_normalization = []
    good_indices_for_normalization = []
    while not indices_for_normalization_queue.empty():
        priority, idx = indices_for_normalization_queue.get()

        all_indices_for_normalization.append(idx)
        if priority > - 0.2**2:    # AF between 0.3 and 0.7
            good_indices_for_normalization.append(idx)

    indices_for_normalization = good_indices_for_normalization if len(good_indices_for_normalization) > MIN_NUM_DATA_FOR_NORMALIZATION else all_indices_for_normalization

    indices_for_normalization.sort()  # sorting indices makes traversing memory maps faster
    return indices_for_normalization


# this normalizes the buffer and also prepends new features to the info tensor
def normalize_raw_data_list(buffer: List[Datum], read_quantile_transform) -> List[Datum]:
    # 2D array.  Rows are ref/alt reads, columns are read features
    all_reads_re = np.vstack([datum.reads_re for datum in buffer])

    # 2D array.  Rows are read sets, columns are info features
    all_info_ve = np.vstack([datum.get_info_1d() for datum in buffer])

    # INFO features from GATK are
    # 0, 1 haplotype equivalence share of 2nd and 3rd most supported haplotype groups (among all haplotypes with
    # the alt allele)
    # 2 edit distance, excluding variant site, of best alt haplotype vs closest germline haplotype
    # 3 fraction of reads supporting alt allele that support the single best alt haplotype
    # 4 TLOD / alt count

    hap_equiv_columns = all_info_ve[:, 0:2]
    hap_equiv_binary_1 = (hap_equiv_columns < 0.1)
    hap_equiv_binary_2 = (hap_equiv_columns >= 0.1) & (hap_equiv_columns < 0.25)
    hap_equiv_binary_3 = (hap_equiv_columns >= 0.25)

    edit_dist_column = all_info_ve[:, 2]
    edit_dist_binary = np.full((len(all_info_ve), 4), 0, dtype=all_info_ve.dtype)
    edit_dist_binary[:, 0] = (edit_dist_column == 0)
    edit_dist_binary[:, 1] = (edit_dist_column == 1)
    edit_dist_binary[:, 2] = (edit_dist_column == 2)
    edit_dist_binary[:, 3] = (edit_dist_column >= 3)

    hap_dom_column = all_info_ve[:, 3]
    hap_dom_binary = np.full((len(all_info_ve), 2), 0, dtype=all_info_ve.dtype)
    hap_dom_binary[:, 0] = (hap_dom_column > 0.9)
    hap_dom_binary[:, 1] = (hap_dom_column <= 0.9) & (hap_dom_column > 0.7)

    # STR INFO features are
    # 5, 6 insertion_length, deletion_length of variant
    # 7 STR repeat unit length
    # 8 number of STR repeat units
    # 9 repeats before variant
    # 10 repeats after variant

    indel_lengths_ve = all_info_ve[:, 5:7] / 10

    str_unit_length_col = all_info_ve[:, 7].reshape(-1,1)
    str_unit_length_binaries = np.hstack([str_unit_length_col == 1, str_unit_length_col == 2, str_unit_length_col == 3,
                                str_unit_length_col == 4, str_unit_length_col >= 5])

    str_total_length = all_info_ve[:, 7] * all_info_ve[:, 8]
    str_total_length_binaries = binarize_str_length_column(str_total_length)

    str_length_before = all_info_ve[:, 7] * all_info_ve[:, 9]
    str_length_before_binaries = binarize_str_length_column(str_length_before)

    str_length_after = all_info_ve[:, 7] * all_info_ve[:, 10]
    str_length_after_binaries = binarize_str_length_column(str_length_after)

    str_info_array_ve = np.hstack([indel_lengths_ve, str_unit_length_binaries, str_total_length_binaries, str_length_before_binaries, str_length_after_binaries])

    binary_info_array_ve = np.hstack(
        (hap_equiv_binary_1, hap_equiv_binary_2, hap_equiv_binary_3, edit_dist_binary, hap_dom_binary, str_info_array_ve))

    tlod_over_nalt = all_info_ve[:, 4]
    orig_alt_counts = np.array([datum.get(Data.ORIGINAL_ALT_COUNT) for datum in buffer])
    orig_depths = np.array([datum.get(Data.ORIGINAL_DEPTH) for datum in buffer])
    orig_ref_counts = orig_depths - orig_alt_counts

    natural_log_tlod = LOG10_TO_LN * tlod_over_nalt * orig_alt_counts
    tlod_correction = gammaln(orig_depths + 2) - gammaln(orig_alt_counts + 1) - gammaln(orig_ref_counts + 1)
    average_qual_feature = ((natural_log_tlod + tlod_correction) / orig_alt_counts)/10

    all_info_transformed_ve = np.hstack([binary_info_array_ve, average_qual_feature.reshape(-1,1)])

    from_read_ends_columns_re = all_reads_re[:, 4:6]
    from_read_ends_transformed_re = np.tanh(from_read_ends_columns_re / DISTANCE_FROM_END_SATURATION)
    from_frag_ends_columns_re = all_reads_re[:, 7:9]
    from_frag_ends_transformed_re = np.tanh(from_frag_ends_columns_re / DISTANCE_FROM_END_SATURATION)
    distance_columns_re = all_reads_re[:, 6:7]
    distance_columns_transformed_re = read_quantile_transform.transform(distance_columns_re)
    float_read_columns_re = np.hstack([from_read_ends_transformed_re, from_frag_ends_transformed_re, distance_columns_transformed_re])


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

    read_counts = np.array([len(datum.reads_re) for datum in buffer])
    read_index_ranges = np.cumsum(read_counts)

    map_qual_column = all_reads_re[:, 0]
    map_qual_boolean = np.full((len(all_reads_re), 4), True, dtype=bool)
    map_qual_boolean[:, 0] = (map_qual_column > 59)
    map_qual_boolean[:, 1] = (map_qual_column <= 59) & (map_qual_column >= 40)
    map_qual_boolean[:, 2] = (map_qual_column < 40) & (map_qual_column >= 20)
    map_qual_boolean[:, 3] = (map_qual_column < 20)

    base_qual_column = all_reads_re[:, 1]
    base_qual_boolean = np.full((len(all_reads_re), 4), True, dtype=bool)
    base_qual_boolean[:, 0] = (base_qual_column >= 30)
    base_qual_boolean[:, 1] = (base_qual_column < 30) & (base_qual_column >= 20)
    base_qual_boolean[:, 2] = (base_qual_column < 20) & (base_qual_column >= 10)
    base_qual_boolean[:, 3] = (base_qual_column < 10)

    strand_and_orientation_boolean = all_reads_re[:, 2:4] < 0.5
    error_counts_boolean_1 = all_reads_re[:, 9:] < 0.5
    error_counts_boolean_2 = (all_reads_re[:, 9:] > 0.5) & (all_reads_re[:, 9:] < 1.5)
    error_counts_boolean_3 = (all_reads_re[:, 9:] > 1.5)

    boolean_output_array_re = np.hstack((map_qual_boolean, base_qual_boolean, strand_and_orientation_boolean,
               error_counts_boolean_1, error_counts_boolean_2, error_counts_boolean_3))

    # axis = 1 is essential so that each row (read) of the packed data corresponds to a row of the unpacked data
    packed_output_array = np.packbits(boolean_output_array_re, axis=1)

    assert packed_output_array.dtype == np.uint8
    assert packed_output_array.shape[1] == NUMBER_OF_BYTES_IN_PACKED_READ, f"boolean array shape {boolean_output_array_re.shape}, packed shape {packed_output_array.shape}"

    # the conversion to uint8 maps the range [-4,4] to [0,255].  Anything less than -4 is mapped to 0 and
    # anything greater than 4 is mapped to 255.
    distance_columns_output = convert_quantile_normalized_to_uint8(float_read_columns_re)
    assert packed_output_array.dtype == distance_columns_output.dtype
    output_uint8_reads_array = np.hstack((packed_output_array, distance_columns_output))

    normalized_result = []
    raw_datum: Datum
    for n, raw_datum in enumerate(buffer):
        ref_start_index = 0 if n == 0 else read_index_ranges[n - 1]     # first index of this datum's reads
        alt_end_index = read_index_ranges[n]
        alt_start_index = ref_start_index + raw_datum.get(Data.REF_COUNT)

        # TODO: maybe we could also have columnwise nonparametric test statistics, like for example we record the
        # TODO: quantiles over all ref reads
        alt_distance_medians_e = np.median(float_read_columns_re[alt_start_index:alt_end_index, :], axis=0)
        alt_boolean_means_e = np.mean(boolean_output_array_re[alt_start_index:alt_end_index, :], axis=0)
        extra_info_e = np.hstack((alt_distance_medians_e, alt_boolean_means_e))

        output_reads_re = output_uint8_reads_array[ref_start_index:alt_end_index]
        output_datum: Datum = Datum(int_array=raw_datum.int_array, float_array=raw_datum.float_array,
                                              reads_re=output_reads_re, compressed_reads=True)

        output_datum.set_info_1d(np.hstack((all_info_transformed_ve[n], extra_info_e)))
        normalized_result.append(output_datum)

    return normalized_result


def line_to_array(line: str) -> np.ndarray:
    tokens = line.strip().split()
    floats = [max(min(MAX_VALUE, float(token)), -MAX_VALUE) for token in tokens]
    return np.array(floats, dtype=FLOAT_DTYPE)


def read_2d_array(file, num_lines: int) -> np.ndarray:
    if num_lines == 0:
        return None
    lines = [file.readline() for _ in range(num_lines)]
    return np.vstack([line_to_array(line) for line in lines])


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
    return np.ndarray.astype((data - 128) / 32, FLOAT_DTYPE)
