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

import collections
import io
import math
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Generator
from typing import Iterable
from typing import Iterator
from typing import List

import numpy as np
import torch
from sklearn.preprocessing import QuantileTransformer
from tqdm import tqdm

from permutect.data.count_binning import cap_alt_count
from permutect.data.count_binning import cap_ref_count
from permutect.data.datum import FLOAT_DTYPE
from permutect.data.datum import INTEGER_DTYPE
from permutect.data.datum import NUMBER_OF_BYTES_IN_PACKED_READ
from permutect.data.datum import Data
from permutect.data.datum import Datum
from permutect.data.datum import gatk_variant_features
from permutect.data.datum import get_data_field
from permutect.data.datum import set_data_field_column
from permutect.data.memory_mapped_data import MemoryMappedData
from permutect.data.memory_mapped_data import StackedData
from permutect.data.memory_mapped_data import stacked_read_offsets
from permutect.utils.enums import Label
from permutect.utils.enums import Variation
from permutect.utils.math_utils import log_binomial_coefficient

MAX_VALUE = 10000
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


@dataclass(frozen=True)
class PlainTextFileIndex:
    """
    Where each datum with alt reads starts in a plain text dataset file, from a pass over the file that skips the rest
    """

    dataset_file: str
    source: int
    data_offsets: np.ndarray  # the byte offset of each datum's label line
    num_reads: int  # the ref and alt reads of these data, before their counts are capped

    @classmethod
    def of(cls, dataset_file: str, source: int) -> PlainTextFileIndex:
        num_reads, data_offsets = 0, []
        with open(dataset_file, "rb") as file:  # binary, so that tell() gives byte offsets
            while label_line := file.readline():  # the label line is the first line of each datum
                offset = file.tell() - len(label_line)
                file.readline()  # skip the contig:position,ref->alt line
                file.readline()  # skip the reference sequence line
                file.readline()  # skip the info array line

                # get the read tensor sizes
                ref_tensor_size, alt_tensor_size, normal_ref_tensor_size, normal_alt_tensor_size = map(
                    int, file.readline().split()
                )

                # skip the read tensors
                for _ in range(ref_tensor_size + alt_tensor_size + normal_ref_tensor_size + normal_alt_tensor_size):
                    file.readline()

                file.readline()  # skip the original depths line
                file.readline()  # skip the seq error log likelihood line
                file.readline()  # skip the normal seq error log likelihood line
                if alt_tensor_size > 0:  # data with no alts is skipped
                    data_offsets.append(offset)
                    num_reads += ref_tensor_size + alt_tensor_size  # we don't use normal reads

        return cls(dataset_file, source, np.array(data_offsets, dtype=np.int64), num_reads)

    def __len__(self) -> int:
        return len(self.data_offsets)

    def segment(self, start: int, end: int) -> PlainTextSegment:
        """
        the file's data start:end, counting only data with alt reads
        """
        return PlainTextSegment(self.dataset_file, self.source, int(self.data_offsets[start]), end - start)


@dataclass(frozen=True)
class PlainTextSegment:
    """
    num_data consecutive data with alt reads of a plain text dataset file, the first of which starts at a byte offset
    """

    dataset_file: str
    source: int
    start: int
    num_data: int


@dataclass(frozen=True)
class PlainTextChunk:
    """
    Data of plain text dataset files that are normalized together: a segment of each file the chunk spans, and the seed
    of its downsampling and quantile transform, so that any process, in any order, normalizes it the same
    """

    segments: tuple[PlainTextSegment, ...]
    seed: int

    def normalize(self) -> StackedData:
        raw = read_raw_stacked_data(self.segments, torch.Generator().manual_seed(self.seed))
        return normalize_raw_data(raw, make_read_quantile_transform(raw, random_state=self.seed))


def plain_text_chunks(file_indices: list[PlainTextFileIndex], seed: int) -> list[PlainTextChunk]:
    """
    The chunks of the data of plain text dataset files, as when the files were read in turn: of all the files' data in
    order, ceil(num_data / NUM_RAW_DATA_TO_NORMALIZE_AT_ONCE) chunks of num_data // num_chunks data, the last of which also
    takes the rest.  Chunk n has seed + n.
    """
    num_data = sum(len(file_index) for file_index in file_indices)
    num_chunks = math.ceil(num_data / NUM_RAW_DATA_TO_NORMALIZE_AT_ONCE)
    data_per_chunk = num_data // num_chunks
    chunk_starts = [chunk * data_per_chunk for chunk in range(num_chunks)] + [num_data]

    # the index of each file's first datum among all the files' data
    file_starts = np.cumsum([0] + [len(file_index) for file_index in file_indices])
    chunks = []
    for n, (chunk_start, chunk_end) in enumerate(zip(chunk_starts[:-1], chunk_starts[1:])):
        segments = tuple(
            file_index.segment(max(chunk_start, file_start) - file_start, min(chunk_end, file_end) - file_start)
            for file_index, file_start, file_end in zip(file_indices, file_starts[:-1], file_starts[1:])
            if file_start < chunk_end and chunk_start < file_end
        )
        chunks.append(PlainTextChunk(segments, seed + n))
    return chunks


def normalized_in_order(chunks: list[PlainTextChunk], num_processes: int) -> Iterator[StackedData]:
    """
    The normalized data of the chunks, in order.  With several processes, at most twice as many chunks as processes are
    pending at once, which bounds the memory held by normalized chunks that await their turn.
    """
    if num_processes == 1:
        yield from (chunk.normalize() for chunk in chunks)
        return

    # spawn rather than fork, which can deadlock a child of a process whose torch or numpy has started threads
    with ProcessPoolExecutor(max_workers=num_processes, mp_context=multiprocessing.get_context("spawn")) as executor:
        pending = collections.deque()
        for chunk in chunks:
            pending.append(executor.submit(chunk.normalize))
            if len(pending) >= 2 * num_processes:
                yield pending.popleft().result()
        while pending:
            yield pending.popleft().result()


def read_raw_stacked_data(segments: Iterable[PlainTextSegment], generator: torch.Generator) -> StackedData:
    """
    The data of segments of plain text dataset files, in order: StackedData.of the data that read_raw_unnormalized_data
    gives, but several times faster.  Rather than making a Datum of each datum, it collects the lines of all the data
    and converts their numbers at once.  The reads beyond the caps on ref and alt counts are dropped as
    copy_with_downsampled_reads drops them, drawing from the generator in the same order.
    """
    records, read_lines = [], []
    for segment in segments:
        with open(segment.dataset_file, "rb") as binary_file:
            binary_file.seek(segment.start)
            file = io.TextIOWrapper(binary_file)
            num_from_segment = 0
            while num_from_segment < segment.num_data and (label := file.readline().strip()):
                variant_line, ref_sequence, info_line = file.readline(), file.readline().strip(), file.readline()
                ref_count, alt_count, normal_ref_count, normal_alt_count = map(int, file.readline().split())
                datum_read_lines = [file.readline() for _ in range(ref_count + alt_count)]
                for _ in range(normal_ref_count + normal_alt_count):  # the normal reads aren't used, only their counts
                    file.readline()
                # the original counts, the sequencing error log likelihood, and the normal's
                last_lines = file.readline(), file.readline(), file.readline()
                if alt_count > 0:  # data without alt reads are skipped
                    datum_lines = (variant_line, ref_sequence, info_line, *last_lines)
                    records.append((label, segment.source, ref_count, alt_count, *datum_lines))
                    read_lines += datum_read_lines
                    num_from_segment += 1

    labels, sources, ref_counts, alt_counts, variant_lines, ref_sequences, info_lines, *last_lines = zip(*records)
    original_counts_lines, seq_error_lines, normal_seq_error_lines = last_lines
    variants = [variant_columns(line, ref_sequence) for line, ref_sequence in zip(variant_lines, ref_sequences)]
    contigs, positions, variant_types, str_info, haplotypes, ref_base5s, alt_base5s = zip(*variants)

    # each datum's reads, or, for those with more ref or alt reads than the caps, the ones copy_with_downsampled_reads
    # chooses.  The rows of the reads of data in between, which are all kept, are kept in runs.
    kept_ref_counts, kept_alt_counts = [cap_ref_count(n) for n in ref_counts], [cap_alt_count(n) for n in alt_counts]
    kept_rows, read_start, all_kept_from = [], 0, 0
    for ref_count, alt_count, kept_ref_count, kept_alt_count in zip(
        ref_counts, alt_counts, kept_ref_counts, kept_alt_counts
    ):
        if kept_ref_count < ref_count or kept_alt_count < alt_count:
            ref_rows = torch.randperm(ref_count, generator=generator)[:kept_ref_count].numpy()
            alt_rows = ref_count + torch.randperm(alt_count, generator=generator)[:kept_alt_count].numpy()
            kept_rows += [np.arange(all_kept_from, read_start), read_start + ref_rows, read_start + alt_rows]
            all_kept_from = read_start + ref_count + alt_count
        read_start += ref_count + alt_count
    kept_rows.append(np.arange(all_kept_from, read_start))
    # the first column, the read group index, isn't used
    reads_re = lines_to_array(read_lines)[np.concatenate(kept_rows), 1:]

    # the STR info follows GATK's info, as Datum.from_gatk appends it
    info_ve = np.hstack((lines_to_array(info_lines), np.vstack(str_info)))
    num_data, haplotypes_vh = len(records), np.vstack(haplotypes)
    int_array_ve = np.zeros((num_data, Data.NUM_SCALAR_INT_ELEMENTS + haplotypes_vh.shape[1]), dtype=INTEGER_DTYPE)
    int_array_ve[:, Data.HAPLOTYPES_START_IDX :] = haplotypes_vh
    # the float fields that GATK doesn't give are NaN, as in Datum.from_gatk
    float_array_ve = np.full((num_data, Data.NUM_SCALAR_FLOAT_ELEMENTS + info_ve.shape[1]), np.nan, dtype=FLOAT_DTYPE)
    float_array_ve[:, Data.INFO_START_IDX :] = info_ve

    original_depths, original_alt_counts, original_normal_depths, original_normal_alt_counts = np.array(
        [line.split() for line in original_counts_lines], dtype=np.int64
    ).T
    fields = {
        Data.REF_COUNT: kept_ref_counts,
        Data.ALT_COUNT: kept_alt_counts,
        Data.LABEL: [Label[label] for label in labels],
        Data.VARIANT_TYPE: variant_types,
        Data.SOURCE: sources,
        Data.ORIGINAL_DEPTH: original_depths,
        Data.ORIGINAL_ALT_COUNT: original_alt_counts,
        Data.ORIGINAL_NORMAL_DEPTH: original_normal_depths,
        Data.ORIGINAL_NORMAL_ALT_COUNT: original_normal_alt_counts,
        Data.CONTIG: contigs,
        Data.POSITION: positions,
        Data.REF_ALLELE_AS_BASE_5: ref_base5s,
        Data.ALT_ALLELE_AS_BASE_5: alt_base5s,
        Data.SEQ_ERROR_LOG_LK: [read_float(line) for line in seq_error_lines],
        Data.NORMAL_SEQ_ERROR_LOG_LK: [read_float(line) for line in normal_seq_error_lines],
    }
    for field, values in fields.items():
        set_data_field_column(int_array_ve, float_array_ve, field, values)

    read_offsets = stacked_read_offsets(np.add(kept_ref_counts, kept_alt_counts))
    return StackedData(int_array_ve, float_array_ve, reads_re, read_offsets)


def variant_columns(variant_line: str, ref_sequence: str) -> tuple:
    """
    A datum's contig index, position and variant type, from its contig:position,ref->alt line, and what
    gatk_variant_features derives from its alleles and reference context
    """
    locus, mutation = variant_line.strip().split(",")
    contig, position = map(int, locus.split(":"))  # contig is an integer *index* from a sequence dictionary
    ref_allele, alt_allele = mutation.strip().split("->")
    features = gatk_variant_features(ref_sequence, ref_allele, alt_allele)
    return contig, position, Variation.get_type(ref_allele, alt_allele), *features


def read_raw_unnormalized_data(
    dataset_file, source: int = 0, start: int = 0, generator: torch.Generator = None
) -> Generator[Datum, None, None]:
    """
    generator that yields the data with alt reads of a plain text dataset file, starting at a byte offset, one Datum at a
    time.  read_raw_stacked_data reads data much faster.

    :param generator: the torch random number generator for downsampling reads beyond the caps on ref and alt counts
    """
    with open(dataset_file, "rb") as binary_file:
        binary_file.seek(start)
        file = io.TextIOWrapper(binary_file)
        while label_str := file.readline().strip():
            label = Label[label_str]

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

            # the ref reads and then the alt reads, one per line, parsed at once.  Their first column, the read group
            # index, isn't used.
            reads_re = read_2d_array(file, ref_tensor_size + alt_tensor_size)

            # the normal reads aren't used, only their counts
            for _ in range(normal_ref_tensor_size + normal_alt_tensor_size):
                next(file)

            original_depth, original_alt_count, original_normal_depth, original_normal_alt_count = read_integers(
                file.readline()
            )
            # this is -log10ToLog(tlod) - log(tumorDepth + 1);
            seq_error_log_lk = read_float(file.readline())
            # this is -log10ToLog(nalod) - log(normalDepth + 1)
            normal_seq_error_log_lk = read_float(file.readline())

            if alt_tensor_size > 0:
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
                    ref_reads_array_re=reads_re[:ref_tensor_size, 1:] if ref_tensor_size > 0 else None,
                    alt_reads_array_re=reads_re[ref_tensor_size:, 1:],
                )

                ref_count = cap_ref_count(datum.get(Data.REF_COUNT))
                alt_count = cap_alt_count(datum.get(Data.ALT_COUNT))
                yield datum.copy_with_downsampled_reads(ref_count, alt_count, generator)


def make_read_quantile_transform(raw: StackedData, random_state: int = None):
    """
    :param random_state: the seed of the random subsample of the reads to which the transform is fit
    """
    # fit to the fragment lengths of the ref reads of every datum, which come first among its reads
    ref_counts, read_starts = get_data_field(raw.int_array_ve, None, Data.REF_COUNT), raw.read_offsets[:-1]
    ref_reads_re = np.vstack([raw.reads_re[start : start + count] for start, count in zip(read_starts, ref_counts)])
    transform = QuantileTransformer(n_quantiles=100, output_distribution="normal", random_state=random_state)
    return transform.fit(ref_reads_re[:, 6:7])


def make_normalized_mmap_data(
    dataset_files, sources: List[int] = None, num_processes: int = 1, seed: int = 0
) -> MemoryMappedData:
    """
    The data with alt reads of unnormalized plain text dataset files from Mutect2, normalized into memory-mapped arrays,
    which save_to_tarfile saves.  In addition to quantile-normalizing the reads it replaces the info with new features.

    :param dataset_files: the plain text files, whose data are taken in order
    :param sources if None, source is set to 0; if singleton list, all files are given that source; otherwise one source per file
    :param num_processes: the number of processes that read and normalize chunks of the data at once
    :param seed: chunk n is normalized with seed + n, so that the data don't depend on the number of processes
    """
    # no sources gives every file source 0, a single source is every file's, and otherwise file n has source n
    file_sources = [0 if sources is None else sources[0 if len(sources) == 1 else n] for n in range(len(dataset_files))]
    file_indices = [PlainTextFileIndex.of(file, source) for file, source in zip(dataset_files, file_sources)]
    chunks = plain_text_chunks(file_indices, seed)
    normalized_chunks = tqdm(normalized_in_order(chunks, num_processes), total=len(chunks), mininterval=60)

    num_data, num_reads = sum(len(index) for index in file_indices), sum(index.num_reads for index in file_indices)
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

    # the float arrays keep their scalar elements, and their info becomes the new features.  They are stored as a Datum
    # holds them, as FLOAT_DTYPE, rather than as the float64 that the new features are computed in, which took four
    # times the space.
    new_info_ve = np.hstack((all_info_transformed_ve, alt_distance_medians_ve, alt_boolean_means_ve))
    normalized_float_array_ve = np.hstack((float_array_ve[:, : Data.INFO_START_IDX], new_info_ve)).astype(FLOAT_DTYPE)
    # the compressed reads are the raw reads row for row, so they have the same offsets
    return StackedData(int_array_ve, normalized_float_array_ve, output_uint8_reads_array, raw.read_offsets)


def clipped(values: np.ndarray) -> np.ndarray:
    """
    float64 numbers, clipped in place to [-MAX_VALUE, MAX_VALUE], as FLOAT_DTYPE.  NaN becomes MAX_VALUE, as it did when
    each number was clipped with Python's min and max.
    """
    # np.nan_to_num checks for infinities too, which the clipping handles, and took a third of the time for each line
    values[np.isnan(values)] = MAX_VALUE
    return np.clip(values, -MAX_VALUE, MAX_VALUE, out=values).astype(FLOAT_DTYPE)


def tokens_to_array(tokens) -> np.ndarray:
    """
    The numbers of a list of strings, or of a list of lists of strings with as many in each, as clipped gives them
    """
    return clipped(np.array(tokens, dtype=np.float64))


def lines_to_array(lines: list[str]) -> np.ndarray:
    """
    The numbers of lines with as many in each, as a 2D array with a row for each line, as tokens_to_array gives them.
    np.loadtxt parses each number as Python's float does, but several times faster.
    """
    return clipped(np.loadtxt(lines, dtype=np.float64, ndmin=2, comments=None))


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
