import numpy as np
import pytest
import torch

from permutect.data import plain_text_data
from permutect.data.count_binning import MAX_ALT_COUNT
from permutect.data.count_binning import MAX_REF_COUNT
from permutect.data.datum import FLOAT_DTYPE
from permutect.data.datum import INTEGER_DTYPE
from permutect.data.datum import Data
from permutect.data.datum import get_data_field
from permutect.data.memory_mapped_data import stacked_read_offsets
from permutect.data.plain_text_data import MAX_VALUE
from permutect.data.plain_text_data import PlainTextChunk
from permutect.data.plain_text_data import PlainTextFileIndex
from permutect.data.plain_text_data import PlainTextSegment
from permutect.data.plain_text_data import alt_read_medians_and_means
from permutect.data.plain_text_data import convert_quantile_normalized_to_uint8
from permutect.data.plain_text_data import convert_uint8_to_quantile_normalized
from permutect.data.plain_text_data import lines_to_array
from permutect.data.plain_text_data import plain_text_chunks
from permutect.data.plain_text_data import read_raw_stacked_data
from permutect.test.test_file_names import SMALL_PLAIN_TEXT_DATA
from permutect.utils.enums import Label
from permutect.utils.enums import Variation


def test_numbers_are_parsed_and_clipped():
    # to [-MAX_VALUE, MAX_VALUE], with NaN at the top
    array = lines_to_array(["1.5 -2 3e2\n", "nan inf -inf\n", "20000 -20000 0\n"])
    expected = [[1.5, -2, 300], [MAX_VALUE, MAX_VALUE, -MAX_VALUE], [MAX_VALUE, -MAX_VALUE, 0]]
    assert array.dtype == FLOAT_DTYPE and np.array_equal(array, np.array(expected, dtype=FLOAT_DTYPE))
    assert lines_to_array(["1 2 3\n"]).shape == (1, 3)


@pytest.mark.parametrize("text", ["1 2 3\n4 abc 6\n", "1 2 3\n4 5\n"])
def test_malformed_lines_are_rejected(text):
    with pytest.raises(ValueError):
        lines_to_array(text.splitlines(keepends=True))


def plain_text_datum(label, variant, ref_sequence, info, counts, reads, original_counts, seq_error, normal_seq_error):
    lines = [label, variant, ref_sequence, info, counts, *reads, original_counts, seq_error, normal_seq_error]
    return "".join(line + "\n" for line in lines)


def numbered_reads(datum: int, allele: int, num_reads: int) -> list[str]:
    # each read records where it came from: its read group, then the datum, 0 for ref or 1 for alt, and its number
    return [f"0 {datum} {allele} {n}" for n in range(num_reads)]


PLAIN_TEXT_DATA = (
    # two ref reads and an alt read, and a normal read that isn't used
    plain_text_datum(
        "ARTIFACT",
        "1:13118,A->G",
        "GAGGAAAGTGAGGTTGCCTGC",
        "0.5 1 2",
        "2 1 1 0",
        numbered_reads(0, 0, 2) + numbered_reads(0, 1, 1) + ["9 9 9 9"],
        "40 6 30 0",
        "-12.5",
        "-0.25",
    )
    # no alt reads, so it's skipped
    + plain_text_datum(
        "VARIANT", "1:13302,C->T", "GTCCTGGACACGCTGTTGGCC", "0 0 0", "1 0 0 0", ["0 1 0 0"], "20 0 10 0", "-1", "-2"
    )
    # more ref and alt reads than the caps, and normal alt reads
    + plain_text_datum(
        "UNLABELED",
        "2:5000,C->T",
        "GTCCTGGACACGCTGTTGGCC",
        "0.25 3 4",
        "12 16 0 2",
        numbered_reads(2, 0, 12) + numbered_reads(2, 1, 16) + ["9 9 9 9"] * 2,
        "100 16 50 2",
        "-30",
        "-3.5",
    )
)


def test_plain_text_data_are_parsed(tmp_path):
    dataset_file = tmp_path / "data.dataset"
    dataset_file.write_text(PLAIN_TEXT_DATA)
    file_index = PlainTextFileIndex.of(str(dataset_file), source=3)
    assert len(file_index) == 2 and file_index.num_reads == 3 + 12 + 16

    # the downsampling draws from its generator, not from torch's global one
    torch_state = torch.get_rng_state()
    data = read_raw_stacked_data([file_index.segment(0, 2)], torch.Generator().manual_seed(1))
    assert torch.equal(torch.get_rng_state(), torch_state)

    def column(field: Data) -> list:
        return get_data_field(data.int_array_ve, data.float_array_ve, field).tolist()

    assert column(Data.LABEL) == [Label.ARTIFACT, Label.UNLABELED] and column(Data.SOURCE) == [3, 3]
    assert column(Data.CONTIG) == [1, 2] and column(Data.POSITION) == [13118, 5000]
    assert column(Data.VARIANT_TYPE) == [Variation.SNV, Variation.SNV]
    assert column(Data.REF_COUNT) == [2, MAX_REF_COUNT] and column(Data.ALT_COUNT) == [1, MAX_ALT_COUNT]
    assert column(Data.ORIGINAL_DEPTH) == [40, 100] and column(Data.ORIGINAL_ALT_COUNT) == [6, 16]
    assert column(Data.ORIGINAL_NORMAL_DEPTH) == [30, 50] and column(Data.ORIGINAL_NORMAL_ALT_COUNT) == [0, 2]
    assert column(Data.SEQ_ERROR_LOG_LK) == [-12.5, -30] and column(Data.NORMAL_SEQ_ERROR_LOG_LK) == [-0.25, -3.5]
    # GATK's info comes first, followed by the STR info
    assert data.float_array_ve[:, Data.INFO_START_IDX : Data.INFO_START_IDX + 3].tolist() == [[0.5, 1, 2], [0.25, 3, 4]]

    # the read group column is dropped, and each datum's reads are its ref reads, then its alt reads
    assert data.read_offsets.tolist() == [0, 3, 3 + MAX_REF_COUNT + MAX_ALT_COUNT]
    assert data.reads_re[:3].tolist() == [[0, 0, 0], [0, 0, 1], [0, 1, 0]]
    # beyond the caps, a random choice of distinct reads of each allele
    capped_ref_reads, capped_alt_reads = data.reads_re[3 : 3 + MAX_REF_COUNT], data.reads_re[3 + MAX_REF_COUNT :]
    for reads, allele, num_reads in ((capped_ref_reads, 0, 12), (capped_alt_reads, 1, 16)):
        assert all(datum == 2 and read_allele == allele for datum, read_allele, _ in reads.tolist())
        numbers = reads[:, 2].tolist()
        assert len(set(numbers)) == len(numbers) and set(numbers) <= set(range(num_reads))

    # the same seed chooses the same reads, segments that split the data read the same data, and a segment can start at
    # any datum
    split = read_raw_stacked_data(
        [file_index.segment(0, 1), file_index.segment(1, 2)], torch.Generator().manual_seed(1)
    )
    for name in ("int_array_ve", "float_array_ve", "reads_re", "read_offsets"):
        assert np.array_equal(getattr(split, name), getattr(data, name), equal_nan=True), name
    second = read_raw_stacked_data([file_index.segment(1, 2)], torch.Generator())
    assert get_data_field(second.int_array_ve, None, Data.POSITION).tolist() == [5000]


def test_file_index_finds_each_datum_with_alt_reads():
    # its segments read the same data as a segment of the whole file
    file_index = PlainTextFileIndex.of(SMALL_PLAIN_TEXT_DATA, source=0)
    whole_file = read_raw_stacked_data([file_index.segment(0, len(file_index))], torch.Generator())
    positions = get_data_field(whole_file.int_array_ve, None, Data.POSITION).tolist()
    assert len(positions) == len(file_index)
    for start, end in ((0, 1), (10, 25), (len(positions) - 3, len(positions))):
        segment_data = read_raw_stacked_data([file_index.segment(start, end)], torch.Generator())
        assert get_data_field(segment_data.int_array_ve, None, Data.POSITION).tolist() == positions[start:end]


def test_alt_read_medians_and_means_are_those_of_each_datum():
    # as np.median and np.mean gave them datum by datum, with odd and even numbers of alt reads, some ref reads or none,
    # and tied values
    rng = np.random.default_rng(0)
    ref_counts, alt_counts = np.array([0, 2, 1, 0, 3, 5, 0]), np.array([1, 2, 3, 4, 5, 6, 9])
    read_offsets = stacked_read_offsets(ref_counts + alt_counts)
    floats_re = np.round(rng.normal(size=(read_offsets[-1], 5)), 1)
    booleans_re = rng.random((read_offsets[-1], 7)) < 0.5

    medians, means = alt_read_medians_and_means(floats_re, booleans_re, read_offsets, ref_counts)
    for n in range(len(ref_counts)):
        alt_reads = slice(read_offsets[n] + ref_counts[n], read_offsets[n + 1])
        assert medians[n].tobytes() == np.median(floats_re[alt_reads], axis=0).tobytes()
        assert means[n].tobytes() == np.mean(booleans_re[alt_reads], axis=0).tobytes()


def test_normalized_data_have_the_dtypes_of_a_datum():
    # the float arrays used to be float64, four times the space, although a Datum made of them holds float16
    normalized = plain_text_data.make_normalized_mmap_data([SMALL_PLAIN_TEXT_DATA])
    assert normalized.int_mmap.dtype == INTEGER_DTYPE and normalized.float_mmap.dtype == FLOAT_DTYPE
    datum = normalized.get_datum(0)
    assert datum.int_array.dtype == INTEGER_DTYPE and datum.float_array.dtype == FLOAT_DTYPE


def test_chunks_take_the_files_data_in_order(monkeypatch):
    # those of all the files' data in turn, so that a chunk can span the end of one file and the start of the next
    monkeypatch.setattr(plain_text_data, "NUM_RAW_DATA_TO_NORMALIZE_AT_ONCE", 3000)
    file_sizes = [2500, 10, 7000]
    file_indices = [
        PlainTextFileIndex(f"file{n}", n, 100 * np.arange(size), num_reads=0) for n, size in enumerate(file_sizes)
    ]

    def segment(file: int, start: int, end: int) -> PlainTextSegment:
        return PlainTextSegment(f"file{file}", file, 100 * start, end - start)

    # 9510 data in ceil(9510 / 3000) = 4 chunks of 9510 // 4 = 2377, the last of which also takes the rest
    assert plain_text_chunks(file_indices, seed=5) == [
        PlainTextChunk((segment(0, 0, 2377),), 5),
        PlainTextChunk((segment(0, 2377, 2500), segment(1, 0, 10), segment(2, 0, 2244)), 6),
        PlainTextChunk((segment(2, 2244, 4621),), 7),
        PlainTextChunk((segment(2, 4621, 7000),), 8),
    ]


def test_preprocessing_does_not_depend_on_the_number_of_processes(monkeypatch):
    # nor does it change from run to run, nor touch the global random number generators.  Two copies of a dataset in
    # chunks of 3000 data put a chunk across the end of one file and the start of the next.
    monkeypatch.setattr(plain_text_data, "NUM_RAW_DATA_TO_NORMALIZE_AT_ONCE", 3000)
    torch_state, numpy_state = torch.get_rng_state(), np.random.get_state()
    dataset_files = [SMALL_PLAIN_TEXT_DATA, SMALL_PLAIN_TEXT_DATA]
    runs = [plain_text_data.make_normalized_mmap_data(dataset_files, [0, 1], num_processes=n) for n in (1, 1, 3)]
    assert torch.equal(torch.get_rng_state(), torch_state)
    assert np.array_equal(np.random.get_state()[1], numpy_state[1])
    for run in runs[1:]:
        for name in ("int_mmap", "float_mmap", "reads_mmap"):
            assert getattr(run, name).tobytes() == getattr(runs[0], name).tobytes(), name


def test_uint8_encoding():
    # x -> 32x + 128, so bytes below 128 hold negative values, decoded here from a 2D block of read columns, as a Batch
    # decodes them
    stored = np.array([[0, 100, 127], [128, 200, 255]], dtype=np.uint8)
    assert np.array_equal(convert_uint8_to_quantile_normalized(stored), [[-4, -0.875, -0.03125], [0, 2.25, 3.96875]])

    # values in the representable range [-4, 3.96875] come back within one quantization step of 1/32, and values outside
    # it are clipped to its ends
    values = np.linspace(-4, 3.96875, 1001)
    round_trip = convert_uint8_to_quantile_normalized(convert_quantile_normalized_to_uint8(values))
    assert np.all(np.abs(round_trip - values) <= 1 / 32)
    clipped = convert_uint8_to_quantile_normalized(convert_quantile_normalized_to_uint8(np.array([-10.0, 10.0])))
    assert np.array_equal(clipped, [-4, 3.96875])
