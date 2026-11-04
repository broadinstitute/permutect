import io
from collections import Counter

import numpy as np
import pytest
import torch

from permutect.data import memory_mapped_data
from permutect.data import plain_text_data
from permutect.data.datum import FLOAT_DTYPE
from permutect.data.datum import Data
from permutect.data.datum import Datum
from permutect.data.memory_mapped_data import stacked_read_offsets
from permutect.data.plain_text_data import MAX_VALUE
from permutect.data.plain_text_data import PlainTextChunk
from permutect.data.plain_text_data import PlainTextFileIndex
from permutect.data.plain_text_data import PlainTextSegment
from permutect.data.plain_text_data import alt_read_medians_and_means
from permutect.data.plain_text_data import convert_quantile_normalized_to_uint8
from permutect.data.plain_text_data import convert_uint8_to_quantile_normalized
from permutect.data.plain_text_data import line_to_array
from permutect.data.plain_text_data import plain_text_chunks
from permutect.data.plain_text_data import read_2d_array
from permutect.data.plain_text_data import read_raw_unnormalized_data
from permutect.test.test_file_names import SMALL_PLAIN_TEXT_DATA


def parse_one_number_at_a_time(line: str) -> np.ndarray:
    # how lines were parsed before numpy parsed them all at once
    return np.array([max(min(MAX_VALUE, float(token)), -MAX_VALUE) for token in line.split()], dtype=FLOAT_DTYPE)


def numeric_lines(text_file) -> list[str]:
    lines = []
    with open(text_file) as file:
        for line in file:
            try:
                parse_one_number_at_a_time(line)
                lines.append(line)
            except ValueError:
                pass
    return lines


def test_numbers_are_parsed_as_before():
    # every numeric line of a dataset, each alone and in blocks of consecutive lines with as many numbers
    lines = numeric_lines(SMALL_PLAIN_TEXT_DATA) + ["1.5 nan inf -inf 20000 -20000\n"]
    for line in lines:
        assert line_to_array(line).tobytes() == parse_one_number_at_a_time(line).tobytes()

    # the most common number of numbers in a line is that of the read lines
    reads_width = Counter(len(line.split()) for line in lines).most_common(1)[0][0]
    reads_lines = [line for line in lines if len(line.split()) == reads_width]
    assert reads_width > 5 and len(reads_lines) > 1000
    for start in range(0, len(reads_lines), 7):
        block = reads_lines[start : start + 7]
        expected = np.vstack([parse_one_number_at_a_time(line) for line in block])
        assert read_2d_array(io.StringIO("".join(block)), len(block)).tobytes() == expected.tobytes()


@pytest.mark.parametrize("text", ["1 2 3\n4 abc 6\n", "1 2 3\n4 5\n"])
def test_malformed_lines_are_rejected(text):
    with pytest.raises(ValueError):
        read_2d_array(io.StringIO(text), 2)


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


def test_preprocessing_writes_only_the_normalized_memory_maps(monkeypatch):
    # the raw data used to be written to memory maps of their own before they were normalized
    created_memory_maps = []
    original_grown_memory_map = memory_mapped_data.grown_memory_map

    def counting_grown_memory_map(*args, **kwargs):
        created_memory_maps.append(kwargs.get("suffix", args[-1]))
        return original_grown_memory_map(*args, **kwargs)

    monkeypatch.setattr(memory_mapped_data, "grown_memory_map", counting_grown_memory_map)
    plain_text_data.make_normalized_mmap_data([SMALL_PLAIN_TEXT_DATA])
    assert len(created_memory_maps) == 3, created_memory_maps


def test_file_index_finds_each_datum_with_alt_reads():
    # its segments read the same data as reading the file from its start
    file_index = PlainTextFileIndex.of(SMALL_PLAIN_TEXT_DATA, source=0)
    positions = [int(datum.get(Data.POSITION)) for datum in read_raw_unnormalized_data(SMALL_PLAIN_TEXT_DATA)]
    assert len(file_index) == len(positions)
    for start, end in ((0, 1), (10, 25), (len(positions) - 3, len(positions))):
        segment_data = file_index.segment(start, end).read(torch.Generator())
        assert [int(datum.get(Data.POSITION)) for datum in segment_data] == positions[start:end]


def test_chunks_have_the_same_boundaries_as_before(monkeypatch):
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


def test_downsampling_draws_from_its_generator():
    datum = Datum(
        np.zeros(Data.NUM_SCALAR_INT_ELEMENTS),
        np.zeros(Data.NUM_SCALAR_FLOAT_ELEMENTS),
        reads_re=np.arange(150, dtype=FLOAT_DTYPE).reshape(50, 3),
    )
    datum.set(Data.REF_COUNT, 30)
    datum.set(Data.ALT_COUNT, 20)
    torch_state = torch.get_rng_state()
    draws = [datum.copy_with_downsampled_reads(10, 5, torch.Generator().manual_seed(7)).reads_re for _ in range(2)]
    assert np.array_equal(draws[0], draws[1]) and draws[0].shape == (15, 3)
    assert torch.equal(torch.get_rng_state(), torch_state)


def test_uint8_decoding():
    # bytes below 128 hold negative values
    stored = np.array([0, 100, 127, 128, 200, 255], dtype=np.uint8)
    decoded = convert_uint8_to_quantile_normalized(stored)
    assert np.array_equal(decoded, [-4, -0.875, -0.03125, 0, 2.25, 3.96875])


def test_uint8_decoding_is_increasing():
    decoded = convert_uint8_to_quantile_normalized(np.arange(256, dtype=np.uint8))
    assert np.all(np.diff(decoded.astype(np.float64)) > 0)


def test_uint8_decoding_of_2d_read_columns():
    # Batch decodes a 2D block of read columns at once
    stored = np.array([[0, 255], [128, 64]], dtype=np.uint8)
    decoded = convert_uint8_to_quantile_normalized(stored)
    assert np.array_equal(decoded, [[-4, 3.96875], [0, -2]])


def test_uint8_round_trip():
    # values in the representable range [-4, 3.96875] come back within one quantization step of 1/32
    values = np.linspace(-4, 3.96875, 1001)
    round_trip = convert_uint8_to_quantile_normalized(convert_quantile_normalized_to_uint8(values))
    assert np.all(np.abs(round_trip - values) <= 1 / 32)

    # values outside that range are clipped to its ends
    clipped = convert_uint8_to_quantile_normalized(convert_quantile_normalized_to_uint8(np.array([-10.0, 10.0])))
    assert np.array_equal(clipped, [-4, 3.96875])
