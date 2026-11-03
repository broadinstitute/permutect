import io
from collections import Counter

import numpy as np
import pytest

from permutect.data.datum import FLOAT_DTYPE
from permutect.data.plain_text_data import MAX_VALUE
from permutect.data.plain_text_data import convert_quantile_normalized_to_uint8
from permutect.data.plain_text_data import convert_uint8_to_quantile_normalized
from permutect.data.plain_text_data import line_to_array
from permutect.data.plain_text_data import read_2d_array
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
