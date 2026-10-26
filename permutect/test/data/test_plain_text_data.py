import numpy as np

from permutect.data.plain_text_data import convert_quantile_normalized_to_uint8
from permutect.data.plain_text_data import convert_uint8_to_quantile_normalized


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
