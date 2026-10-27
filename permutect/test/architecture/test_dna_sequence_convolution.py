import torch
from torch import nn

from permutect.architecture.dna_sequence_convolution import INITIAL_NUM_CHANNELS
from permutect.architecture.dna_sequence_convolution import DNASequenceConvolution
from permutect.architecture.dna_sequence_convolution import conv_output_length
from permutect.architecture.dna_sequence_convolution import pool_output_length


def test_constructor():
    input_length = 20
    layer_strings = [
        "convolution/kernel_size=3/out_channels=64",
        "pool/kernel_size=2",
        "leaky_relu",
        "convolution/kernel_size=3/dilation=2/out_channels=5",
        "leaky_relu",
        "flatten",
        "linear/out_features=10",
    ]
    model = DNASequenceConvolution(layer_strings, input_length)

    data = torch.randn(8, INITIAL_NUM_CHANNELS, input_length)
    model(data)


def test_constructor_with_padding():
    # layer string keys are passed to nn.Conv1d and nn.MaxPool1d, which call it 'padding'
    input_length = 20
    layer_strings = [
        "convolution/kernel_size=3/padding=1/out_channels=4",
        "pool/kernel_size=2/padding=1",
        "flatten",
        "linear/out_features=10",
    ]
    model = DNASequenceConvolution(layer_strings, input_length)

    data = torch.randn(8, INITIAL_NUM_CHANNELS, input_length)
    assert model(data).shape == (8, model.output_dimension())


def test_output_lengths_match_torch():
    input_length = 25
    data = torch.randn(1, 1, input_length)
    for kwargs in (
        {"kernel_size": 3},
        {"kernel_size": 3, "padding": 1},
        {"kernel_size": 5, "stride": 2, "padding": 2},
        {"kernel_size": 3, "dilation": 2, "padding": 1},
    ):
        conv = nn.Conv1d(in_channels=1, out_channels=1, **kwargs)
        assert conv_output_length(input_length, **kwargs) == conv(data).shape[-1], kwargs

    for kwargs in ({"kernel_size": 2}, {"kernel_size": 2, "padding": 1}, {"kernel_size": 4, "stride": 3, "padding": 2}):
        pool = nn.MaxPool1d(**kwargs)
        assert pool_output_length(input_length, **kwargs) == pool(data).shape[-1], kwargs
