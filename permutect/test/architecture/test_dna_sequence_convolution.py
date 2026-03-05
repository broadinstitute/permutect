import torch

from permutect.architecture.dna_sequence_convolution import INITIAL_NUM_CHANNELS
from permutect.architecture.dna_sequence_convolution import DNASequenceConvolution


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
