import math

import pytest
import torch

from permutect.segmentation.indel_parameters import DINUCLEOTIDE
from permutect.segmentation.indel_parameters import HOMOPOLYMER
from permutect.segmentation.indel_parameters import LONGER_UNIT
from permutect.segmentation.indel_parameters import NOT_IN_STR
from permutect.segmentation.indel_parameters import NUM_REPEAT_CLASSES
from permutect.segmentation.indel_parameters import IndelParameters
from permutect.segmentation.indel_parameters import repeat_classes


def parameters() -> IndelParameters:
    # a constant shortfall of 0.1 outside STRs, a steep one in homopolymers and none in the other classes
    return IndelParameters(
        shortfall_scales=[0.2, 0.5, 0.0, 0.0],
        shortfall_offsets=[0.0, -10.0, 0.0, 0.0],
        shortfall_slopes=[0.0, 1.0, 0.0, 0.0],
        concentration_upper_bounds=[[0], [5, 10, 15], [4], [6, 12]],
        log_concentrations=[
            [math.log(30)],
            [math.log(100), math.log(20), math.log(3)],
            [math.log(50)],
            [math.log(40), math.log(10)],
        ],
    )


def tensor(values) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.float64)


def test_repeat_classes():
    assert (
        repeat_classes(tensor([0, 1, 2, 3, 4, 6])).tolist()
        == [NOT_IN_STR, HOMOPOLYMER, DINUCLEOTIDE] + [LONGER_UNIT] * 3
    )


def test_alt_fraction_factors():
    unit_lengths, lengths = tensor([0, 1, 1, 1, 2, 3]), tensor([0, 5, 10, 20, 30, 40])
    factors = parameters().alt_fraction_factors(unit_lengths, lengths)
    sigmoid = lambda x: 1 / (1 + math.exp(-x))
    expected = [1 - 0.2 * 0.5, 1 - 0.5 * sigmoid(-5), 1 - 0.5 * sigmoid(0), 1 - 0.5 * sigmoid(10), 1, 1]
    assert factors.tolist() == pytest.approx(expected)


def test_concentrations_are_step_functions_of_length():
    # each bin includes its upper bound, and the last bin's concentration holds beyond it
    unit_lengths = tensor([0, 1, 1, 1, 1, 1, 1, 2, 3, 3, 3])
    lengths = tensor([0, 1, 5, 6, 10, 15, 40, 20, 6, 7, 100])
    concentrations = parameters().concentrations(unit_lengths, lengths)
    assert concentrations.tolist() == pytest.approx([30, 100, 100, 20, 20, 3, 3, 50, 40, 10, 10])


def test_concentrations_mustnt_increase_with_length():
    with pytest.raises(AssertionError):
        IndelParameters([0.0] * 4, [0.0] * 4, [0.0] * 4, [[0], [5, 10], [4], [6]], [[1.0], [2.0, 3.0], [1.0], [1.0]])


def test_shortfall_scales_are_below_1():
    with pytest.raises(AssertionError):
        IndelParameters([0.0, 1.0, 0.0, 0.0], [0.0] * 4, [0.0] * 4, [[0], [5], [4], [6]], [[1.0]] * 4)


def test_defaults():
    defaults = IndelParameters.default()
    # in each class, the expected alt fraction and the concentration don't grow with STR length
    lengths = tensor(range(0, 60))
    for k in range(NUM_REPEAT_CLASSES):
        unit_lengths = torch.full_like(lengths, float(k))
        factors = defaults.alt_fraction_factors(unit_lengths, lengths)
        concentrations = defaults.concentrations(unit_lengths, lengths)
        assert torch.all((factors > 0) & (factors <= 1))
        assert torch.all(concentrations > 0)
        assert torch.all(torch.diff(concentrations) <= 0)
        if k != NOT_IN_STR:
            assert torch.all(torch.diff(factors) <= 1e-12)
    # long homopolymers' alt fractions fall well short
    assert defaults.alt_fraction_factors(tensor([1]), tensor([20])).item() < 0.7


def test_state_round_trip():
    original = parameters()
    copy = IndelParameters.from_state(original.state())
    unit_lengths, lengths = tensor([0, 1, 1, 2, 3]), tensor([0, 7, 20, 10, 50])
    assert torch.equal(
        copy.alt_fraction_factors(unit_lengths, lengths), original.alt_fraction_factors(unit_lengths, lengths)
    )
    assert torch.equal(copy.concentrations(unit_lengths, lengths), original.concentrations(unit_lengths, lengths))
