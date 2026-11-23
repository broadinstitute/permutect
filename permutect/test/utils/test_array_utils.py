import numpy as np
import torch
from torch import IntTensor
from torch import LongTensor
from torch import Tensor

from permutect.utils.array_utils import add_at_index
from permutect.utils.array_utils import flattened_indices
from permutect.utils.array_utils import index_tensor
from permutect.utils.array_utils import select_and_sum


def test_flattened_indices():
    # the example from the docstring: 1 * (4 * 5) + 2 * 5 + 3 = 33
    assert flattened_indices((3, 4, 5), (1, 2, 3)) == 33

    shape = (3, 4, 5)
    idx = (LongTensor([0, 1, 2, 2]), LongTensor([0, 2, 3, 0]), LongTensor([0, 3, 4, 1]))
    expected = LongTensor(np.ravel_multi_index(tuple(i.numpy() for i in idx), shape))
    assert torch.equal(flattened_indices(shape, idx), expected)


def test_flattened_indices_edge_cases():
    # one dimension: the flattened index is the index itself
    idx = LongTensor([0, 3, 1])
    assert torch.equal(flattened_indices((5,), (idx,)), idx)

    # first and last elements map to 0 and numel - 1
    shape = (2, 3, 4)
    assert flattened_indices(shape, (0, 0, 0)) == 0
    assert flattened_indices(shape, (1, 2, 3)) == 2 * 3 * 4 - 1

    # size-1 dimensions contribute nothing
    assert flattened_indices((1, 4, 1), (0, 3, 0)) == 3

    # empty index tensors give an empty result
    assert flattened_indices((3, 4), (LongTensor([]), LongTensor([]))).numel() == 0


def test_flattened_indices_covers_every_element_once():
    shape = (2, 3, 4)
    grids = torch.meshgrid(*[torch.arange(n) for n in shape], indexing="ij")
    flat = flattened_indices(shape, tuple(g.reshape(-1) for g in grids))
    assert torch.equal(flat, torch.arange(2 * 3 * 4))


def test_index_tensor():
    x = torch.arange(60).reshape(3, 4, 5).float()
    idx = (LongTensor([0, 2, 1, 2]), LongTensor([3, 0, 1, 3]), LongTensor([4, 0, 2, 4]))
    assert torch.equal(index_tensor(x, idx), x[idx])


def test_index_tensor_edge_cases():
    # one dimension
    x = Tensor([10, 20, 30])
    assert torch.equal(index_tensor(x, (LongTensor([2, 0]),)), Tensor([30, 10]))

    # repeated indices return repeated values
    x = torch.arange(12).reshape(3, 4)
    assert torch.equal(index_tensor(x, (LongTensor([1, 1, 1]), LongTensor([2, 2, 2]))), LongTensor([6, 6, 6]))

    # empty indices give an empty result
    assert index_tensor(x, (LongTensor([]), LongTensor([]))).numel() == 0

    # 2D index tensors give a result of the same shape
    rows = LongTensor([[0, 1], [2, 0]])
    cols = LongTensor([[3, 0], [1, 1]])
    assert torch.equal(index_tensor(x, (rows, cols)), x[rows, cols])

    # non-contiguous tensors, eg a transpose
    t = torch.arange(12).reshape(3, 4).t()
    assert torch.equal(index_tensor(t, (LongTensor([0, 3]), LongTensor([1, 2]))), t[[0, 3], [1, 2]])


def test_add_to_array():
    x = torch.zeros(4, 5).int()
    add_at_index(x, (LongTensor([0, 2]), LongTensor([1, 3])), IntTensor([7, 8]))
    assert x[0, 1] == 7
    assert x[2, 3] == 8
    assert torch.sum(x) == 15


def test_add_at_index_accumulates_duplicate_indices():
    # unlike x[idx] += values, repeated indices must all be added
    x = torch.zeros(3, 3)
    add_at_index(x, (LongTensor([1, 1, 1, 0]), LongTensor([2, 2, 2, 0])), Tensor([1, 2, 3, 4]))
    assert x[1, 2] == 6
    assert x[0, 0] == 4
    assert torch.sum(x) == 10


def test_add_at_index_adds_to_existing_values():
    x = torch.ones(2, 3, 4)
    expected = x.clone()
    idx = (LongTensor([0, 1, 1]), LongTensor([2, 0, 2]), LongTensor([3, 1, 0]))
    values = Tensor([5, -2, 0.5])
    add_at_index(x, idx, values)
    expected[idx] += values
    assert torch.equal(x, expected)


def test_add_at_index_edge_cases():
    # one dimension
    x = torch.zeros(4)
    add_at_index(x, (LongTensor([3, 0, 3]),), Tensor([1, 2, 3]))
    assert torch.equal(x, Tensor([2, 0, 0, 4]))

    # empty indices leave the tensor unchanged
    x = torch.ones(2, 2)
    add_at_index(x, (LongTensor([]), LongTensor([])), Tensor([]))
    assert torch.equal(x, torch.ones(2, 2))

    # the input tensor is returned, modified in place
    x = torch.zeros(2, 3)
    assert add_at_index(x, (LongTensor([1]), LongTensor([2])), Tensor([9])) is x
    assert x[1, 2] == 9

    # non-contiguous tensors, eg a transpose, are modified in place, with repeated indices accumulated
    x = torch.zeros(3, 4).t()
    add_at_index(x, (LongTensor([0, 3, 3]), LongTensor([1, 2, 2])), Tensor([1, 2, 3]))
    assert x[0, 1] == 1 and x[3, 2] == 5 and torch.sum(x) == 6


def test_select_and_sum():
    x = Tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    assert torch.equal(select_and_sum(x, select={0: 0}, sum=(2,)), Tensor([3, 7]))
    assert torch.equal(select_and_sum(x, select={0: 1, 1: 0}), Tensor([5, 6]))
    assert torch.equal(select_and_sum(x, sum=(1, 2)), Tensor([10, 26]))


def test_select_and_sum_matches_direct_computation():
    x = torch.arange(2 * 3 * 4 * 5).reshape(2, 3, 4, 5).float()
    assert torch.equal(select_and_sum(x, select={1: 2}, sum=(3,)), x[:, 2, :, :].sum(dim=2))
    assert torch.equal(select_and_sum(x, select={0: 1, 2: 3}, sum=(1,)), x[1, :, 3, :].sum(dim=0))
    assert torch.equal(select_and_sum(x, select={3: 4}, sum=(0, 2)), x[:, :, :, 4].sum(dim=(0, 2)))

    # the order of dimensions in select and sum doesn't matter
    assert torch.equal(select_and_sum(x, select={2: 3, 0: 1}, sum=(1,)), x[1, :, 3, :].sum(dim=0))
    assert torch.equal(select_and_sum(x, sum=(2, 0)), x.sum(dim=(0, 2)))


def test_select_and_sum_edge_cases():
    x = torch.arange(24).reshape(2, 3, 4)

    # selecting and summing nothing returns the input unchanged
    assert torch.equal(select_and_sum(x), x)

    # selecting every dimension gives a scalar
    result = select_and_sum(x, select={0: 1, 1: 2, 2: 3})
    assert result.shape == ()
    assert result == x[1, 2, 3]

    # summing every dimension gives a scalar
    result = select_and_sum(x, sum=(0, 1, 2))
    assert result.shape == ()
    assert result == x.sum()

    # selecting some dimensions and summing the rest gives a scalar
    result = select_and_sum(x, select={1: 0}, sum=(0, 2))
    assert result.shape == ()
    assert result == x[:, 0, :].sum()

    # selecting the last index of each dimension
    assert torch.equal(select_and_sum(x, select={0: 1, 2: 3}), x[1, :, 3])

    # negative dimensions
    assert torch.equal(select_and_sum(x, sum=(-1,)), x.sum(dim=-1))
    assert torch.equal(select_and_sum(x, select={-1: 2}), x[:, :, 2])

    # negative indices count from the end
    assert torch.equal(select_and_sum(x, select={0: -1}), x[1])
    assert torch.equal(select_and_sum(x, select={2: -1}, sum=(1,)), x[:, :, 3].sum(dim=1))

    # size-1 dimensions
    y = torch.arange(6).reshape(1, 6, 1)
    assert torch.equal(select_and_sum(y, select={0: 0, 2: 0}), torch.arange(6))
    assert torch.equal(select_and_sum(y, sum=(0, 2)), torch.arange(6))

    # the input is not modified
    x_copy = x.clone()
    select_and_sum(x, select={0: 0}, sum=(1,))
    assert torch.equal(x, x_copy)
