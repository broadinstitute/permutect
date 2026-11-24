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

    # every element of a tensor, in order, once each
    shape = (2, 3, 4)
    grids = torch.meshgrid(*[torch.arange(n) for n in shape], indexing="ij")
    assert torch.equal(flattened_indices(shape, tuple(g.reshape(-1) for g in grids)), torch.arange(2 * 3 * 4))

    # one dimension, size-1 dimensions and empty index tensors
    idx = LongTensor([0, 3, 1])
    assert torch.equal(flattened_indices((5,), (idx,)), idx)
    assert flattened_indices((1, 4, 1), (0, 3, 0)) == 3
    assert flattened_indices((3, 4), (LongTensor([]), LongTensor([]))).numel() == 0


def test_index_tensor():
    x = torch.arange(60).reshape(3, 4, 5).float()
    idx = (LongTensor([0, 2, 1, 2]), LongTensor([3, 0, 1, 3]), LongTensor([4, 0, 2, 4]))
    assert torch.equal(index_tensor(x, idx), x[idx])

    # one dimension
    assert torch.equal(index_tensor(Tensor([10, 20, 30]), (LongTensor([2, 0]),)), Tensor([30, 10]))

    # repeated indices, empty indices, and 2D index tensors, which give a result of their shape
    x = torch.arange(12).reshape(3, 4)
    assert torch.equal(index_tensor(x, (LongTensor([1, 1, 1]), LongTensor([2, 2, 2]))), LongTensor([6, 6, 6]))
    assert index_tensor(x, (LongTensor([]), LongTensor([]))).numel() == 0
    rows, cols = LongTensor([[0, 1], [2, 0]]), LongTensor([[3, 0], [1, 1]])
    assert torch.equal(index_tensor(x, (rows, cols)), x[rows, cols])

    # non-contiguous tensors, eg a transpose
    t = torch.arange(12).reshape(3, 4).t()
    assert torch.equal(index_tensor(t, (LongTensor([0, 3]), LongTensor([1, 2]))), t[[0, 3], [1, 2]])


def test_add_at_index():
    # in place, adding to the existing values, and unlike x[idx] += values, adding every value of a repeated index
    x = torch.ones(2, 3, 4)
    idx = (LongTensor([0, 1, 1, 1]), LongTensor([2, 0, 2, 2]), LongTensor([3, 1, 0, 0]))
    assert add_at_index(x, idx, Tensor([5, -2, 0.5, 3])) is x
    expected = torch.ones(2, 3, 4)
    expected[0, 2, 3], expected[1, 0, 1], expected[1, 2, 0] = 6, -1, 4.5
    assert torch.equal(x, expected)

    # one dimension, of integers
    x = torch.zeros(4, dtype=torch.int32)
    add_at_index(x, (LongTensor([3, 0, 3]),), IntTensor([1, 2, 3]))
    assert torch.equal(x, IntTensor([2, 0, 0, 4]))

    # empty indices leave the tensor unchanged
    x = torch.ones(2, 2)
    add_at_index(x, (LongTensor([]), LongTensor([])), Tensor([]))
    assert torch.equal(x, torch.ones(2, 2))

    # non-contiguous tensors, eg a transpose, are modified in place, with repeated indices accumulated
    x = torch.zeros(3, 4).t()
    add_at_index(x, (LongTensor([0, 3, 3]), LongTensor([1, 2, 2])), Tensor([1, 2, 3]))
    assert x[0, 1] == 1 and x[3, 2] == 5 and torch.sum(x) == 6


def test_select_and_sum():
    # the examples from the docstring
    x = Tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    assert torch.equal(select_and_sum(x, select={0: 0}, sum=(2,)), Tensor([3, 7]))
    assert torch.equal(select_and_sum(x, select={0: 1, 1: 0}), Tensor([5, 6]))
    assert torch.equal(select_and_sum(x, sum=(1, 2)), Tensor([10, 26]))

    # in any order of dimensions in select and sum
    x = torch.arange(2 * 3 * 4 * 5).reshape(2, 3, 4, 5).float()
    assert torch.equal(select_and_sum(x, select={1: 2}, sum=(3,)), x[:, 2, :, :].sum(dim=2))
    assert torch.equal(select_and_sum(x, select={2: 3, 0: 1}, sum=(1,)), x[1, :, 3, :].sum(dim=0))
    assert torch.equal(select_and_sum(x, select={3: 4}, sum=(2, 0)), x[:, :, :, 4].sum(dim=(0, 2)))

    # selecting and summing nothing returns the input, and selecting or summing every dimension gives a scalar
    x = torch.arange(24).reshape(2, 3, 4)
    assert torch.equal(select_and_sum(x), x)
    assert (
        select_and_sum(x, select={0: 1, 1: 2, 2: 3}).shape == () and select_and_sum(x, select={0: 1, 1: 2, 2: 3}) == 23
    )
    assert select_and_sum(x, sum=(0, 1, 2)).shape == () and select_and_sum(x, sum=(0, 1, 2)) == x.sum()
    assert select_and_sum(x, select={1: 0}, sum=(0, 2)) == x[:, 0, :].sum()

    # negative dimensions and indices count from the end
    assert torch.equal(select_and_sum(x, sum=(-1,)), x.sum(dim=-1))
    assert torch.equal(select_and_sum(x, select={-1: 2}), x[:, :, 2])
    assert torch.equal(select_and_sum(x, select={2: -1}, sum=(1,)), x[:, :, 3].sum(dim=1))

    # size-1 dimensions
    y = torch.arange(6).reshape(1, 6, 1)
    assert torch.equal(select_and_sum(y, select={0: 0, 2: 0}), torch.arange(6))
    assert torch.equal(select_and_sum(y, sum=(0, 2)), torch.arange(6))

    # the input is not modified
    x_copy = x.clone()
    select_and_sum(x, select={0: 0}, sum=(1,))
    assert torch.equal(x, x_copy)
