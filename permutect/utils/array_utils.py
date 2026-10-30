from typing import Tuple

import numpy as np
import torch
from torch import IntTensor
from torch import Tensor


def flattened_indices(shape: Tuple[int, ...], idx: Tuple[IntTensor, ...]):
    """
    Example: for shape = (3, 4, 5) and idx = (1, 2, 3) the flattened index is
        1 * (4 * 5) + 2 * (5) + 3
        = [(1) * 4 + 2] * 5 + 3

    In general, suppose that indices 0, 1. . . D-1 yield a flattened index F_{D-1}.  If the D + 1 dimension has size
    S_D and index I_D then the flattened index F_D is F_{D-1} * S_D + I_D, because every combination of the indices
    0,1. . .D-1 can be associated with any of S_D values of index D.  Thus flattened indices can be computed with
    a quick recursion.

    This is equivalent to numpy's ravel_multi_index (torch has no equivalent), but don't replace it with that!  Using
    ravel_multi_index on torch tensors requires a round trip to numpy, which is much slower for large batches:
    2-3x slower on CPU at ~100K elements, and 15-40x slower on GPU (measured on MPS) at all batch sizes, because
    of the device -> host -> device copy and synchronization.  Two other differences: unlike ravel_multi_index,
    this does no bounds checking, and shape[0] is never used so it may be None.
    """
    assert len(shape) == len(idx)
    dim = len(shape)

    result = idx[0]
    for i in range(1, dim):
        result = result * shape[i] + idx[i]
    return result


def index_tensor(tens: Tensor, idx: Tuple[IntTensor, ...]) -> Tensor:
    # reshape is a view of a contiguous tensor, and copies a non-contiguous one, eg a transpose
    return tens.reshape(-1)[flattened_indices(tens.shape, idx)]


# add in-place, and return the input tensor
def add_at_index(tens: Tensor, idx: Tuple[IntTensor, ...], values: Tensor) -> Tensor:
    if tens.is_contiguous():
        # index_add_ on the flattened view(-1), which shares memory with the original tensor
        tens.view(-1).index_add_(dim=0, index=flattened_indices(tens.shape, idx), source=values)
    else:
        # a non-contiguous tensor, eg a transpose, has no flattened view
        tens.index_put_(idx, values, accumulate=True)
    return tens


def downsample_tensor(tensor2d: np.ndarray, new_length: int):
    if tensor2d is None or new_length >= len(tensor2d):
        return tensor2d
    perm = np.random.permutation(len(tensor2d))
    return tensor2d[perm[:new_length]]


def select_and_sum(x: Tensor, select: dict[int, int] = {}, sum: Tuple[int, ...] = ()):
    """
    select specific indices over certain dimensions and sum over others.  For example suppose
    x = [ [[1,2], [3,4]],
            [[5,6], [7,8]]]
    Then we want:
        select_and_sum(x, select={0:0}, sum=(2)) = [3, 7]
        select_and_sum(x, select={0:1,1:0}, sum=()) = [5,6]
        select_and_sum(x, select={}, sum=(1,2)) = [10,26]
    :param x: the input tensor
    :param select: dict of dimension:index of that dimension to select
    :param sum tuple of dimensions over which to sum
    :return:

    select_indices and sum_dims must be disjoint but need not include all input dimensions.
    """

    # initialize indexing to be complete slices i.e. select everything, then use the given select_indices
    indices = [slice(dim_size) for dim_size in x.shape]
    for select_dim, select_index in select.items():
        # a negative index counts from the end, but slice(-1, 0) would be empty
        if select_index < 0:
            select_index += x.shape[select_dim]
        indices[select_dim] = slice(select_index, select_index + 1)  # one-element slice

    selected = x[tuple(indices)]  # retains original dimensions; selected dimensions have length 1
    summed = (
        selected if len(sum) == 0 else torch.sum(selected, dim=sum, keepdim=True)
    )  # still retain original dimension

    # Finally, select element 0 from the selected and summed axes to contract the dimensions
    for sum_dim in sum:
        indices[sum_dim] = 0
    for select_dim in select.keys():
        indices[select_dim] = 0

    return summed[tuple(indices)]
