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
    """
    assert len(shape) == len(idx)
    dim = len(shape)

    result = idx[0]
    for i in range(1, dim):
        result = result * shape[i] + idx[i]
    return result


def index_tensor(tens: Tensor, idx: Tuple[IntTensor, ...]) -> Tensor:
    return tens.view(-1)[flattened_indices(tens.shape, idx)]


# add in-place
# note that the flattened view(-1) shares memory with the original tensor
def add_at_index(tens: Tensor, idx: Tuple[IntTensor, ...], values: Tensor) -> Tensor:
    return tens.view(-1).index_add_(dim=0, index=flattened_indices(tens.shape, idx), source=values)


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
