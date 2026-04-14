from __future__ import annotations

from typing import Iterable

# in Python 3.11 this would be from typing import Self
from typing import TypeVar

import torch
from torch import IntTensor
from torch import LongTensor
from torch import Tensor

ThisClass = TypeVar("ThisClass", bound="RaggedSets")


class RaggedSets:
    """
    Class for batch of ragged sets.  Sets in the batch can have different sizes; elements are vectors of fixed dimension.

    Conceptually the data are an array X_bsf, where b indexes the set within the batch, s indexes the element within a set,
    and f is the feature index.  b ranges from 0 to B-1, f from 0 to F-1, but the range of s is different for each b.

    Because Pytorch's support for ragged tensors (the nested package) is incomplete and not yet stable, this is my own implementation.

    The data in this class are stored as i) partially-flattened tensor X_nf, where n indexes both b and s in the natural order,
    and ii) an LongTensor of set bounds, which should start at 0, end at N, and have size B+1.

    The bth set starts with the bounds[b]-th flattened element (inclusive) and ends with the bounds[b+1]-th flattened element
    (exclusive).  bounds must be in non-decreasing order.

    Example: sets of sizes 1,3,2; bounds = [0, 1, 4, 6]
    """

    def __init__(self, flattened_tensor_nf: Tensor, lengths_b: IntTensor | LongTensor):
        self.flattened_tensor_nf = flattened_tensor_nf
        assert lengths_b.dim() == 1
        assert torch.sum(lengths_b) == len(flattened_tensor_nf)
        self.lengths_b = lengths_b

    def batch_size(self) -> int:
        return len(self.lengths_b)

    def expand_from_b_to_n(self, tensor_bf: Tensor) -> Tensor:
        """
        given input tensor tensor_bf with values for each set in the batch, expand by repeating so that
        output_nf has values for each element.  The b-th set gets repeated size[b] times
        :param tensor_bf:
        :return:
        """
        return torch.repeat_interleave(tensor_bf, dim=0, repeats=self.lengths_b)

    def apply_elementwise(self, func) -> ThisClass:
        """
        conceptually, transform each vector element X_bs to func(X_bs).  In practice, transform each vector element
        X_n to func(X_n).

        :param func: the operation to be applied elementwise.  func must be designed to act on 2D batches of elements X_bf.
        For example, a Pytorch linear layer or my MLP class work in this way.
        :return: the elementwise-transformed RaggedSets
        """
        return RaggedSets(func(self.flattened_tensor_nf), self.lengths_b)

    # override the * operator for elementwise multiplication
    # works for numeric scalars and torch Tensors of compatible shape
    def __mul__(self, other) -> ThisClass:
        return RaggedSets(self.flattened_tensor_nf * other, self.lengths_b)

    def __rmul__(self, other) -> ThisClass:
        return self.__mul__(other)

    # override the + operator for elementwise addition
    # works for numeric scalars and torch Tensors of compatible shape
    def __add__(self, other) -> ThisClass:
        return RaggedSets(self.flattened_tensor_nf + other, self.lengths_b)

    # override the - operator for elementwise addition
    # works for numeric scalars and torch Tensors of compatible shape
    def __sub__(self, other) -> ThisClass:
        return RaggedSets(self.flattened_tensor_nf - other, self.lengths_b)

    def __radd__(self, other) -> ThisClass:
        return self.__add__(other)

    def multiply_elementwise(self, other: ThisClass) -> ThisClass:
        """
        elementwise multiplication of two RaggedSets.  They need to have the same sizes, which we don't check for.

        Implementation is trivial since X_bsf * Y_bsf is equivalent to X_nf * Y_nf
        """
        return RaggedSets(self.flattened_tensor_nf * other.flattened_tensor_nf, self.lengths_b)

    def add_elementwise(self, other: ThisClass) -> ThisClass:
        """
        elementwise addition of two RaggedSets.  They need to have the same sizes, which we don't check for.

        Implementation is trivial since X_bsf * Y_bsf is equivalent to X_nf * Y_nf
        """
        return RaggedSets(self.flattened_tensor_nf + other.flattened_tensor_nf, self.lengths_b)

    def broadcast_add(self, other_bf: Tensor) -> ThisClass:
        """
        given a 2D tensor Z_bf with same batch size and feature size, broadcast the addition over batches.  That is,
        result_bsf = X_bsf + other_bf

        Implement in our flattened representation by expanding the other to other_bsf, then adding elementwise
        """
        assert len(other_bf) == self.batch_size()
        assert self.flattened_tensor_nf.shape[-1] == other_bf.shape[-1]
        other_bsf = self.expand_from_b_to_n(other_bf)
        return self + other_bsf

    def chunk_over_features(self, num_chunks) -> Iterable[ThisClass]:
        chunks = torch.chunk(self.flattened_tensor_nf, chunks=num_chunks, dim=-1)
        return (RaggedSets(chunk, self.lengths_b) for chunk in chunks)

    def split_in_two_by_features(self) -> tuple[ThisClass, ThisClass]:
        return self.chunk_over_features(num_chunks=2)

    def softmax_within_sets(self) -> ThisClass:
        """
        take the softmax over the elements of each set, independently.

        Conceptually this is just softmax(X_bsf, dim=-2).  However, this is impossible with our flattened representation,
        so instead we
        i) take the featurewise max within each set
        ii) expand the maxes (repeat the bth maximum size[b] times)
        iii) subtract the set maxima for numerical stability
        iv) exponentiate
        v) sum over the exponentiated values within each set
        vi) expand
        vii) divide iv) by vi)
        :return: a RaggedSets object with the same shape, but with softmax "normalization" applied
        """
        maxes_bf = torch.segment_reduce(self.flattened_tensor_nf, reduce="max", axis=0, lengths=self.lengths_b)
        maxes_nf = self.expand_from_b_to_n(maxes_bf)

        stable_nf = self.flattened_tensor_nf - maxes_nf
        exp_nf = torch.exp(stable_nf)
        denom_bf = torch.segment_reduce(exp_nf, reduce="sum", lengths=self.lengths_b, axis=0)
        denom_nf = self.expand_from_b_to_n(denom_bf)
        result_nf = exp_nf / denom_nf
        return RaggedSets(result_nf, self.lengths_b)

    def means_over_sets(self, regularizer_f: Tensor = None, regularizer_weight: float = 0.0001) -> Tensor:
        """
        mean element of each set, with a regularizer to handle sets with zero or few elements.  The very small default
        regularizer weight means that the regularizer acts as an imputed value for empty sets and has basically no
        effect otherwise.
        """
        sums_bf = self.sums_over_sets()
        reg_bf = 0 if regularizer_f is None else (regularizer_weight * regularizer_f).view(1, -1)
        regularized_sums_bf = sums_bf + reg_bf
        regularized_sizes_b = self.lengths_b + regularizer_weight
        means_bf = regularized_sums_bf / regularized_sizes_b.view(-1, 1)
        return means_bf

    def sums_over_sets(self) -> Tensor:
        return torch.segment_reduce(self.flattened_tensor_nf, lengths=self.lengths_b, reduce="sum", axis=0)
