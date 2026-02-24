from __future__ import annotations

import copy
from enum import IntEnum
from turtledemo.clock import datum
from typing import List, Tuple

import torch
from torch import IntTensor, FloatTensor, Tensor
import numpy as np

from permutect.data.count_binning import ref_count_bin_name, NUM_REF_COUNT_BINS, alt_count_bin_name, NUM_ALT_COUNT_BINS, \
    logit_bin_name, NUM_LOGIT_BINS, ref_count_bin_indices, alt_count_bin_indices, logit_bin_indices, \
    ref_count_bin_index, alt_count_bin_index
from permutect.data.datum import Datum, Data, uint32_from_two_int16s, INTEGER_DTYPE, FLOAT_DTYPE, LARGE_INTEGER_DTYPE, \
    NUMBER_OF_BYTES_IN_PACKED_READ, COMPRESSED_READS_ARRAY_DTYPE
from permutect.data.plain_text_data import convert_uint8_to_quantile_normalized
from permutect.misc_utils import gpu_if_available
from permutect.utils.array_utils import flattened_indices
from permutect.utils.enums import Label, Variation


class Batch:
    def __init__(self, data: List[Datum]):
        self.int_tensor = torch.from_numpy(np.vstack([d.get_int_array() for d in data])).to(torch.long)
        self.float_tensor = torch.from_numpy(np.vstack([d.get_float_array() for d in data])).to(torch.float)

        ref_arrays = [datum.get_ref_reads_re() for datum in data]
        alt_arrays = [datum.get_alt_reads_re() for datum in data]
        reads_re = np.vstack(ref_arrays + alt_arrays)

        reads_are_compressed = data[0].reads_re.dtype == COMPRESSED_READS_ARRAY_DTYPE

        if reads_are_compressed:
            packed_binary_columns_re = reads_re[:, :NUMBER_OF_BYTES_IN_PACKED_READ]
            compressed_float_columns_re = reads_re[:, NUMBER_OF_BYTES_IN_PACKED_READ:]
            binary_columns_re = np.ndarray.astype(np.unpackbits(packed_binary_columns_re, axis=1), FLOAT_DTYPE)
            float_columns_re = convert_uint8_to_quantile_normalized(compressed_float_columns_re)
            self.reads_re = torch.from_numpy(np.hstack((binary_columns_re, float_columns_re)))
        else:
            self.reads_re = torch.from_numpy(reads_re)

        # assert that the decompression got the expected tensor shape
        assert self.reads_re.shape[1] == data[0].num_read_features()
        self._finish_initializiation_from_arrays()

    def _finish_initializiation_from_arrays(self):
        self._size = len(self.int_tensor)
        self.lazy_batch_indices = None

    def batch_indices(self, use_original_counts: bool = False) -> BatchIndices:
        if self.lazy_batch_indices is not None:
            return self.lazy_batch_indices
        else:
            ref_counts = (self.get(Data.ORIGINAL_DEPTH) - self.get(Data.ORIGINAL_ALT_COUNT)) if use_original_counts \
                else self.get(Data.REF_COUNT)
            alt_counts = self.get(Data.ORIGINAL_ALT_COUNT if use_original_counts else Data.ALT_COUNT)
            self.lazy_batch_indices = BatchIndices(sources=self.get(Data.SOURCE), labels=self.get(Data.LABEL),
                var_types=self.get(Data.VARIANT_TYPE), ref_counts=ref_counts, alt_counts=alt_counts)
            return self.lazy_batch_indices

    def get(self, data_field: Data):
        index = data_field.idx
        if data_field.dtype == INTEGER_DTYPE:
            return self.int_tensor[:, index]
        elif data_field.dtype == LARGE_INTEGER_DTYPE:
            return uint32_from_two_int16s(self.int_tensor[:, index], self.int_tensor[:, index + 1])
        elif data_field.dtype == FLOAT_DTYPE:
            return self.float_tensor[:, index]
        else:
            assert False, "Unsupported data type"

    # convert to the training format of 0.0 / 0.5 / 1.0 for variant / unlabeled / artifact
    # the 0.5 for unlabeled data is reasonable but should never actually be used due to the is_labeled mask
    def get_training_labels(self) -> FloatTensor:
        int_enum_labels = self.get(Data.LABEL)
        return 1.0 * (int_enum_labels == Label.ARTIFACT) + 0.5 * (int_enum_labels == Label.UNLABELED)

    def get_is_labeled_mask(self) -> IntTensor:
        int_enum_labels = self.get(Data.LABEL)
        return (int_enum_labels != Label.UNLABELED).int()

    def get_info_be(self) -> Tensor:
        return self.float_tensor[:, Data.INFO_START_IDX:]

    def get_haplotypes_bs(self) -> IntTensor:
        # each row is 1D array of integer array reference and alt haplotypes concatenated -- A, C, G, T, deletion = 0, 1, 2, 3, 4
        return self.int_tensor[:, Data.HAPLOTYPES_START_IDX:]

    def get_reads_re(self) -> Tensor:
        return self.reads_re

    # pin memory for all tensors that are sent to the GPU
    def pin_memory(self):
        self.int_tensor = self.int_tensor.pin_memory()
        self.float_tensor = self.float_tensor.pin_memory()
        self.reads_re = self.reads_re.pin_memory()
        return self

    def copy_to(self, device, dtype):
        is_cuda = device.type == 'cuda'
        new_batch = copy.copy(self)
        new_batch.reads_re = self.reads_re.to(device=device, dtype=dtype, non_blocking=is_cuda)
        new_batch.int_tensor = self.int_tensor.to(device, non_blocking=is_cuda)  # don't cast dtype -- needs to stay integral!
        new_batch.float_tensor = self.float_tensor.to(device, non_blocking=is_cuda)  # don't cast dtype -- needs to stay integral!
        return new_batch

    def get_int_array_be(self) -> np.ndarray:
        return self.int_tensor.cpu().numpy()

    def get_float_array_be(self) -> np.ndarray:
        return self.float_tensor.cpu().numpy()

    def size(self) -> int:
        return self._size


class BatchProperty(IntEnum):
    SOURCE = (0, None)
    LABEL = (1, [label.name for label in Label])
    VARIANT_TYPE = (2, [var_type.name for var_type in Variation])
    REF_COUNT_BIN = (3, [ref_count_bin_name(idx) for idx in range(NUM_REF_COUNT_BINS)])
    ALT_COUNT_BIN = (4, [alt_count_bin_name(idx) for idx in range(NUM_ALT_COUNT_BINS)])
    LOGIT_BIN = (5, [logit_bin_name(idx) for idx in range(NUM_LOGIT_BINS)])

    def __new__(cls, value, names_list):
        member = int.__new__(cls, value)
        member._value_ = value
        member.names_list = names_list
        return member

    def get_name(self, n: int):
        return str(n) if self.names_list is None else self.names_list[n]


class BatchIndices:
    PRODUCT_OF_NON_SOURCE_DIMS_INCLUDING_LOGITS = len(Label) * len(Variation) * NUM_REF_COUNT_BINS * NUM_ALT_COUNT_BINS * NUM_LOGIT_BINS

    def __init__(self, sources: IntTensor, labels: IntTensor, var_types: IntTensor, ref_counts: IntTensor, alt_counts:IntTensor):
        """
        sources override is used for something of a hack where in filtering there is only one source, so we use the
        source dimensipn to instead represent the call type
        """
        self.sources = sources
        self.labels = labels
        self.var_types = var_types
        self.ref_count_bins = ref_count_bin_indices(ref_counts)
        self.alt_count_bins = alt_count_bin_indices(alt_counts)

        # We do something kind of dangerous-seeming here: sources is the zeroth dimension and so the formula for
        # flattened indices *doesn't depend on the number of sources* since the stride from one source to the next is the
        # product of all the *other* dimensionalities.  Thus we can set the zeroth dimension to anythiong we want!
        # Just to make sure that this doesn't cause a silent error, we set it to None so that things will blow up
        # if my little analysis here is wrong
        dims = (None, len(Label), len(Variation), NUM_REF_COUNT_BINS, NUM_ALT_COUNT_BINS)
        idx = (self.sources, self.labels, self.var_types, self.ref_count_bins, self.alt_count_bins)
        self.flattened_idx = flattened_indices(dims, idx)

    def _flattened_idx_with_logits(self, logits: Tensor):
        # because logits are the last index, the flattened indices with logits are related to those without in a simple way
        logit_bins = logit_bin_indices(logits)
        return logit_bins + NUM_LOGIT_BINS * self.flattened_idx

    def index_into_tensor(self, tens: BatchIndexedTensor, logits: Tensor = None):
        """
        given 5D batch-indexed tensor x_slvra get the 1D tensor
        result[i] = x_slvra[source[i], label[i], variant type[i], ref bin[i], alt bin[i]]
        This is equivalent to flattening x and indexing by the cached flattened indices
        :return:
        """
        if logits is None and not tens.has_logits():
            return tens.view(-1)[self.flattened_idx]
        elif logits is not None and tens.has_logits():
            return tens.view(-1)[self._flattened_idx_with_logits(logits)]
        else:
            raise Exception("Logits are used if and only if batch-indexed tensor to be indexed includes a logit dimension.")

    def increment_tensor(self, tens: BatchIndexedTensor, values: Tensor, logits: Tensor = None):
        # Similar, but implements: x_slvra[source[i], label[i], variant type[i], ref bin[i], alt bin[i]] += values[i]
        # Addition is in-place. The flattened view(-1) shares memory with the original tensor
        if logits is None and not tens.has_logits():
            return tens.view(-1).index_add_(dim=0, index=self.flattened_idx, source=values)
        elif logits is not None and tens.has_logits():
            return tens.view(-1).index_add_(dim=0, index=self._flattened_idx_with_logits(logits), source=values)
        else:
            raise Exception("Logits are used if and only if batch-indexed tensor to be indexed includes a logit dimension.")

    def increment_tensor_with_sources_and_logits(self, tens: BatchIndexedTensor, values: Tensor, sources_override: IntTensor, logits: Tensor):
        # we sometimes need to override the sources (in filter_variants.py there is a hack where we use the Call type
        # in place of the sources).  This is how we do that.
        assert tens.has_logits(), "Tensor must have a logit dimension"
        indices_with_logits = self._flattened_idx_with_logits(logits)

        # eg, if the dimensions after source are 2, 3, 4 then every increase of the source by 1 is accompanied by an increase
        # of 2x3x4 = 24 in the flattened indices.
        indices = indices_with_logits + BatchIndices.PRODUCT_OF_NON_SOURCE_DIMS_INCLUDING_LOGITS * (sources_override - self.sources)
        return tens.view(-1).index_add_(dim=0, index=indices, source=values)


class BatchIndexedTensor(Tensor):
    """
    stores sums, indexed by batch properties source (s), label (l), variant type (v), ref count (r), alt count (a)
    and, optionally, logit (g).
    
    It's worth noting that as of Pytorch 1.7: 1) when this is wrapped in a Parameter the resulting Parameter works
    for autograd and remains an instance of this class, 2) torch functions like exp etc acting on this class return
    an instance of this class, 3) addition and multiplication by regular torch Tensors ALSO yield a resulting instance
    of this class, 4) indexing this class also produces an instance of this class, which is DEFINITELY undesired.
    """

    # together this init and new construct a BatchIndexedTensor from an existing tensor, or from a numpy array, list etc,
    # sharing the underlying data.  That is, it essentially casts to a BatchIndexedTensor in-place
    # we don't check that the shape is correct.
    @staticmethod
    def __new__(cls, data: Tensor):
        return torch.Tensor._make_subclass(cls, data)

    # I think this needs to have the same signature as __new__?
    def __init__(self, data: Tensor):
        assert data.dim() == 5 or data.dim() == 6, "batch-indexed tensors have either 5 or 6 dimensions"

    def has_logits(self) -> bool:
        return self.dim() == 6

    def num_sources(self) -> int:
        return self.shape[0]

    @classmethod
    def make_zeros(cls, num_sources: int, include_logits: bool = False, device=gpu_if_available()):
        base_shape = (num_sources, len(Label), len(Variation), NUM_REF_COUNT_BINS, NUM_ALT_COUNT_BINS)
        shape = (base_shape + (NUM_LOGIT_BINS, )) if include_logits else base_shape
        return cls(torch.zeros(shape, device=device))

    @classmethod
    def make_ones(cls, num_sources: int, include_logits: bool = False, device=gpu_if_available()):
        base_shape = (num_sources, len(Label), len(Variation), NUM_REF_COUNT_BINS, NUM_ALT_COUNT_BINS)
        shape = (base_shape + (NUM_LOGIT_BINS,)) if include_logits else base_shape
        return cls(torch.ones(shape, device=device))

    def resize_sources(self, new_num_sources):
        old_num_sources = self.num_sources()
        base_shape = (new_num_sources, len(Label), len(Variation), NUM_REF_COUNT_BINS, NUM_ALT_COUNT_BINS)
        shape = (base_shape + (NUM_LOGIT_BINS,)) if self.has_logits() else base_shape
        self.resize_(shape)
        self[old_num_sources:] = 0

    def index_by_batch(self, batch: Batch, logits: Tensor = None):
        return batch.batch_indices().index_into_tensor(self, logits)

    # TODO: move to subclass as in comments below
    def record_datum(self, datum: Datum, value: float = 1.0, grow_source_if_necessary: bool = True):
        assert not self.has_logits(), "this only works when not including logits"
        source = datum.get(Data.SOURCE)
        if source >= self.num_sources():
            if grow_source_if_necessary:
                self.resize_sources(source + 1)
            else:
                raise Exception("Datum source doesn't fit.")
        # no logits here
        ref_idx, alt_idx = ref_count_bin_index(datum.get(Data.REF_COUNT)), alt_count_bin_index(datum.get(Data.ALT_COUNT))
        self[source, datum.get(Data.LABEL), datum.get(Data.VARIANT_TYPE), ref_idx, alt_idx] += value

    # TODO: move to a metrics subclass -- this class should really only be for indexing, not recording
    def record(self, batch: Batch, values: Tensor, logits: Tensor=None, use_original_counts: bool = False):
        batch.batch_indices(use_original_counts).increment_tensor(self, values=values, logits=logits)

    def get_marginal(self, *properties: Tuple[BatchProperty, ...]) -> Tensor:
        """
        sum over all but one or more batch properties.
        For example self.get_marginal(BatchProperty.SOURCE, BatchProperty.LABEL) yields a (num sources x len(Label)) output
        """
        property_set = set(*properties)
        num_dims = len(BatchProperty) - (0 if self.has_logits() else 1)
        other_dims = tuple(n for n in range(num_dims) if n not in property_set)
        return torch.sum(self, dim=other_dims)

