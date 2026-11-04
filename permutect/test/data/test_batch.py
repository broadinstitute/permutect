import numpy as np
import pytest
import torch

from permutect.data.batch import Batch
from permutect.data.batch import BatchIndexedTensor
from permutect.data.batch import BatchIndices
from permutect.data.batch import DownsampledBatch
from permutect.data.count_binning import NUM_ALT_COUNT_BINS
from permutect.data.count_binning import NUM_LOGIT_BINS
from permutect.data.count_binning import NUM_REF_COUNT_BINS
from permutect.data.count_binning import alt_count_bin_index
from permutect.data.count_binning import logit_bin_indices
from permutect.data.count_binning import ref_count_bin_index
from permutect.data.datum import RAW_READS_ARRAY_DTYPE
from permutect.data.datum import Data
from permutect.data.datum import Datum
from permutect.metrics.loss_metrics import AccuracyMetrics
from permutect.utils.enums import Label
from permutect.utils.enums import Variation

# ref and alt read counts of each variant in the test batch, including a variant with no ref reads
COUNTS = [(3, 2), (0, 4), (5, 1), (2, 3)]


def make_datum(variant: int, ref_count: int, alt_count: int) -> Datum:
    # each read records where it came from: [variant index, 0 for ref or 1 for alt, read number within ref or alt]
    reads = [[variant, 0, n] for n in range(ref_count)] + [[variant, 1, n] for n in range(alt_count)]
    datum = Datum(
        int_array=np.zeros(Data.NUM_SCALAR_INT_ELEMENTS),
        float_array=np.zeros(Data.NUM_SCALAR_FLOAT_ELEMENTS),
        reads_re=np.array(reads, dtype=RAW_READS_ARRAY_DTYPE),
    )
    datum.set(Data.REF_COUNT, ref_count)
    datum.set(Data.ALT_COUNT, alt_count)
    return datum


def make_batch() -> Batch:
    return Batch([make_datum(variant, ref, alt) for variant, (ref, alt) in enumerate(COUNTS)])


def test_downsampled_batch_keeping_every_read_matches_parent():
    parent = make_batch()
    keep_all = torch.ones(parent.size())
    downsampled = DownsampledBatch(parent, ref_fracs_b=keep_all, alt_fracs_b=keep_all)

    assert torch.equal(downsampled.get(Data.REF_COUNT), parent.get(Data.REF_COUNT).int())
    assert torch.equal(downsampled.get(Data.ALT_COUNT), parent.get(Data.ALT_COUNT).int())
    assert torch.equal(downsampled.get_reads_re(), parent.get_reads_re())


def test_downsampled_reads_belong_to_their_variant():
    parent = make_batch()
    batch_size = parent.size()
    for _ in range(20):
        downsampled = DownsampledBatch(parent, ref_fracs_b=torch.rand(batch_size), alt_fracs_b=torch.rand(batch_size))
        ref_counts, alt_counts = downsampled.get(Data.REF_COUNT), downsampled.get(Data.ALT_COUNT)
        reads = downsampled.get_reads_re()
        total_ref = int(torch.sum(ref_counts))

        assert len(reads) == total_ref + int(torch.sum(alt_counts))
        assert torch.all(alt_counts >= 1)  # every variant keeps at least one alt read

        # the downsampled ref section holds each variant's kept ref reads in turn, then the alt section its alt reads
        ref_variants = torch.repeat_interleave(torch.arange(batch_size), ref_counts).to(reads.dtype)
        alt_variants = torch.repeat_interleave(torch.arange(batch_size), alt_counts).to(reads.dtype)
        assert torch.equal(reads[:total_ref, 0], ref_variants)
        assert torch.all(reads[:total_ref, 1] == 0)
        assert torch.equal(reads[total_ref:, 0], alt_variants)
        assert torch.all(reads[total_ref:, 1] == 1)

        # no read is kept twice
        assert len(set(map(tuple, reads.tolist()))) == len(reads)


def test_downsampled_int_arrays_leave_the_parent_batch_unchanged():
    parent = make_batch()
    parent_int_tensor = parent.int_tensor.clone()
    keep_none = torch.zeros(parent.size())
    downsampled = DownsampledBatch(parent, ref_fracs_b=keep_none, alt_fracs_b=keep_none)
    assert downsampled.get_int_array_be()[:, Data.ALT_COUNT.idx].tolist() == [1] * parent.size()
    assert torch.equal(parent.int_tensor, parent_int_tensor)


def test_each_variant_keeps_a_random_alt_read_of_its_own():
    # keeping no reads at random leaves each variant only the alt read that downsampling always keeps, which must be
    # uniformly random and independent of other variants' choices
    parent = make_batch()  # alt counts 2, 4, 1, 3
    keep_none = torch.zeros(parent.size())
    torch.manual_seed(0)
    # the read number of each variant's kept alt read, in each of many downsamplings
    kept_nb = torch.stack(
        [
            DownsampledBatch(parent, ref_fracs_b=keep_none, alt_fracs_b=keep_none).get_reads_re()[:, 2]
            for _ in range(2000)
        ]
    ).long()
    for variant, (_, alt_count) in enumerate(COUNTS):
        frequencies = torch.bincount(kept_nb[:, variant], minlength=alt_count) / len(kept_nb)
        assert torch.allclose(frequencies, torch.full((alt_count,), 1 / alt_count), atol=0.05)
    # one draw shared by the batch made the variant with 2 alt reads keep read n mod 2 whenever the one with 4 kept n
    assert abs((kept_nb[:, 0] == kept_nb[:, 1] % 2).float().mean() - 0.5) < 0.05


def test_downsampling_is_reproducible_from_the_torch_seed():
    parent = make_batch()
    keep_none = torch.zeros(parent.size())
    runs = []
    for _ in range(2):
        torch.manual_seed(0)
        runs.append(
            [DownsampledBatch(parent, ref_fracs_b=keep_none, alt_fracs_b=keep_none).get_reads_re() for _ in range(10)]
        )
    assert all(torch.equal(first, second) for first, second in zip(*runs))


def test_batch_get_matches_datum_get_for_every_field():
    # uint32 fields such as the position are stored as two int16s, and decoding them only worked for a single Datum
    data = [make_datum(variant, ref, alt) for variant, (ref, alt) in enumerate(COUNTS)]
    for n, datum in enumerate(data):
        datum.set(Data.POSITION, [100, 2**16 + 1, 4_000_000_000, 123_456_789][n])
        datum.set(Data.REF_ALLELE_AS_BASE_5, 5**13 - n)
        datum.set(Data.ALT_ALLELE_AS_BASE_5, n)
        datum.set(Data.SOURCE, n)
        datum.set(Data.SEQ_ERROR_LOG_LK, -0.5 * n)
    batch = Batch(data)
    keep_all = torch.ones(batch.size())
    downsampled = DownsampledBatch(batch, ref_fracs_b=keep_all, alt_fracs_b=keep_all)

    for data_field in Data:
        # float64 holds both the float16 fields and the uint32 fields exactly
        expected = torch.tensor([float(datum.get(data_field)) for datum in data], dtype=torch.float64)
        for batch_to_check in (batch, downsampled):
            values = batch_to_check.get(data_field)
            assert isinstance(values, torch.Tensor) and values.shape == (len(data),), data_field
            assert torch.equal(values.double(), expected), data_field


@pytest.mark.parametrize("include_logits", [False, True])
def test_batch_indices_follow_the_batch_indexed_tensor_layout(include_logits):
    # flattened indices, including those with overridden sources and labels, pick the same elements as plain indexing
    generator = torch.Generator().manual_seed(0)
    num_sources, size = 3, 50
    sources, override_sources = torch.randint(num_sources, (2, size), generator=generator)
    labels, override_labels = torch.randint(len(Label), (2, size), generator=generator)
    var_types = torch.randint(len(Variation), (size,), generator=generator)
    ref_counts, alt_counts = torch.randint(1, 20, (2, size), generator=generator)
    logits = 5 * torch.randn(size, generator=generator) if include_logits else None
    indices = BatchIndices(sources, labels, var_types, ref_counts, alt_counts)

    plain_index = (override_sources, override_labels, var_types)
    plain_index += (ref_count_bin_index(ref_counts), alt_count_bin_index(alt_counts))
    plain_index += (logit_bin_indices(logits),) if include_logits else ()
    shape = BatchIndexedTensor.shape_for(num_sources, include_logits)

    values = torch.rand(shape, generator=generator)
    picked = indices.index_into_tensor(BatchIndexedTensor(values), override_sources, override_labels, logits)
    assert torch.equal(picked, values[plain_index])

    totals = BatchIndexedTensor.zeros(num_sources, include_logits, device=torch.device("cpu"))
    indices.increment_tensor(totals, torch.ones(size), override_sources, override_labels, logits)
    assert torch.equal(totals, torch.zeros(shape).index_put_(plain_index, torch.ones(size), accumulate=True))


@pytest.mark.parametrize("include_logits", [False, True])
def test_batch_indexed_tensor_shapes(include_logits):
    cpu = torch.device("cpu")
    shape = (3, len(Label), len(Variation), NUM_REF_COUNT_BINS, NUM_ALT_COUNT_BINS) + (
        (NUM_LOGIT_BINS,) if include_logits else ()
    )
    assert BatchIndexedTensor.shape_for(3, include_logits) == shape

    zeros = BatchIndexedTensor.zeros(3, include_logits, device=cpu)
    ones = BatchIndexedTensor.ones(3, include_logits, device=cpu)
    assert zeros.shape == ones.shape == shape and ones.has_logits() == include_logits

    if include_logits:
        accuracy_metrics = AccuracyMetrics.create(3, device=cpu)
        assert isinstance(accuracy_metrics, AccuracyMetrics) and accuracy_metrics.shape == shape
