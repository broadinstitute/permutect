import numpy as np
import pytest
import torch

from permutect.data.batch import Batch
from permutect.data.batch import BatchIndexedTensor
from permutect.data.batch import DownsampledBatch
from permutect.data.count_binning import NUM_ALT_COUNT_BINS
from permutect.data.count_binning import NUM_LOGIT_BINS
from permutect.data.count_binning import NUM_REF_COUNT_BINS
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
def test_batch_indexed_tensor_shapes(include_logits):
    cpu = torch.device("cpu")
    shape = (3, len(Label), len(Variation), NUM_REF_COUNT_BINS, NUM_ALT_COUNT_BINS) + (
        (NUM_LOGIT_BINS,) if include_logits else ()
    )
    assert BatchIndexedTensor.shape_for(3, include_logits) == shape

    zeros = BatchIndexedTensor.zeros(3, include_logits, device=cpu)
    ones = BatchIndexedTensor.ones(3, include_logits, device=cpu)
    assert zeros.shape == ones.shape == shape and ones.has_logits() == include_logits

    # growing the source dimension keeps the existing sources and zeroes the new ones
    ones.resize_sources(5)
    assert ones.shape == BatchIndexedTensor.shape_for(5, include_logits)
    assert torch.all(ones[:3] == 1) and torch.all(ones[3:] == 0)

    if include_logits:
        accuracy_metrics = AccuracyMetrics.create(3, device=cpu)
        assert isinstance(accuracy_metrics, AccuracyMetrics) and accuracy_metrics.shape == shape
