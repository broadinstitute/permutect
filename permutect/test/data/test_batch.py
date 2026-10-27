import numpy as np
import torch

from permutect.data.batch import Batch
from permutect.data.batch import DownsampledBatch
from permutect.data.datum import RAW_READS_ARRAY_DTYPE
from permutect.data.datum import Data
from permutect.data.datum import Datum

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
