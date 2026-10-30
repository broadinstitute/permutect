import itertools

import pytest
import torch

from permutect.architecture.artifact_model import load_model
from permutect.data.batch import Batch
from permutect.data.batch import DownsampledBatch
from permutect.data.datum import Data
from permutect.data.datum import Datum
from permutect.data.memory_mapped_data import MemoryMappedData
from permutect.test.test_file_names import PERMUTECT_V0_9_2_ARTIFACT_MODEL
from permutect.test.test_file_names import PREPROCESSED_DATA

META = torch.device("meta")


def first_data(num_data: int) -> list[Datum]:
    return list(itertools.islice(MemoryMappedData.load_from_tarfile(PREPROCESSED_DATA).generate(), num_data))


def test_reads_must_match_the_counts():
    # RaggedSets don't check their lengths, since on a GPU that makes the CPU wait, so the reads of a batch are checked
    # against its counts once, when its sets are made
    model = load_model(PERMUTECT_V0_9_2_ARTIFACT_MODEL, device=torch.device("cpu"))
    data = first_data(8)
    model.calculate_features(Batch(data))

    # one more alt read than the datum has
    data[0] = Datum(data[0].int_array.copy(), data[0].float_array.copy(), data[0].reads_re, compressed=True)
    data[0].set(Data.ALT_COUNT, data[0].get(Data.ALT_COUNT) + 1)
    with pytest.raises(AssertionError, match="reads don't match"):
        model.calculate_features(Batch(data))


def test_downsampled_forward_pass_reads_device_data_on_the_cpu_once(monkeypatch):
    # meta tensors have no data, so on the meta device anything that reads GPU data on the CPU, and makes the CPU wait for
    # the GPU, fails.  Downsampling must read the numbers of kept reads once, to size the downsampled reads, so this
    # stubs that one read and checks that the forward pass of a downsampled batch makes no others.
    model = load_model(PERMUTECT_V0_9_2_ARTIFACT_MODEL, device=torch.device("cpu")).to(META)
    batch = Batch(first_data(8)).copy_to(META, dtype=torch.float32)

    host_reads = []
    real_tolist = torch.Tensor.tolist

    def tolist(tensor):
        if not tensor.is_meta:
            return real_tolist(tensor)
        host_reads.append(tuple(tensor.shape))
        return [3, 9]  # the numbers of kept ref and alt reads

    monkeypatch.setattr(torch.Tensor, "tolist", tolist)
    keep_half_b = torch.full((batch.size(),), 0.5, device=META)
    downsampled = DownsampledBatch(batch, ref_fracs_b=keep_half_b, alt_fracs_b=keep_half_b)
    output = model.compute_batch_output(downsampled)
    model.compute_batch_losses(output, downsampled)

    assert host_reads == [(2,)]
    assert len(downsampled.get_reads_re()) == 3 + 9


def test_batch_without_ref_reads():
    # every ref tensor is empty, and the ref features of each datum come from the regularizer of its empty set
    model = load_model(PERMUTECT_V0_9_2_ARTIFACT_MODEL, device=torch.device("cpu"))
    data = []
    for datum in first_data(16):
        alt_only = Datum(datum.int_array.copy(), datum.float_array.copy(), datum.get_alt_reads_re(), compressed=True)
        alt_only.set(Data.REF_COUNT, 0)
        data.append(alt_only)
    batch = Batch(data)
    assert batch.ref_and_alt_totals[0] == 0

    keep_half_b = torch.full((batch.size(),), 0.5)
    for batch_to_check in (batch, DownsampledBatch(batch, ref_fracs_b=keep_half_b, alt_fracs_b=keep_half_b)):
        model.zero_grad()
        output = model.compute_batch_output(batch_to_check)
        losses = model.compute_batch_losses(output, batch_to_check)
        losses.total_losses_b.sum().backward()
        assert torch.isfinite(output.ref_features_be).all() and torch.isfinite(losses.total_losses_b).all()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
