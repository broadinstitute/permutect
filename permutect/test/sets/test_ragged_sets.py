import pytest
import torch

from permutect.architecture.gated_mlp import GatedRefAltMLP
from permutect.sets.ragged_sets import RaggedSets

META = torch.device("meta")


def make_read_encoder_inputs(device: torch.device) -> tuple[GatedRefAltMLP, RaggedSets, RaggedSets]:
    # the ref and alt reads of three variants, one of which has no ref reads
    encoder = GatedRefAltMLP(d_model=6, d_ffn=8, num_blocks=2).to(device)
    ref_bre = RaggedSets(torch.randn(5, 6, device=device), torch.tensor([2, 0, 3], device=device))
    alt_bre = RaggedSets(torch.randn(4, 6, device=device), torch.tensor([1, 2, 1], device=device))
    return encoder, ref_bre, alt_bre


def run_read_encoder(encoder: GatedRefAltMLP, ref_bre: RaggedSets, alt_bre: RaggedSets) -> tuple[torch.Tensor, ...]:
    ref_bre, alt_bre = encoder.forward(ref_bre, alt_bre)
    return ref_bre.means_over_sets(), alt_bre.sums_over_sets()


def test_set_operations_do_not_read_data_on_the_cpu():
    # meta tensors have no data, so on the meta device anything that reads GPU data on the CPU, and makes the CPU wait for
    # the GPU, fails.  For example, checking the sum of the set lengths, or repeat_interleave without an output size.
    ref_means_bf, alt_sums_bf = run_read_encoder(*make_read_encoder_inputs(META))
    assert ref_means_bf.shape == alt_sums_bf.shape == (3, 6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_set_operations_do_not_synchronize_cuda():
    # unlike the meta device, this also catches segment_reduce checking its lengths
    inputs = make_read_encoder_inputs(torch.device("cuda"))
    torch.cuda.set_sync_debug_mode("error")
    try:
        run_read_encoder(*inputs)
    finally:
        torch.cuda.set_sync_debug_mode("default")


def test_set_sums_and_expansion():
    # sets of 2, 0, 3 and 1 elements
    sets = RaggedSets(torch.arange(12.0).view(6, 2), torch.tensor([2, 0, 3, 1]))
    assert torch.equal(sets.sums_over_sets(), torch.tensor([[2.0, 4.0], [0.0, 0.0], [18.0, 21.0], [10.0, 11.0]]))

    expanded_nf = sets.expand_from_b_to_n(torch.tensor([[1.0], [2.0], [3.0], [4.0]]))
    assert torch.equal(expanded_nf, torch.tensor([[1.0], [1.0], [3.0], [3.0], [3.0], [4.0]]))
