import pytest
import torch

from permutect.architecture.gated_mlp import GatedRefAltMLP
from permutect.sets.ragged_sets import RaggedSets


def make_read_encoder_inputs(device: torch.device) -> tuple[GatedRefAltMLP, RaggedSets, RaggedSets]:
    # the ref and alt reads of three variants, one of which has no ref reads
    encoder = GatedRefAltMLP(d_model=6, d_ffn=8, num_blocks=2).to(device)
    ref_bre = RaggedSets(torch.randn(5, 6, device=device), torch.tensor([2, 0, 3], device=device))
    alt_bre = RaggedSets(torch.randn(4, 6, device=device), torch.tensor([1, 2, 1], device=device))
    return encoder, ref_bre, alt_bre


def run_read_encoder(encoder: GatedRefAltMLP, ref_bre: RaggedSets, alt_bre: RaggedSets) -> tuple[torch.Tensor, ...]:
    ref_bre, alt_bre = encoder.forward(ref_bre, alt_bre)
    return ref_bre.means_over_sets(), alt_bre.sums_over_sets()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_set_operations_do_not_synchronize_cuda():
    # anything that makes the CPU wait for the GPU, such as segment_reduce checking its lengths, is an error
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
