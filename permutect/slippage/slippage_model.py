"""
A model of polymerase slippage, the insertion or deletion of one repeat unit of a short tandem repeat (STR) during
amplification.  It's a sequencing error endemic to STRs, which indel base qualities don't capture, and it ports
FilterMutectCalls' PolymeraseSlippageFilter: an indel can be slippage if it changes by one the number of repeat units of
an STR spanning at least 8 reference bases, and slippage gives alt reads at a low allele fraction.
"""

import torch
from torch import Tensor
from torch import nn

from permutect.utils.enums import INDEL_VARIATIONS
from permutect.utils.stats_utils import beta_binomial_log_lk

# FilterMutectCalls' --min-slippage-length: the minimum number of bases of an STR in the reference for slippage
MIN_SLIPPAGE_STR_LENGTH = 8

# FilterMutectCalls' likelihood of slippage is the regularized incomplete beta function I_r(k + 1, n - k + 1) of the alt
# count k and depth n with a slippage rate r = 0.1, the upper tail P(Bin(n + 1, r) >= k + 1).  That isn't normalized:
# it's (n + 1) r times the likelihood of an alt allele fraction uniform on [0, r].  This beta-binomial's pseudocounts
# (alpha, beta) are a least-squares fit to that uniform model's log likelihoods at depths 20 to 1000 and alt counts from 2
# to 2.5 r n.  Against a somatic prior of 1e-6 and a flat somatic allele fraction, the allele fraction above which a
# somatic indel beats slippage is 0.03 to 0.07 lower than FilterMutectCalls' at depths 20 to 1000, and 0.02 to 0.05 lower
# than the uniform model's.
SLIPPAGE_ALLELE_FRACTION_PSEUDOCOUNTS = (67.13, 1055.24)


class SlippageModel(nn.Module):
    """
    Which variants can be slippage, and the likelihoods of their alt counts given slippage.

    TODO: the pseudocounts are fixed for now.  Learn them, eg for each repeat unit length and STR length, from the
    slippage data that GATK collects.
    """

    def __init__(
        self,
        allele_fraction_pseudocounts: tuple[float, float] = SLIPPAGE_ALLELE_FRACTION_PSEUDOCOUNTS,
        min_str_length: int = MIN_SLIPPAGE_STR_LENGTH,
    ):
        super().__init__()
        self.min_str_length = min_str_length
        # buffers rather than parameters, since nothing that uses the model changes them
        self.register_buffer(
            "allele_fraction_pseudocounts_2", torch.tensor(allele_fraction_pseudocounts, dtype=torch.float64)
        )

    def possible_b(self, variant_types_b: Tensor, repeat_lengths_b: Tensor, repeat_count_changes_b: Tensor) -> Tensor:
        """
        Whether each variant can be slippage: an indel that changes by one the number of repeat units of an STR of at
        least min_str_length reference bases.  Unannotated data, whose repeat lengths and changes are NaN, can't be.
        """
        indels_b = torch.isin(variant_types_b, torch.tensor(INDEL_VARIATIONS, device=variant_types_b.device))
        return indels_b & (repeat_lengths_b >= self.min_str_length) & (repeat_count_changes_b.abs() == 1)

    def tumor_log_likelihoods_b(self, depths_b: Tensor, alt_counts_b: Tensor) -> Tensor:
        """the log likelihoods of each variant's alt count in the tumor, given its depth and slippage"""
        alpha, beta = self.allele_fraction_pseudocounts_2
        return beta_binomial_log_lk(depths_b, alt_counts_b, alpha, beta)

    def save(self, path):
        """save everything needed to reconstruct the model, as a list and a number, which torch.load can read safely"""
        state = {"allele_fraction_pseudocounts": self.allele_fraction_pseudocounts_2.tolist()}
        torch.save(state | {"min_str_length": self.min_str_length}, path)

    @classmethod
    def load(cls, path) -> "SlippageModel":
        state = torch.load(path, map_location="cpu", weights_only=True)
        return cls(tuple(state["allele_fraction_pseudocounts"]), state["min_str_length"])
