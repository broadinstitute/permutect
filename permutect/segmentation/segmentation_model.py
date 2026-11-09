"""
What segment_pileups learns about a sample and, optionally, its matched normal: each one's minor allele fraction (MAF)
segments and the parameters of the hidden Markov model that found them, and the sample's contamination.  filter_variants
uses it for the MAF at each variant and for the likelihoods of germline variants and of contamination.
"""

import math
from collections import defaultdict

import torch
from intervaltree import IntervalTree
from torch import Tensor
from torch import nn

from permutect import constants
from permutect.misc_utils import freeze
from permutect.segmentation.emissions import EmissionModel
from permutect.segmentation.emissions import Pileups
from permutect.segmentation.segmenter import Segment
from permutect.utils.math_utils import log1mexp

# the floating-point fields of a Segment, after its contig, start and end
SEGMENT_MAF_FIELDS = ("maf", "secondary_maf", "maf_low", "maf_high")


class SampleSegmentation(nn.Module):
    """
    The MAF segments of one sample and the parameters that segment_pileups learned for them: those of the emission model,
    and the prior probabilities of the hidden states' MAFs and the rate of jumps between states.  Without segments, the
    MAF is 1/2 everywhere, and the emission model has its default parameters.
    """

    def __init__(
        self,
        sample: str = "",
        segments: list[Segment] = (),
        emission_model: EmissionModel = None,
        maf_states_k: Tensor = None,
        maf_prior_k: Tensor = None,
        jump_rate: float = math.nan,
    ):
        super().__init__()
        self.sample = sample
        self.segments = list(segments)
        self.emission_model = EmissionModel() if emission_model is None else emission_model
        self.register_buffer("maf_states_k", torch.zeros(0) if maf_states_k is None else maf_states_k)
        self.register_buffer("maf_prior_k", torch.zeros(0) if maf_prior_k is None else maf_prior_k)
        self.jump_rate = jump_rate

    def maf_intervals(self) -> defaultdict:
        """the segments' MAFs in interval trees by contig, to look up the MAF of each variant"""
        result = defaultdict(IntervalTree)
        for segment in self.segments:
            if not math.isnan(segment.maf):
                result[segment.contig][segment.start : segment.end + 1] = segment.maf  # intervals are half-open
        return result

    def emission_log_likelihoods(
        self, alt_counts: Tensor, depths: Tensor, allele_frequencies: Tensor, mafs: Tensor
    ) -> Tensor:
        """
        The segmentation's log likelihood of each site's alt count given its depth, population allele frequency and MAF:
        hom ref, het and hom alt genotypes with their Hardy-Weinberg probabilities, and outliers
        """
        pileups = Pileups(alt_counts, depths, allele_frequencies)
        return self.emission_model.log_likelihoods(pileups, mafs[:, None]).squeeze(1)

    def germline_log_likelihoods(
        self, alt_counts: Tensor, depths: Tensor, log_allele_frequencies: Tensor, mafs: Tensor
    ) -> Tensor:
        """
        The log likelihood of each site's alt count given that the sample carries the alt allele: the segmentation's het
        and hom alt genotypes, whose Hardy-Weinberg probabilities 2q(1 - q) and q^2 are divided by the probability q(2 - q)
        of carrying the alt allele, without hom refs and outliers.  The allele frequencies are logs so that tiny ones
        don't underflow.
        """
        allele_frequencies = torch.exp(log_allele_frequencies)
        _, het, hom_alt = self.emission_model.genotype_log_likelihoods(alt_counts, depths, allele_frequencies, mafs)
        # the het term is the sum over both phases, each of probability q(1 - q), so the het weight is (1 - q) / (2 - q)
        log_weighted_het = log1mexp(log_allele_frequencies) + het
        log_weighted_hom_alt = log_allele_frequencies + hom_alt
        return torch.logaddexp(log_weighted_het, log_weighted_hom_alt) - torch.log(2 - allele_frequencies)

    def contamination_log_likelihoods(
        self, alt_counts: Tensor, depths: Tensor, log_allele_frequencies: Tensor
    ) -> Tensor:
        """
        The log likelihood of each site's alt count given that the sample is hom ref and its alt reads come from a
        contaminant who carries the alt allele: a het or hom alt contaminant, whose Hardy-Weinberg probabilities
        2q(1 - q) and q^2 are divided by the probability q(2 - q) of carrying the alt allele, as in
        germline_log_likelihoods.  This is the single-contaminant likelihood of GATK's contamination filter divided by
        q(2 - q), with reference bias.  Without contamination, alt reads are impossible.
        """
        allele_frequencies = torch.exp(log_allele_frequencies)
        het, hom_alt = self.emission_model.contaminant_log_likelihoods(alt_counts, depths)
        log_weighted_het = math.log(2) + log1mexp(log_allele_frequencies) + het
        log_weighted_hom_alt = log_allele_frequencies + hom_alt
        result = torch.logaddexp(log_weighted_het, log_weighted_hom_alt) - torch.log(2 - allele_frequencies)
        # finite, rather than the -inf of alt reads without contamination
        return torch.clamp(result, min=constants.IMPOSSIBLE_LOG_PROB)

    def state(self) -> dict:
        """
        Everything needed to reconstruct it, as tensors and Python strings and numbers, which torch.load can read with
        weights_only, unlike numpy scalars
        """
        segments = {"contig": [str(segment.contig) for segment in self.segments]}
        for field in ("start", "end"):
            segments[field] = torch.tensor([getattr(s, field) for s in self.segments], dtype=torch.long)
        for field in SEGMENT_MAF_FIELDS:
            segments[field] = torch.tensor([getattr(s, field) for s in self.segments], dtype=torch.float64)
        return {
            "sample": str(self.sample),
            "segments": segments,
            "contamination": float(self.emission_model.contamination),
            "max_outlier_weight": float(self.emission_model.max_outlier_weight),
            "emission_model": self.emission_model.state_dict(),
            "maf_states": self.maf_states_k,
            "maf_prior": self.maf_prior_k,
            "jump_rate": float(self.jump_rate),
        }

    @classmethod
    def from_state(cls, state: dict) -> "SampleSegmentation":
        emission_model = EmissionModel(
            contamination=state["contamination"], max_outlier_weight=state["max_outlier_weight"]
        )
        emission_model.load_state_dict(state["emission_model"])
        columns = state["segments"]
        fields = [columns["start"].tolist(), columns["end"].tolist()]
        fields += [columns[field].tolist() for field in SEGMENT_MAF_FIELDS]
        segments = [Segment(contig, *values) for contig, *values in zip(columns["contig"], *fields)]
        args = (state["maf_states"], state["maf_prior"], state["jump_rate"])
        return cls(state["sample"], segments, emission_model, *args)


class SegmentationModel(nn.Module):
    """
    The segmentation of a sample and of its matched normal, if any, and the sample's contamination and its error.  Its
    parameters are frozen, since they were fit to the pileups, not to the variants.
    """

    def __init__(
        self,
        tumor: SampleSegmentation = None,
        normal: SampleSegmentation = None,
        contamination: float = 0.0,
        contamination_error: float = 1.0,
    ):
        super().__init__()
        self.tumor = SampleSegmentation() if tumor is None else tumor
        self.normal = SampleSegmentation() if normal is None else normal
        self.contamination = contamination
        self.contamination_error = contamination_error
        freeze(self.parameters())

    def save(self, path):
        state = {"tumor": self.tumor.state(), "normal": self.normal.state()}
        state.update(contamination=float(self.contamination), contamination_error=float(self.contamination_error))
        torch.save(state, path)

    @classmethod
    def load(cls, path) -> "SegmentationModel":
        state = torch.load(path, map_location="cpu", weights_only=True)
        tumor, normal = (SampleSegmentation.from_state(state[sample]) for sample in ("tumor", "normal"))
        return cls(tumor, normal, state["contamination"], state["contamination_error"])
