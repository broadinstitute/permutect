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

# Below this MAF, a normal is taken to have only one distinct haplotype: its other haplotype is identical, in a run of
# homozygosity, or missing, in a hemizygous deletion.  The segment_pileups HMM gives such regions MAFs of 0 or nearly 0,
# and a normal's other regions MAFs near 1/2.
HOMOZYGOUS_NORMAL_MAF = 0.05


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

    def contamination_log_likelihoods(
        self, alt_counts: Tensor, depths: Tensor, log_allele_frequencies: Tensor
    ) -> Tensor:
        """
        The log likelihood of each site's alt count given that the sample is hom ref and its alt reads come from a
        contaminant who carries the alt allele: a het or hom alt contaminant, whose Hardy-Weinberg probabilities
        2q(1 - q) and q^2 are divided by the probability q(2 - q) of carrying the alt allele, as in
        SegmentationModel.germline_log_likelihoods.  This is the single-contaminant likelihood of GATK's contamination
        filter divided by q(2 - q), with reference bias.  Without contamination, alt reads are impossible.
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

    def germline_log_likelihoods(
        self,
        tumor_alt_counts: Tensor,
        tumor_depths: Tensor,
        normal_alt_counts: Tensor,
        normal_depths: Tensor,
        log_allele_frequencies: Tensor,
        tumor_mafs: Tensor,
        normal_mafs: Tensor,
    ) -> Tensor:
        """
        The joint log likelihood of the tumor's and the normal's alt counts at each site given that the person carries
        the alt allele in the germline.  Each sample's counts get its own segmentation's emission model at its own MAF.
        Without a normal, its depths are 0 and its MAFs 1/2, and this is the likelihood of the tumor's counts alone.  The
        population allele frequencies q are logs so that tiny ones don't underflow.  The inputs broadcast together.

        Genotypes: a carrier of the alt allele is a het, with Hardy-Weinberg probability 2q(1 - q), or a hom alt, with
        probability q^2.  The posterior model's germline prior is the probability q(2 - q) of being a carrier, so here
        these are divided by it, giving 2(1 - q) / (2 - q) for a het and q / (2 - q) for a hom alt.  Hom refs aren't
        carriers, and the emission model's outliers are left out.

        Sharing the genotype: the tumor and the normal have the same germline genotype, so the likelihood isn't the
        product of the tumor's and the normal's likelihoods, each summed over genotypes.  Instead each genotype's term
        multiplies the two samples' likelihoods of that genotype.

        Sharing the haplotype: a het's alt allele is on one of the person's two haplotypes, each with probability 1/2,
        and it's the same haplotype in the tumor and the normal.  Each sample's segmentation gives its minor haplotype's
        fraction of its alleles, the MAF, but not which haplotype that is.  We assume that the tumor's minor haplotype is
        the normal's, so that the het term is

            (1 - q) / (2 - q) * [ T(MAF_T) N(MAF_N) + T(1 - MAF_T) N(1 - MAF_N) ]

        where T(f) and N(f) are the tumor's and the normal's likelihoods of their alt counts if a fraction f of their
        alleles carry the alt allele: the alt allele is on the minor haplotype of both samples or on the major haplotype
        of both.  This assumption costs nothing where the normal is balanced, MAF_N = 1/2, because its two terms are
        equal.  Where the normal has only one distinct haplotype, MAF_N = 0, it's exact, because the tumor descends from
        the same germline: its only haplotype is the normal's.  It's wrong only where an imbalanced normal's minor
        haplotype is the tumor's major one, which takes both a germline copy number change and a tumor one.

        Summing each sample's likelihood over the phases separately would let the alt allele be on the normal's missing
        haplotype, which explains a normal without alt reads, and on the tumor's only haplotype, which explains a tumor
        all of whose reads are alt.  Then a clonal somatic variant in a run of homozygosity of the normal, which the
        tumor shares, would look germline.

        One distinct haplotype: where the normal's MAF is below HOMOZYGOUS_NORMAL_MAF, the person's other haplotype is
        identical or missing, so no allele is on the minor haplotype alone, and the first het term is left out.  Its
        only use would be to explain alt reads in a tumor whose MAF, wrongly, isn't 0 there, as when the tumor's
        segmentation smooths over a short run of homozygosity.  The weights then sum to 1 / (2 - q) rather than 1, so
        that, with the germline prior q(2 - q), the probability of carrying the alt allele, on the one distinct
        haplotype, is q.
        """
        allele_frequencies = torch.exp(log_allele_frequencies)

        # each sample's log likelihoods of its counts with the alt allele on its minor haplotype, on its major haplotype,
        # and on both, from its own emission model.  Their hom ref terms are unused.
        _, tumor_alt_on_minor, tumor_alt_on_major, tumor_hom_alt = (
            self.tumor.emission_model.phased_genotype_log_likelihoods(
                tumor_alt_counts, tumor_depths, allele_frequencies, tumor_mafs
            )
        )
        _, normal_alt_on_minor, normal_alt_on_major, normal_hom_alt = (
            self.normal.emission_model.phased_genotype_log_likelihoods(
                normal_alt_counts, normal_depths, allele_frequencies, normal_mafs
            )
        )

        # the log probabilities, given that the person is a carrier, of a het with the alt allele on a given haplotype,
        # q(1 - q) / q(2 - q) = (1 - q) / (2 - q), and of a hom alt, q^2 / q(2 - q) = q / (2 - q).  The factors of q are
        # cancelled so that q = 0 doesn't give -inf - (-inf) = NaN.
        log_two_minus_q = torch.log(2 - allele_frequencies)
        log_het_haplotype_weight = log1mexp(log_allele_frequencies) - log_two_minus_q
        log_hom_alt_weight = log_allele_frequencies - log_two_minus_q

        # the genotype and the alt allele's haplotype are shared by the tumor and the normal
        alt_on_minor = log_het_haplotype_weight + tumor_alt_on_minor + normal_alt_on_minor
        alt_on_major = log_het_haplotype_weight + tumor_alt_on_major + normal_alt_on_major
        hom_alt = log_hom_alt_weight + tumor_hom_alt + normal_hom_alt

        # a normal with one distinct haplotype has no alt allele on its minor haplotype alone.  IMPOSSIBLE_LOG_PROB rather
        # than -inf avoids NaNs.
        normal_has_one_haplotype = normal_mafs < HOMOZYGOUS_NORMAL_MAF
        alt_on_minor = torch.where(normal_has_one_haplotype, constants.IMPOSSIBLE_LOG_PROB, alt_on_minor)

        return torch.logsumexp(torch.stack(torch.broadcast_tensors(alt_on_minor, alt_on_major, hom_alt)), dim=0)

    def save(self, path):
        state = {"tumor": self.tumor.state(), "normal": self.normal.state()}
        state.update(contamination=float(self.contamination), contamination_error=float(self.contamination_error))
        torch.save(state, path)

    @classmethod
    def load(cls, path) -> "SegmentationModel":
        state = torch.load(path, map_location="cpu", weights_only=True)
        tumor, normal = (SampleSegmentation.from_state(state[sample]) for sample in ("tumor", "normal"))
        return cls(tumor, normal, state["contamination"], state["contamination_error"])
