import math

import numpy as np
import pytest
import torch

from permutect import constants
from permutect.segmentation.emissions import EmissionModel
from permutect.segmentation.emissions import Pileups
from permutect.segmentation.segmentation_model import HOMOZYGOUS_NORMAL_MAF
from permutect.segmentation.segmentation_model import SampleSegmentation
from permutect.segmentation.segmentation_model import SegmentationModel
from permutect.segmentation.segmenter import Segment

# learned parameters unlike the defaults, with the outlier weight at its cap
EMISSION_PARAMETERS = dict(
    contamination=0.03, error_rate=0.002, hom_concentration=150.0, het_concentration=80.0, reference_bias=1.1
)


def emission_model() -> EmissionModel:
    model = EmissionModel(**EMISSION_PARAMETERS, outlier_weight=0.005)
    with torch.no_grad():
        model.parametrizations.outlier_weight.original.fill_(50.0)
    return model


def sample_segmentation(sample: str) -> SampleSegmentation:
    segments = [
        Segment("chr1", 100, 2000, 0.3, 0.3, 0.28, 0.32),
        # a gap between segments, with the MAFs of the segments before and after
        Segment("chr1", 2001, 2999, 0.3, 0.5),
        Segment("chr1", 3000, 5000, 0.5, 0.5, 0.48, 0.5),
        Segment("chr2", 10, 20, 0.1, 0.1, 0.0, 0.2),
    ]
    maf_states_k = torch.linspace(0, 0.5, 51, dtype=torch.float64)
    maf_prior_k = torch.softmax(torch.randn(51, dtype=torch.float64, generator=torch.Generator().manual_seed(0)), 0)
    return SampleSegmentation(sample, segments, emission_model(), maf_states_k, maf_prior_k, 3e-8)


def all_alt_counts(depth: int, num_sites: int) -> tuple[torch.Tensor, torch.Tensor]:
    """every alt count from 0 to the depth for each of num_sites sites, as (sites, alt counts) tensors"""
    alt_counts_nk = torch.arange(depth + 1, dtype=torch.float64).repeat(num_sites, 1)
    return alt_counts_nk, torch.full_like(alt_counts_nk, float(depth))


def segmentation_model() -> SegmentationModel:
    # the tumor's and the normal's emission models have different parameters
    model = SegmentationModel(sample_segmentation("tumor"), sample_segmentation("normal"), 0.03, 0.002)
    with torch.no_grad():
        model.normal.emission_model.parametrizations.het_concentration.original.fill_(math.log(20.0))
    return model


def tumor_only_germline_log_likelihoods(model: SegmentationModel, alt_counts, depths, log_afs, mafs) -> torch.Tensor:
    # without a normal, whose depths are 0 and MAFs 1/2
    zeros = torch.zeros_like(alt_counts)
    return model.germline_log_likelihoods(alt_counts, depths, zeros, zeros, log_afs, mafs, torch.full_like(mafs, 0.5))


def test_germline_log_likelihoods_are_normalized():
    # the joint distribution of the tumor's and normal's alt counts given that the person carries the alt allele sums
    # to 1, except where the normal has one distinct haplotype, so that no alt allele is on its minor haplotype alone,
    # and it sums to 1 / (2 - q)
    model = segmentation_model()
    allele_frequencies_n = torch.tensor([1e-6, 0.01, 0.3, 0.5, 0.9, 1.0, 0.2, 0.01], dtype=torch.float64)
    tumor_mafs_n = torch.tensor([0.5, 0.4, 0.2, 0.0, 0.5, 0.3, 0.0, 0.3], dtype=torch.float64)
    normal_mafs_n = torch.tensor([0.5, 0.45, 0.2, 0.5, 0.3, 0.5, 0.0, 0.01], dtype=torch.float64)
    tumor_depth, normal_depth = 30, 20
    tumor_alt_counts_nkj = torch.arange(tumor_depth + 1, dtype=torch.float64)[None, :, None]
    normal_alt_counts_nkj = torch.arange(normal_depth + 1, dtype=torch.float64)[None, None, :]

    def per_site(tensor_n):
        return tensor_n[:, None, None]

    log_likelihoods_nkj = model.germline_log_likelihoods(
        tumor_alt_counts_nkj,
        torch.tensor(float(tumor_depth), dtype=torch.float64),
        normal_alt_counts_nkj,
        torch.tensor(float(normal_depth), dtype=torch.float64),
        per_site(torch.log(allele_frequencies_n)),
        per_site(tumor_mafs_n),
        per_site(normal_mafs_n),
    )
    totals_n = torch.exp(torch.logsumexp(log_likelihoods_nkj.flatten(start_dim=1), dim=1))
    one_haplotype_n = normal_mafs_n < HOMOZYGOUS_NORMAL_MAF
    expected_n = torch.where(one_haplotype_n, 1 / (2 - allele_frequencies_n), torch.ones_like(allele_frequencies_n))
    assert torch.allclose(totals_n, expected_n)


def test_tumor_and_normal_share_the_haplotype_of_a_het():
    # in an imbalanced normal and tumor, MAF 0.25, a het's alt allele on the minor haplotype gives an alt fraction of
    # 0.25 in both samples, and on the major haplotype 0.75 in both, but never 0.25 in one and 0.75 in the other.
    # Summing each sample's likelihood over the phases separately would make all four equally likely.
    model = SegmentationModel()
    log_afs, mafs = torch.log(torch.tensor([1e-4])), torch.tensor([0.25])

    def germline(tumor_alt_count: float, normal_alt_count: float) -> float:
        depth = torch.tensor([200.0])
        return model.germline_log_likelihoods(
            torch.tensor([tumor_alt_count]), depth, torch.tensor([normal_alt_count]), depth, log_afs, mafs, mafs
        ).item()

    assert germline(50, 50) == pytest.approx(germline(150, 150), abs=1e-4)
    assert germline(50, 50) > germline(150, 50) + 20
    assert germline(50, 50) > germline(50, 150) + 20


def test_a_somatic_variant_in_shared_loss_of_heterozygosity_is_not_germline():
    # SEQC2's tumor and normal share runs of homozygosity, MAF 0, where a clonal somatic variant has every tumor read
    # alt and no normal read alt.  The alt allele can't be on the normal's missing haplotype and the tumor's only one,
    # so this is far less likely than a germline hom alt, every read of both samples alt.
    model = segmentation_model()
    log_afs, mafs = torch.log(torch.tensor([1e-6])), torch.tensor([0.0])

    def germline(tumor_alt_count: float, normal_alt_count: float) -> float:
        tumor_depth, normal_depth = torch.tensor([72.0]), torch.tensor([144.0])
        tumor_alt, normal_alt = torch.tensor([tumor_alt_count]), torch.tensor([normal_alt_count])
        return model.germline_log_likelihoods(
            tumor_alt, tumor_depth, normal_alt, normal_depth, log_afs, mafs, mafs
        ).item()

    assert germline(72, 0) < -50
    assert germline(72, 144) > -5


def test_without_a_normal_haplotype_no_alt_allele_is_on_the_minor_haplotype_alone():
    # where the normal has one distinct haplotype, a tumor MAF of 0.33, from a tumor segment that smooths over a short
    # run of homozygosity, can't explain a tumor alt fraction of 0.33 with a normal without alt reads.  That holds for a
    # normal MAF slightly above 0 too, at which the normal's lack of alt reads would be likely if the alt allele could be
    # on its minor haplotype.
    model = segmentation_model()
    log_afs, tumor_mafs = torch.log(torch.tensor([1e-6])), torch.tensor([0.33])
    tumor_alt, tumor_depth = torch.tensor([46.0]), torch.tensor([136.0])
    normal_alt, normal_depth = torch.tensor([0.0]), torch.tensor([165.0])

    def germline(normal_maf: float) -> float:
        normal_mafs = torch.tensor([normal_maf])
        return model.germline_log_likelihoods(
            tumor_alt, tumor_depth, normal_alt, normal_depth, log_afs, tumor_mafs, normal_mafs
        ).item()

    assert germline(0.0) < -50
    assert germline(HOMOZYGOUS_NORMAL_MAF / 10) < -50


def test_emission_log_likelihoods_are_the_segmentation_emissions():
    segmentation = sample_segmentation("tumor")
    alt_counts_n = torch.tensor([0.0, 3.0, 12.0, 30.0, 38.0], dtype=torch.float64)
    depths_n = torch.tensor([40.0, 40.0, 25.0, 60.0, 40.0], dtype=torch.float64)
    allele_frequencies_n = torch.tensor([0.05, 0.3, 0.5, 0.2, 0.9], dtype=torch.float64)
    mafs_n = torch.tensor([0.5, 0.3, 0.1, 0.5, 0.4], dtype=torch.float64)
    result_n = segmentation.emission_log_likelihoods(alt_counts_n, depths_n, allele_frequencies_n, mafs_n)
    for n in range(len(mafs_n)):
        site = Pileups(alt_counts_n[n : n + 1], depths_n[n : n + 1], allele_frequencies_n[n : n + 1])
        expected = segmentation.emission_model.log_likelihoods(site, mafs_n[n : n + 1])
        assert result_n[n].item() == pytest.approx(expected.item(), rel=1e-12)

    # with the Hardy-Weinberg weights of hom ref, het and hom alt, a normalized distribution of the alt count
    alt_counts_nk, depths_nk = all_alt_counts(40, len(mafs_n))
    log_likelihoods_nk = torch.stack(
        [
            segmentation.emission_log_likelihoods(alt_counts_nk[:, k], depths_nk[:, k], allele_frequencies_n, mafs_n)
            for k in range(41)
        ],
        dim=1,
    )
    assert torch.allclose(torch.logsumexp(log_likelihoods_nk, dim=1), torch.zeros(len(mafs_n), dtype=torch.float64))


def test_without_a_normal_germline_log_likelihoods_are_the_carrier_part_of_the_tumors_emissions():
    # the emission is a mixture of hom refs, carriers and outliers, with weights (1 - w)(1 - q)^2, (1 - w) q (2 - q) and w
    model = segmentation_model()
    emission_model = model.tumor.emission_model
    alt_counts_n = torch.tensor([0.0, 5.0, 20.0, 39.0], dtype=torch.float64)
    depths_n = torch.full_like(alt_counts_n, 40.0)
    q_n = torch.tensor([0.02, 0.2, 0.5, 0.7], dtype=torch.float64)
    mafs_n = torch.tensor([0.5, 0.25, 0.45, 0.5], dtype=torch.float64)
    hom_ref_n, _, _ = emission_model.genotype_log_likelihoods(alt_counts_n, depths_n, q_n, mafs_n)
    carrier_n = tumor_only_germline_log_likelihoods(model, alt_counts_n, depths_n, torch.log(q_n), mafs_n)
    w = emission_model.outlier_weight
    expected_n = (1 - w) * ((1 - q_n) ** 2 * torch.exp(hom_ref_n) + q_n * (2 - q_n) * torch.exp(carrier_n))
    expected_n += w / (depths_n + 1)
    emission_n = model.tumor.emission_log_likelihoods(alt_counts_n, depths_n, q_n, mafs_n)
    assert torch.allclose(torch.exp(emission_n), expected_n)


def test_germline_log_likelihoods_of_no_reads_are_zero():
    model = SegmentationModel()
    log_afs = torch.log(torch.tensor([1e-12, 1e-6, 0.01, 0.3, 0.9, 1.0]))
    zeros, mafs = torch.zeros_like(log_afs), torch.full_like(log_afs, 0.4)
    result = model.germline_log_likelihoods(zeros, zeros, zeros, zeros, log_afs, mafs, mafs)
    # float32 lgamma cancellation in the beta binomials leaves errors of order 1e-5
    assert torch.allclose(result, zeros, atol=1e-4)


def test_tiny_allele_frequencies():
    # allele frequencies that underflow to 0 in float32, but whose logs are fine, give the het likelihood, since a
    # carrier of a very rare allele is almost surely het
    model = SegmentationModel()
    log_afs = torch.tensor([-50.0, -120.0, -300.0])
    alt_counts, depths, mafs = torch.full((3,), 20.0), torch.full((3,), 40.0), torch.full((3,), 0.5)
    result = tumor_only_germline_log_likelihoods(model, alt_counts, depths, log_afs, mafs)
    assert torch.all(torch.isfinite(result))
    _, het, _ = model.tumor.emission_model.genotype_log_likelihoods(alt_counts, depths, torch.zeros(3), mafs)
    # the het term sums the two phases, each of probability 1/2
    assert torch.allclose(result, het - math.log(2), atol=1e-5)


def test_maf_of_zero_or_one():
    # eg the tumor's copy-neutral loss of heterozygosity, where a het's reads are almost all ref or all alt
    model = SegmentationModel()
    log_afs = torch.log(torch.full((4,), 1e-3))
    mafs = torch.tensor([0.0, 1.0, 0.0, 1.0])
    alt_counts, depths = torch.tensor([0.0, 30.0, 15.0, 15.0]), torch.full((4,), 30.0)
    result = tumor_only_germline_log_likelihoods(model, alt_counts, depths, log_afs, mafs)
    assert torch.all(torch.isfinite(result))
    # all reads ref or all alt are far likelier than half of them alt
    assert torch.all(result[:2] > result[2:] + 10)


def test_contamination_log_likelihoods_are_normalized():
    # the distribution of the alt count of a hom ref sample contaminated by a carrier of the alt allele
    segmentation = sample_segmentation("tumor")
    allele_frequencies_n = torch.tensor([1e-6, 0.01, 0.3, 0.9, 1.0], dtype=torch.float64)
    alt_counts_nk, depths_nk = all_alt_counts(40, len(allele_frequencies_n))
    log_likelihoods_nk = segmentation.contamination_log_likelihoods(
        alt_counts_nk, depths_nk, torch.log(allele_frequencies_n)[:, None]
    )
    expected = torch.zeros(len(allele_frequencies_n), dtype=torch.float64)
    assert torch.allclose(torch.logsumexp(log_likelihoods_nk, dim=1), expected)


def test_contamination_log_likelihoods_are_gatks_single_contaminant_likelihoods():
    # without reference bias, those of GATK's contamination filter, 2q(1 - q) Binom(k | n, c/2) + q^2 Binom(k | n, c),
    # divided by the probability q(2 - q) that the contaminant carries the alt allele.  Sequencing error, a separate
    # call, has no part in them.
    contamination = 0.04
    segmentation = SampleSegmentation(emission_model=EmissionModel(contamination=contamination, error_rate=0.01))
    alt_counts_n = torch.tensor([0.0, 1.0, 3.0, 8.0, 20.0], dtype=torch.float64)
    depths_n = torch.tensor([50.0, 50.0, 100.0, 100.0, 60.0], dtype=torch.float64)
    q_n = torch.tensor([0.3, 1e-4, 0.05, 0.5, 0.9], dtype=torch.float64)
    result_n = segmentation.contamination_log_likelihoods(alt_counts_n, depths_n, torch.log(q_n))

    def binomial(alt_fraction: float) -> torch.Tensor:
        probs = torch.tensor(alt_fraction, dtype=torch.float64)
        return torch.exp(torch.distributions.Binomial(depths_n, probs=probs).log_prob(alt_counts_n))

    expected_n = 2 * q_n * (1 - q_n) * binomial(contamination / 2) + q_n**2 * binomial(contamination)
    assert torch.allclose(torch.exp(result_n), expected_n / (q_n * (2 - q_n)))


def test_contamination_by_a_carrier_of_a_rare_allele_is_het():
    # allele frequencies that underflow to 0 in float32, but whose logs are fine, give the likelihood of a het contaminant
    segmentation = SampleSegmentation(emission_model=EmissionModel(contamination=0.02))
    log_afs = torch.tensor([-50.0, -120.0, -300.0])
    alt_counts, depths = torch.full((3,), 2.0), torch.full((3,), 100.0)
    result = segmentation.contamination_log_likelihoods(alt_counts, depths, log_afs)
    assert torch.all(torch.isfinite(result))
    het, _ = segmentation.emission_model.contaminant_log_likelihoods(alt_counts, depths)
    assert torch.allclose(result, het, atol=1e-5)


def test_without_contamination_alt_reads_are_impossible():
    segmentation = SampleSegmentation()
    log_afs = torch.log(torch.tensor([1e-6, 0.1, 0.5]))
    alt_counts, depths = torch.tensor([0.0, 2.0, 10.0]), torch.full((3,), 30.0)
    result = segmentation.contamination_log_likelihoods(alt_counts, depths, log_afs)
    assert result[0].item() == pytest.approx(0, abs=1e-5)
    assert torch.all(result[1:] == constants.IMPOSSIBLE_LOG_PROB)


def test_maf_intervals():
    intervals = sample_segmentation("tumor").maf_intervals()

    def mafs(contig, position):
        return [interval.data for interval in intervals[contig][position]]

    assert mafs("chr1", 100) == mafs("chr1", 2000) == [0.3]
    # a gap has the MAF of the segment before it
    assert mafs("chr1", 2500) == [0.3]
    assert mafs("chr1", 3000) == [0.5]
    assert mafs("chr2", 15) == [0.1]
    # outside the segments
    assert mafs("chr1", 99) == mafs("chr1", 5001) == mafs("chr3", 1) == []


def test_save_and_load(tmp_path):
    model = SegmentationModel(sample_segmentation("tumor"), sample_segmentation("normal"), 0.03, 0.002)
    path = tmp_path / "segmentation_model.pt"
    model.save(path)
    loaded = SegmentationModel.load(path)

    assert (loaded.contamination, loaded.contamination_error) == (0.03, 0.002)
    for original, copy in ((model.tumor, loaded.tumor), (model.normal, loaded.normal)):
        assert copy.sample == original.sample and copy.jump_rate == original.jump_rate
        assert torch.equal(copy.maf_states_k, original.maf_states_k) and torch.equal(
            copy.maf_prior_k, original.maf_prior_k
        )

        def loci(segments):
            return [(segment.contig, segment.start, segment.end) for segment in segments]

        def mafs(segments):
            return np.array([[s.maf, s.secondary_maf, s.maf_low, s.maf_high] for s in segments])

        # the gap's NaN interval makes segments unequal to themselves, so the MAFs are compared with NaNs equal
        assert loci(copy.segments) == loci(original.segments)
        assert np.array_equal(mafs(copy.segments), mafs(original.segments), equal_nan=True)
        assert copy.emission_model.contamination == original.emission_model.contamination
        assert copy.emission_model.max_outlier_weight == original.emission_model.max_outlier_weight
        original_state, copied_state = original.emission_model.state_dict(), copy.emission_model.state_dict()
        assert original_state.keys() == copied_state.keys()
        assert all(torch.equal(original_state[key], copied_state[key]) for key in original_state)

    # the outlier weight at its cap, which the EmissionModel constructor would reject, is restored exactly
    assert loaded.tumor.emission_model.outlier_weight.item() == model.tumor.emission_model.outlier_weight.item()
    assert all(not parameter.requires_grad for parameter in loaded.parameters())


def test_default_model(tmp_path):
    # without segments, every MAF is 1/2, and the emission models have their default parameters
    model = SegmentationModel()
    assert all(len(intervals) == 0 for intervals in (model.tumor.maf_intervals(), model.normal.maf_intervals()))
    path = tmp_path / "default.pt"
    model.save(path)
    loaded = SegmentationModel.load(path)
    assert loaded.tumor.segments == [] and math.isnan(loaded.tumor.jump_rate)


def test_parameters_are_frozen():
    model = SegmentationModel(sample_segmentation("tumor"))
    parameters = list(model.parameters())
    assert parameters and all(not parameter.requires_grad for parameter in parameters)
