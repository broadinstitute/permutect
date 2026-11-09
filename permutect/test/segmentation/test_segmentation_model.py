import math

import numpy as np
import pytest
import torch

from permutect.segmentation.emissions import EmissionModel
from permutect.segmentation.emissions import Pileups
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


def test_germline_log_likelihoods_are_normalized():
    # the distribution of the alt count of a carrier of the alt allele, so the likelihoods of every alt count sum to 1
    segmentation = sample_segmentation("tumor")
    allele_frequencies_n = torch.tensor([1e-6, 0.01, 0.3, 0.5, 0.9, 1.0], dtype=torch.float64)
    mafs_n = torch.tensor([0.5, 0.4, 0.2, 0.0, 0.5, 0.3], dtype=torch.float64)
    alt_counts_nk, depths_nk = all_alt_counts(40, len(mafs_n))
    log_likelihoods_nk = segmentation.germline_log_likelihoods(
        alt_counts_nk, depths_nk, torch.log(allele_frequencies_n)[:, None], mafs_n[:, None]
    )
    assert torch.allclose(torch.logsumexp(log_likelihoods_nk, dim=1), torch.zeros(len(mafs_n), dtype=torch.float64))


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


def test_germline_log_likelihoods_are_the_carrier_part_of_the_emissions():
    # the emission is a mixture of hom refs, carriers and outliers, with weights (1 - w)(1 - q)^2, (1 - w) q (2 - q) and w
    segmentation = sample_segmentation("tumor")
    model = segmentation.emission_model
    alt_counts_n = torch.tensor([0.0, 5.0, 20.0, 39.0], dtype=torch.float64)
    depths_n = torch.full_like(alt_counts_n, 40.0)
    q_n = torch.tensor([0.02, 0.2, 0.5, 0.7], dtype=torch.float64)
    mafs_n = torch.tensor([0.5, 0.25, 0.45, 0.5], dtype=torch.float64)
    hom_ref_n, _, _ = model.genotype_log_likelihoods(alt_counts_n, depths_n, q_n, mafs_n)
    carrier_n = segmentation.germline_log_likelihoods(alt_counts_n, depths_n, torch.log(q_n), mafs_n)
    w = model.outlier_weight
    expected_n = (1 - w) * ((1 - q_n) ** 2 * torch.exp(hom_ref_n) + q_n * (2 - q_n) * torch.exp(carrier_n))
    expected_n += w / (depths_n + 1)
    emission_n = segmentation.emission_log_likelihoods(alt_counts_n, depths_n, q_n, mafs_n)
    assert torch.allclose(torch.exp(emission_n), expected_n)


def test_germline_log_likelihoods_of_no_reads_are_zero():
    segmentation = SampleSegmentation()
    log_afs = torch.log(torch.tensor([1e-12, 1e-6, 0.01, 0.3, 0.9, 1.0]))
    zeros = torch.zeros_like(log_afs)
    result = segmentation.germline_log_likelihoods(zeros, zeros, log_afs, torch.full_like(log_afs, 0.4))
    # float32 lgamma cancellation in the beta binomials leaves errors of order 1e-5
    assert torch.allclose(result, zeros, atol=1e-4)


def test_tiny_allele_frequencies():
    # allele frequencies that underflow to 0 in float32, but whose logs are fine, give the het likelihood, since a
    # carrier of a very rare allele is almost surely het
    segmentation = SampleSegmentation()
    log_afs = torch.tensor([-50.0, -120.0, -300.0])
    alt_counts, depths, mafs = torch.full((3,), 20.0), torch.full((3,), 40.0), torch.full((3,), 0.5)
    result = segmentation.germline_log_likelihoods(alt_counts, depths, log_afs, mafs)
    assert torch.all(torch.isfinite(result))
    _, het, _ = segmentation.emission_model.genotype_log_likelihoods(alt_counts, depths, torch.zeros(3), mafs)
    # the het term sums the two phases, each of probability 1/2
    assert torch.allclose(result, het - math.log(2), atol=1e-5)


def test_maf_of_zero_or_one():
    # eg copy-neutral loss of heterozygosity, where a het's reads are almost all ref or all alt
    segmentation = SampleSegmentation()
    log_afs = torch.log(torch.full((4,), 1e-3))
    mafs = torch.tensor([0.0, 1.0, 0.0, 1.0])
    alt_counts, depths = torch.tensor([0.0, 30.0, 15.0, 15.0]), torch.full((4,), 30.0)
    result = segmentation.germline_log_likelihoods(alt_counts, depths, log_afs, mafs)
    assert torch.all(torch.isfinite(result))
    # all reads ref or all alt are far likelier than half of them alt
    assert torch.all(result[:2] > result[2:] + 10)


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
