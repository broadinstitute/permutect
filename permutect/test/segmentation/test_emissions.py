import numpy as np
import pytest
import torch

from permutect.segmentation.emissions import EmissionModel
from permutect.segmentation.emissions import Pileups

MAFS_K = torch.tensor([0.0, 0.1, 0.3, 0.5], dtype=torch.float64)


def every_alt_count(depth: int, allele_frequency: float) -> Pileups:
    alt_counts_n = torch.arange(depth + 1, dtype=torch.float64)
    return Pileups(alt_counts_n, torch.full_like(alt_counts_n, depth), torch.full_like(alt_counts_n, allele_frequency))


def simulate_pileups(model: EmissionModel, mafs_n: np.ndarray, rng: np.random.Generator) -> Pileups:
    """sample sites from the emission model, given the MAF of each site"""
    num_sites = len(mafs_n)
    allele_frequencies_n = rng.uniform(0.02, 0.98, num_sites)
    depths_n = rng.poisson(60, num_sites)
    # the number of alt alleles, 0, 1 or 2 with the Hardy-Weinberg probabilities
    genotypes_n = rng.binomial(2, allele_frequencies_n)
    het_alt_fractions_n = np.where(rng.random(num_sites) < 0.5, mafs_n, 1 - mafs_n)
    sample_alt_fractions_n = np.where(genotypes_n == 1, het_alt_fractions_n, genotypes_n / 2)
    with torch.no_grad():
        means_n = model.alt_fractions(torch.from_numpy(sample_alt_fractions_n), torch.from_numpy(allele_frequencies_n))
        means_n = means_n.numpy()
        concentrations_n = np.where(genotypes_n == 1, model.het_concentration.item(), model.hom_concentration.item())
        outlier_weight = model.outlier_weight.item()
    alt_counts_n = rng.binomial(depths_n, rng.beta(means_n * concentrations_n, (1 - means_n) * concentrations_n))
    outliers_n = rng.random(num_sites) < outlier_weight
    alt_counts_n[outliers_n] = rng.integers(0, depths_n[outliers_n] + 1)
    arrays = (alt_counts_n, depths_n, allele_frequencies_n)
    return Pileups(*(torch.tensor(array, dtype=torch.float64) for array in arrays))


def test_alt_fractions():
    # the population allele frequency
    q = torch.tensor(0.3, dtype=torch.float64)
    with torch.no_grad():
        # a contaminant's reads have the population allele frequency on average
        contaminated = EmissionModel(contamination=0.05, error_rate=1e-9)
        assert contaminated.alt_fractions(0.0, q).item() == pytest.approx(0.05 * 0.3, abs=1e-6)
        assert contaminated.alt_fractions(1.0, q).item() == pytest.approx(1 - 0.05 * 0.7, abs=1e-6)
        assert contaminated.alt_fractions(0.4, q).item() == pytest.approx(0.95 * 0.4 + 0.05 * 0.3, abs=1e-6)

        # reference bias divides the odds of an alt read, which doesn't affect homs
        biased = EmissionModel(reference_bias=1.25, error_rate=1e-9)
        assert biased.alt_fractions(0.5, q).item() == pytest.approx(1 / 2.25, abs=1e-6)
        assert biased.alt_fractions(0.0, q).item() == pytest.approx(0.0, abs=1e-6)
        assert biased.alt_fractions(1.0, q).item() == pytest.approx(1.0, abs=1e-6)

        # sequencing errors turn ref reads into alt reads and vice versa
        erroneous = EmissionModel(error_rate=0.01)
        assert erroneous.alt_fractions(0.0, q).item() == pytest.approx(0.01, rel=1e-6)
        assert erroneous.alt_fractions(1.0, q).item() == pytest.approx(0.99, rel=1e-6)
        assert erroneous.alt_fractions(0.5, q).item() == pytest.approx(0.5, rel=1e-6)


@pytest.mark.parametrize("depth", [0, 1, 30])
@pytest.mark.parametrize("allele_frequency", [0.01, 0.3, 0.9])
def test_likelihoods_are_normalized(depth, allele_frequency):
    model = EmissionModel(
        contamination=0.03,
        error_rate=0.01,
        hom_concentration=50.0,
        het_concentration=80.0,
        reference_bias=1.15,
        outlier_weight=0.004,
    )
    with torch.no_grad():
        log_likelihoods_nk = model.log_likelihoods(every_alt_count(depth, allele_frequency), MAFS_K)
    assert torch.allclose(torch.logsumexp(log_likelihoods_nk, dim=0), torch.zeros(len(MAFS_K), dtype=torch.float64))


def test_ref_alt_symmetry():
    # without reference bias, swapping ref and alt reads and the population frequencies of the alleles changes nothing
    model = EmissionModel(contamination=0.03, error_rate=0.01, hom_concentration=50.0, het_concentration=80.0)
    pileups = every_alt_count(30, 0.2)
    mirrored = Pileups(30 - pileups.alt_counts_n, pileups.depths_n, 1 - pileups.allele_frequencies_n)
    with torch.no_grad():
        assert torch.allclose(model.log_likelihoods(pileups, MAFS_K), model.log_likelihoods(mirrored, MAFS_K))


@pytest.mark.parametrize("allele_frequency", [0.1, 0.3, 0.5])
def test_homozygous_sites_favor_loss_of_heterozygosity(allele_frequency):
    # with MAF 0 a het looks hom ref half of the time, so a site without alt reads has likelihood about
    # (1 - q)^2 + q(1 - q) = 1 - q, rather than the (1 - q)^2 of MAF 1/2
    model = EmissionModel(error_rate=1e-4, hom_concentration=1000.0, het_concentration=1000.0, outlier_weight=1e-6)
    no_alt_reads = Pileups(*(torch.tensor([value], dtype=torch.float64) for value in (0, 40, allele_frequency)))
    with torch.no_grad():
        log_likelihoods_k = model.log_likelihoods(no_alt_reads, torch.tensor([0.0, 0.5], dtype=torch.float64))[0]
    log_likelihood_ratio = (log_likelihoods_k[0] - log_likelihoods_k[1]).item()
    assert log_likelihood_ratio == pytest.approx(-np.log(1 - allele_frequency), rel=1e-5)


def test_fit_recovers_parameters():
    truth = EmissionModel(
        contamination=0.02,
        error_rate=0.005,
        hom_concentration=30.0,
        het_concentration=60.0,
        reference_bias=1.1,
        outlier_weight=0.003,
    )
    rng = np.random.default_rng(0)
    mafs_k = torch.tensor([0.2, 0.5], dtype=torch.float64)
    states_n = rng.integers(0, 2, 50000)
    pileups = simulate_pileups(truth, mafs_k.numpy()[states_n], rng)
    posteriors_nk = torch.nn.functional.one_hot(torch.from_numpy(states_n), 2).to(torch.float64)

    # starting from the default parameters, with the true contamination
    model = EmissionModel(contamination=0.02)
    model.fit(pileups, posteriors_nk, mafs_k, max_iterations=100)

    def objective(emission_model: EmissionModel) -> float:
        with torch.no_grad():
            return torch.sum(posteriors_nk * emission_model.log_likelihoods(pileups, mafs_k)).item()

    # the M-step maximizes the objective, so it beats the true parameters
    assert objective(model) >= objective(truth)
    for name, tolerance in (("error_rate", 0.1), ("hom_concentration", 0.1), ("het_concentration", 0.1)):
        assert getattr(model, name).item() == pytest.approx(getattr(truth, name).item(), rel=tolerance)
    assert model.reference_bias.item() == pytest.approx(truth.reference_bias.item(), rel=0.01)
    # only about 150 sites are outliers, and many of them look like plausible genotypes, so the outlier weight is
    # imprecise: different seeds give 0.0017 to 0.0049
    assert truth.outlier_weight.item() / 2 < model.outlier_weight.item() < 2 * truth.outlier_weight.item()


def test_fit_step_size():
    truth = EmissionModel(error_rate=0.005, hom_concentration=30.0, het_concentration=60.0, reference_bias=1.1)
    mafs_k = torch.tensor([0.5], dtype=torch.float64)
    pileups = simulate_pileups(truth, np.full(5000, 0.5), np.random.default_rng(1))
    posteriors_nk = torch.ones(5000, 1, dtype=torch.float64)
    initial, plain, overrelaxed = EmissionModel(), EmissionModel(), EmissionModel()
    plain.fit(pileups, posteriors_nk, mafs_k, max_iterations=5)
    overrelaxed.fit(pileups, posteriors_nk, mafs_k, max_iterations=5, step_size=2.0)
    # a step size of 2 doubles the change in the unconstrained parameters
    for start, once, twice in zip(initial.parameters(), plain.parameters(), overrelaxed.parameters()):
        assert torch.allclose(twice, start + 2 * (once - start))
