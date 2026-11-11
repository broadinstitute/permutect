import math

import numpy as np
import torch

from permutect.segmentation.emissions import EmissionModel
from permutect.segmentation.indel_fitting import GermlineIndels
from permutect.segmentation.indel_fitting import fit_indel_parameters
from permutect.segmentation.indel_fitting import make_bins
from permutect.segmentation.indel_fitting import merge_bins
from permutect.segmentation.indel_parameters import DINUCLEOTIDE
from permutect.segmentation.indel_parameters import HOMOPOLYMER
from permutect.segmentation.indel_parameters import LONGER_UNIT
from permutect.segmentation.indel_parameters import NOT_IN_STR
from permutect.segmentation.indel_parameters import IndelParameters

# a steep shortfall and falling concentrations in homopolymers, and a constant shortfall and low concentration outside STRs
TRUE_PARAMETERS = IndelParameters(
    shortfall_scales=[0.1, 0.4, 0.0, 0.0],
    shortfall_offsets=[0.0, -7.0, 0.0, 0.0],
    shortfall_slopes=[0.0, 0.5, 0.0, 0.0],
    concentration_upper_bounds=[[0], [6, 10, 14, 18], [4], [6]],
    log_concentrations=[
        [math.log(10)],
        [math.log(40), math.log(25), math.log(10), math.log(4)],
        [math.log(50)],
        [math.log(50)],
    ],
)
HOMOPOLYMER_LENGTHS = (6, 10, 14, 18)


def simulate(
    parameters: IndelParameters, emission_model: EmissionModel, num_per_group: int, seed: int
) -> GermlineIndels:
    """
    Germline indels outside STRs and in homopolymers of each length, from the germline model's own generative model: a
    carrier of allele frequency q is a hom alt with probability q / (2 - q), and otherwise a het with the alt allele on
    the minor or the major haplotype, in a balanced region, MAF 1/2, or one with a single haplotype, MAF 0
    """
    rng = np.random.default_rng(seed)
    groups = [(0, 0)] + [(1, length) for length in HOMOPOLYMER_LENGTHS]
    unit_lengths = np.repeat([unit for unit, _ in groups], num_per_group).astype(np.float64)
    lengths = np.repeat([length for _, length in groups], num_per_group).astype(np.float64)
    num_sites = len(lengths)
    allele_frequencies = rng.uniform(0.05, 0.5, num_sites)
    mafs = rng.choice([0.0, 0.5], num_sites)
    hom_alt = rng.random(num_sites) < allele_frequencies / (2 - allele_frequencies)
    on_minor = rng.random(num_sites) < 0.5
    sample_alt_fractions = np.where(hom_alt, 1.0, np.where(on_minor, mafs, 1 - mafs))

    t = lambda values: torch.tensor(values, dtype=torch.float64)
    with torch.no_grad():
        means = emission_model.alt_fractions(t(sample_alt_fractions), t(allele_frequencies))
        means = (means * parameters.alt_fraction_factors(t(unit_lengths), t(lengths))).numpy()
        concentrations = parameters.concentrations(t(unit_lengths), t(lengths)).numpy()
    depths = np.full(num_sites, 80.0)
    alt_counts = rng.binomial(depths.astype(int), rng.beta(means * concentrations, (1 - means) * concentrations))
    return GermlineIndels(
        emission_model,
        t(alt_counts.astype(np.float64)),
        t(depths),
        t(np.log(allele_frequencies)),
        t(mafs),
        t(unit_lengths),
        t(lengths),
    )


def test_make_bins():
    # consecutive lengths merge until each bin has at least 3 indels, 1, 2-3 and 4-6, and the one indel of length 9,
    # too few for a bin of its own, joins the last bin
    lengths = np.array([1, 1, 1, 2, 3, 3, 4, 5, 6, 6, 6, 6, 9])
    assert make_bins(lengths, min_indels=3) == [1, 3, 9]
    assert make_bins(np.array([7, 7]), min_indels=3) == [7]


def test_merge_bins():
    bounds, logs = merge_bins(
        torch.tensor([2.0, 4.0, 6.0, 8.0]), torch.tensor([3.0, 2.995, 2.0, 1.999]), tolerance=0.01
    )
    assert bounds.tolist() == [4.0, 8.0] and logs.tolist() == [3.0, 2.0]


def test_fit_recovers_simulated_parameters():
    emission_model = EmissionModel(contamination=0.0, error_rate=1e-3, reference_bias=1.05)
    samples = [simulate(TRUE_PARAMETERS, emission_model, num_per_group=300, seed=seed) for seed in (0, 1)]
    fitted = fit_indel_parameters(samples)

    unit_lengths = torch.tensor([0.0] + [1.0] * len(HOMOPOLYMER_LENGTHS), dtype=torch.float64)
    lengths = torch.tensor([0.0, *HOMOPOLYMER_LENGTHS], dtype=torch.float64)
    # the longest homopolymers' concentration of 4 leaves their alt fractions noisy, and their factor 0.03 off
    true_factors = TRUE_PARAMETERS.alt_fraction_factors(unit_lengths, lengths).double()
    fitted_factors = fitted.alt_fraction_factors(unit_lengths, lengths)
    assert torch.allclose(fitted_factors, true_factors, atol=0.05), (fitted_factors, true_factors)

    true_logs = torch.log(TRUE_PARAMETERS.concentrations(unit_lengths, lengths)).double()
    fitted_logs = torch.log(fitted.concentrations(unit_lengths, lengths))
    assert torch.allclose(fitted_logs, true_logs, atol=0.4), (torch.exp(fitted_logs), torch.exp(true_logs))

    # the homopolymers' concentrations can't increase with length
    homopolymer_logs = fitted.log_concentrations()[HOMOPOLYMER]
    assert torch.all(torch.diff(homopolymer_logs) <= 0)


def test_classes_without_indels_keep_the_initial_parameters():
    emission_model = EmissionModel()
    samples = [simulate(TRUE_PARAMETERS, emission_model, num_per_group=20, seed=2)]
    initial = IndelParameters.default()
    fitted = fit_indel_parameters(samples, initial=initial)
    for k in (DINUCLEOTIDE, LONGER_UNIT):
        assert fitted.shortfall_scales[k].item() == initial.shortfall_scales[k].item()
        assert fitted.concentration_upper_bounds()[k].tolist() == initial.concentration_upper_bounds()[k].tolist()
        assert fitted.log_concentrations()[k].tolist() == initial.log_concentrations()[k].tolist()
    # while the classes with indels are fit, outside STRs to a single bin
    assert fitted.concentration_upper_bounds()[NOT_IN_STR].tolist() == [0.0]
