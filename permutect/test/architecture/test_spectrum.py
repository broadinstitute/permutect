import matplotlib
import numpy as np
import pytest
import torch
from matplotlib import pyplot as plt

from permutect.architecture.spectra.artifact_spectra import ArtifactSpectra
from permutect.architecture.spectra.overdispersed_binomial_mixture import OverdispersedBinomialMixture
from permutect.architecture.spectra.somatic_spectrum import SomaticSpectrum
from permutect.architecture.spectra.spectrum import allele_fraction_grid
from permutect.architecture.spectra.spectrum import smeared_densities
from permutect.utils.enums import Variation

matplotlib.use("Agg")

# each spectrum, and the arguments of its spectrum_density_vs_fraction
SPECTRA = {
    "somatic": (lambda: SomaticSpectrum(num_components=5), ()),
    "artifact": (lambda: ArtifactSpectra(), (Variation.SNV, 30)),
    "beta binomial mixture": (lambda: OverdispersedBinomialMixture(num_components=3), (Variation.INSERTION, 30)),
    "binomial mixture": (lambda: OverdispersedBinomialMixture(num_components=3, mode="none"), (Variation.SNV, 30)),
}


@pytest.mark.parametrize("name", SPECTRA)
def test_plot_spectrum(name):
    # every copy of plot_spectrum used to call numpy() on densities that require grad
    make_spectrum, density_args = SPECTRA[name]
    spectrum = make_spectrum()
    fig, axis = spectrum.plot_spectrum("title", *density_args)

    fractions, densities = spectrum.spectrum_density_vs_fraction(*density_args)
    (line,) = axis.get_lines()
    assert np.allclose(line.get_xdata(), fractions.numpy())
    assert np.allclose(line.get_ydata(), densities.detach().numpy())
    plt.close(fig)


def test_smeared_densities():
    fractions = allele_fraction_grid()
    centers, weights = torch.tensor([0.2, 0.7]), torch.tensor([0.25, 0.75])
    densities = smeared_densities(centers, torch.log(weights), fractions)

    # each component is a narrow Gaussian holding its weight
    step = (fractions[1] - fractions[0]).item()
    assert torch.sum(densities).item() * step == pytest.approx(1, abs=0.01)
    assert torch.sum(densities[fractions < 0.45]).item() * step == pytest.approx(0.25, abs=0.01)


def test_fit_improves_the_likelihood():
    torch.manual_seed(0)
    depths = torch.full((500,), 50)
    alt_counts = torch.distributions.Binomial(50, torch.full((500,), 0.2)).sample()
    mafs = torch.full((500,), 0.49)
    spectrum = SomaticSpectrum(num_components=2)

    initial_log_lk = torch.mean(spectrum.forward(depths, alt_counts, mafs)).item()
    spectrum.fit(num_epochs=5, depths_b=depths, alt_counts_b=alt_counts, mafs_b=mafs)
    assert torch.mean(spectrum.forward(depths, alt_counts, mafs)).item() > initial_log_lk
