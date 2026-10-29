import math

import torch
from torch import Tensor
from torch import nn

from permutect.metrics.plotting import simple_plot
from permutect.misc_utils import backpropagate


def allele_fraction_grid() -> Tensor:
    # the allele fractions at which spectrum densities are plotted, a 1D tensor on CPU
    return torch.arange(0.01, 0.99, 0.001)


def smeared_densities(centers_k: Tensor, log_weights_k: Tensor, fractions_f: Tensor) -> Tensor:
    """
    For plotting a mixture of binomials, whose allele fraction density is a sum of delta functions, smear each delta
    function into a narrow Gaussian.

    :param centers_k: the allele fraction of each binomial component
    :param log_weights_k: the log weight of each component
    :param fractions_f: allele fractions at which to evaluate the density
    :return: the density at each fraction
    """
    gauss_k = torch.distributions.normal.Normal(centers_k, 0.01 * torch.ones_like(centers_k))
    log_densities_fk = gauss_k.log_prob(fractions_f.unsqueeze(dim=1))
    return torch.exp(torch.logsumexp(log_weights_k.view(1, -1) + log_densities_fk, dim=-1))


class Spectrum(nn.Module):
    """
    Base class for spectra, whose forward method gives the log likelihood of each datum in a batch of 1D tensors and
    whose spectrum_density_vs_fraction method gives the allele fractions and densities of a spectrum plot.
    """

    def fit_to_data(self, num_epochs: int, data_b: tuple, batch_size: int = 64):
        """
        Fit by maximizing the likelihood with minibatch gradient descent.

        :param data_b: the 1D tensors that forward takes, in order
        """
        optimizer = torch.optim.Adam(self.parameters())
        num_data = len(data_b[0])
        num_batches = math.ceil(num_data / batch_size)

        for epoch in range(num_epochs):
            for batch in range(num_batches):
                batch_slice = slice(batch * batch_size, min((batch + 1) * batch_size, num_data))
                loss = -torch.mean(self.forward(*(tensor_b[batch_slice] for tensor_b in data_b)))
                backpropagate(optimizer, loss)

    def plot_spectrum(self, title: str, *density_args):
        """
        Plot the density vs allele fraction.  density_args are the arguments of spectrum_density_vs_fraction.
        """
        fractions_f, densities_f = self.spectrum_density_vs_fraction(*density_args)
        fractions_f, densities_f = fractions_f.detach().cpu().numpy(), densities_f.detach().cpu().numpy()
        return simple_plot([(fractions_f, densities_f, " ")], "AF", "density", title)
