import torch
from torch import Tensor
from torch.distributions.binomial import Binomial

from permutect.architecture.spectra.overdispersed_binomial_mixture import (
    OverdispersedBinomialMixture,
)
from permutect.utils.enums import Variation


# given a discrete distribution of allele fractions between 0 and 1, and desired depths, generate alt counts,
# fit a BetaBinomialMixture, and compare moments of the underlying Beta mixture (without the binomial part) to
# those of the empirical allele fractions
def check_discrete_af_distribution(
    fractions_1d: Tensor,
    weights_1d: Tensor,
    training_depths_1d: Tensor,
    num_components: int = 10,
    num_epochs=1000,
):
    idx = weights_1d.multinomial(num_samples=len(training_depths_1d), replacement=True)
    empirical_fractions_1d = fractions_1d[idx]

    empirical_counts = Binomial(training_depths_1d, empirical_fractions_1d).sample().squeeze()
    # use SNV (index 0) as a dummy variant type for all samples
    types_b = torch.zeros(len(empirical_counts), dtype=torch.long)

    model = OverdispersedBinomialMixture(num_components=num_components)
    model.fit(
        num_epochs=num_epochs,
        types_b=types_b,
        depths_b=training_depths_1d,
        alt_counts_b=empirical_counts,
    )

    # moments E[x], E[ln(x)], E[x ln(x)] for SNV variant type (index 0)
    model_mean, model_log_mean, model_log_linear_mean = model.moments_of_underlying_beta_mixture(
        Variation.SNV
    )

    given_mean = torch.sum(weights_1d * fractions_1d)
    given_log_mean = torch.sum(weights_1d * torch.log(fractions_1d))
    given_log_linear_mean = torch.sum(weights_1d * fractions_1d * torch.log(fractions_1d))

    assert torch.abs(model_mean - given_mean).item() < 0.05
    assert torch.abs(model_log_mean - given_log_mean).item() < 0.3
    assert torch.abs(model_log_linear_mean - given_log_linear_mean).item() < 0.1


def test_single_component():
    torch.manual_seed(42)
    num_samples = 1000
    for fraction in [0.05, 0.1, 0.5, 0.75]:
        for depth in [10, 100, 1000]:
            depths = depth * torch.ones(num_samples).int()
            check_discrete_af_distribution(
                fractions_1d=Tensor([fraction]), weights_1d=Tensor([1.0]), training_depths_1d=depths
            )


def test_two_components():
    num_samples = 1000
    depth = 100
    depths = depth * torch.ones(num_samples).int()
    for fraction_pair in [(0.1, 0.9), (0.2, 0.4), (0.4, 0.8)]:
        for weight_pair in [(0.5, 0.5), (0.25, 0.75), (0.1, 0.9)]:
            check_discrete_af_distribution(
                fractions_1d=Tensor(fraction_pair),
                weights_1d=Tensor(weight_pair),
                training_depths_1d=depths,
            )


def test_three_components():
    num_samples = 1000
    depth = 100
    depths = depth * torch.ones(num_samples).int()
    fractions = (0.1, 0.4, 0.7)
    for unnormalized_weights in [(1, 1, 1), (1, 4, 1), (1, 5, 9)]:
        weights = Tensor(unnormalized_weights) / sum(unnormalized_weights)
        check_discrete_af_distribution(
            fractions_1d=Tensor(fractions), weights_1d=weights, training_depths_1d=depths
        )


def test_peak_over_background():
    num_samples = 1000
    depth = 100
    depths = depth * torch.ones(num_samples).int()
    fractions = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
    for unnormalized_weights in [(5, 1, 1, 1, 1, 1, 1, 1, 1), (1, 1, 1, 1, 5, 1, 1, 1, 1)]:
        weights = Tensor(unnormalized_weights) / sum(unnormalized_weights)
        check_discrete_af_distribution(
            fractions_1d=Tensor(fractions), weights_1d=weights, training_depths_1d=depths
        )
