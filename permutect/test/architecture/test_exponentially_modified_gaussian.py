import numpy as np
import pytest
import torch
from scipy import stats

from permutect.architecture.exponentially_modified_gaussian import ExponentiallyModifiedGaussian
from permutect.architecture.exponentially_modified_gaussian import ExponentiallyModifiedGaussianDistribution

# (mu, sigma, lambda) from nearly exponential (sigma * lambda << 1) to nearly Gaussian (sigma * lambda >> 1)
PARAMETERS = [
    (0.0, 0.01, 0.01),
    (0.5, 0.05, 10.0),
    (-1.0, 2.0, 0.2),
    (0.0, 1.0, 1.0),
    (2.0, 0.5, 3.0),
    (3.0, 10.0, 10.0),
]

# typical of trained models, which use float32
FLOAT32_PARAMETERS = [(0.0, 1.0, 1.0), (2.0, 0.5, 3.0), (1.0, 0.3, 0.5)]


def scipy_emg(mu: float, sigma: float, lambda_: float):
    # scipy's exponnorm has shape parameter K = 1 / (sigma * lambda), location mu, and scale sigma
    return stats.exponnorm(K=1 / (sigma * lambda_), loc=mu, scale=sigma)


def make_emg(parameters, dtype=torch.float64, validate_args=None) -> ExponentiallyModifiedGaussianDistribution:
    # parameters is a (mu, sigma, lambda) tuple for one distribution, or a list of tuples for a batch
    mu, sigma, lambda_ = torch.tensor(parameters, dtype=dtype).unbind(dim=-1)
    return ExponentiallyModifiedGaussianDistribution(mu, sigma, lambda_, validate_args=validate_args)


def tails_grid(mu: float, sigma: float, lambda_: float) -> np.ndarray:
    # from far in the Gaussian left tail, where logerfc switches to its asymptotic expansion, to far in the exponential
    # right tail
    return np.linspace(mu - 15 * sigma, mu + 15 * sigma + 30 / lambda_, num=1000)


@pytest.mark.parametrize("parameters", PARAMETERS)
def test_log_prob_matches_scipy(parameters):
    x = tails_grid(*parameters)
    actual = make_emg(parameters).log_prob(torch.from_numpy(x)).numpy()
    np.testing.assert_allclose(actual, scipy_emg(*parameters).logpdf(x), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("parameters", FLOAT32_PARAMETERS)
def test_float32_log_prob_matches_scipy(parameters):
    x = tails_grid(*parameters)
    actual = make_emg(parameters, dtype=torch.float32).log_prob(torch.from_numpy(x).float()).numpy()
    np.testing.assert_allclose(actual, scipy_emg(*parameters).logpdf(x), rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("parameters", PARAMETERS)
def test_log_prob_gradients(parameters):
    # gradients with respect to the parameters train the model, and gradients with respect to x train the embeddings
    x = torch.from_numpy(tails_grid(*parameters)[::25]).requires_grad_()
    mu, sigma, lambda_ = (torch.tensor(value, dtype=torch.float64, requires_grad=True) for value in parameters)
    log_prob = lambda x, mu, sigma, lambda_: ExponentiallyModifiedGaussianDistribution(mu, sigma, lambda_).log_prob(x)
    assert torch.autograd.gradcheck(log_prob, (x, mu, sigma, lambda_))


def test_mean_and_variance_match_scipy():
    emg = make_emg(PARAMETERS)
    np.testing.assert_allclose(emg.mean.numpy(), [scipy_emg(*parameters).mean() for parameters in PARAMETERS])
    np.testing.assert_allclose(emg.variance.numpy(), [scipy_emg(*parameters).var() for parameters in PARAMETERS])


def test_samples_follow_distribution():
    torch.manual_seed(0)
    num_samples = 20000
    samples_nk = make_emg(PARAMETERS).sample((num_samples,))
    assert samples_nk.shape == (num_samples, len(PARAMETERS))

    # each distribution in the batch is sampled independently of the others
    for k, parameters in enumerate(PARAMETERS):
        assert stats.kstest(samples_nk[:, k].numpy(), scipy_emg(*parameters).cdf).pvalue > 0.001, parameters
    assert abs(stats.spearmanr(samples_nk[:, 0], samples_nk[:, 1]).statistic) < 0.03


def test_rsample_gradients():
    # the gradients of the mean of reparametrized samples are estimates of those of the mean mu + 1/lambda
    torch.manual_seed(0)
    mu, sigma, lambda_ = (torch.tensor(value, dtype=torch.float64, requires_grad=True) for value in (0.5, 1.5, 2.0))
    ExponentiallyModifiedGaussianDistribution(mu, sigma, lambda_).rsample((100000,)).mean().backward()
    assert mu.grad.item() == pytest.approx(1.0)
    assert sigma.grad.item() == pytest.approx(0.0, abs=0.02)
    assert lambda_.grad.item() == pytest.approx(-1 / 2.0**2, rel=0.02)


def test_shapes_and_broadcasting():
    # numbers broadcast against tensors
    emg = ExponentiallyModifiedGaussianDistribution(mu=torch.zeros(3), sigma=1.0, lambda_=torch.tensor([1.0, 2.0, 3.0]))
    assert emg.batch_shape == (3,)
    assert emg.event_shape == ()
    assert emg.sample((5, 2)).shape == (5, 2, 3)

    # log likelihoods of each of 7 values under each of the 3 distributions, as FeatureClustering needs
    assert emg.log_prob(torch.zeros(7, 1)).shape == (7, 3)

    expanded = emg.expand((4, 3))
    assert expanded.batch_shape == (4, 3)
    assert expanded.sample().shape == (4, 3)
    assert torch.equal(expanded.log_prob(torch.ones(4, 3))[2], emg.log_prob(torch.ones(3)))


@pytest.mark.parametrize("sigma, lambda_", [(0.0, 1.0), (-1.0, 1.0), (1.0, 0.0), (1.0, -2.0)])
def test_invalid_parameters_are_rejected(sigma, lambda_):
    with pytest.raises(ValueError):
        make_emg((0.0, sigma, lambda_), validate_args=True)


def test_module_distribution():
    module = ExponentiallyModifiedGaussian(num_distributions=3)
    emg = module.distribution()
    assert emg.batch_shape == (3,)
    assert torch.equal(emg.mu, module.mu_k)
    assert torch.equal(emg.sigma, module.sigma_k)
    assert torch.equal(emg.lambda_, module.lambda_k)

    # gradients reach the parameters underneath the parametrizations
    emg.log_prob(torch.randn(5, 3)).sum().backward()
    for name, parameter in module.named_parameters():
        assert parameter.grad is not None and parameter.grad.abs().sum() > 0, name
