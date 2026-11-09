import itertools

import numpy as np
import pytest

from permutect.segmentation.hmm import MIN_JUMP_RATE
from permutect.segmentation.hmm import fit_jump_rate
from permutect.segmentation.hmm import fit_prior
from permutect.segmentation.hmm import forward_backward
from permutect.segmentation.hmm import jump_probabilities
from permutect.segmentation.hmm import overrelax_prior
from permutect.segmentation.hmm import site_distances
from permutect.segmentation.hmm import viterbi

# chains not sorted by length, including one with a single site
LENGTHS_C = np.array([3, 1, 5, 4])
STARTS_C = np.cumsum(LENGTHS_C) - LENGTHS_C


def random_hmm(seed: int, prune_state: bool):
    rng = np.random.default_rng(seed)
    num_sites, num_states = int(np.sum(LENGTHS_C)), 3
    log_emissions_nk = 2 * rng.standard_normal((num_sites, num_states))
    jump_probs_n = rng.uniform(0.05, 0.95, num_sites)
    prior_k = rng.dirichlet(np.ones(num_states))
    if prune_state:
        prior_k[1] = 0
        prior_k /= np.sum(prior_k)
    return log_emissions_nk, jump_probs_n, prior_k


def brute_force(log_emissions_nk, jump_probs_n, prior_k):
    """
    Forward-backward's posteriors and log likelihood and Viterbi's path, by enumerating every path of each chain with
    the dense transition probabilities P(k at n | j at n - 1) = (1 - jump_n) [j == k] + jump_n prior_k
    """
    num_states = len(prior_k)
    state_posteriors_nk = np.zeros_like(log_emissions_nk)
    jump_posteriors_n = np.zeros(len(jump_probs_n))
    expected_draws_k = np.zeros(num_states)
    log_likelihood = 0.0
    best_path_n = np.zeros(len(jump_probs_n), dtype=np.int64)
    for start, length in zip(STARTS_C, LENGTHS_C):
        paths = list(itertools.product(range(num_states), repeat=length))
        # P(path, data) and P(jump at each site | path) for each path
        probs, jumps_given_paths = [], []
        for path in paths:
            prob, jumps_given_path = prior_k[path[0]] * np.exp(log_emissions_nk[start, path[0]]), [1.0]
            for t in range(1, length):
                stay = (1 - jump_probs_n[start + t]) * (path[t] == path[t - 1])
                jump = jump_probs_n[start + t] * prior_k[path[t]]
                prob *= (stay + jump) * np.exp(log_emissions_nk[start + t, path[t]])
                jumps_given_path.append(jump / (stay + jump) if stay + jump > 0 else 0.0)
            probs.append(prob)
            jumps_given_paths.append(jumps_given_path)

        total = sum(probs)
        log_likelihood += np.log(total)
        for path, prob, jumps_given_path in zip(paths, probs, jumps_given_paths):
            for t in range(length):
                state_posteriors_nk[start + t, path[t]] += prob / total
                jump_posteriors_n[start + t] += prob / total * jumps_given_path[t]
                expected_draws_k[path[t]] += prob / total * jumps_given_path[t]
        best_path_n[start : start + length] = paths[int(np.argmax(probs))]
    return state_posteriors_nk, jump_posteriors_n, expected_draws_k, log_likelihood, best_path_n


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("prune_state", [False, True])
def test_forward_backward_matches_brute_force(seed, prune_state):
    log_emissions_nk, jump_probs_n, prior_k = random_hmm(seed, prune_state)
    state_posteriors_nk, jump_posteriors_n, expected_draws_k, log_likelihood, _ = brute_force(
        log_emissions_nk, jump_probs_n, prior_k
    )

    # the first site of each chain is drawn from the prior whatever its jump probability
    jump_probs_n[STARTS_C] = 0.5
    posteriors = forward_backward(log_emissions_nk, jump_probs_n, prior_k, LENGTHS_C)
    assert np.allclose(posteriors.state_posteriors_nk, state_posteriors_nk, rtol=0, atol=1e-12)
    assert np.allclose(posteriors.jump_posteriors_n, jump_posteriors_n, rtol=0, atol=1e-12)
    assert np.allclose(posteriors.expected_draws_k, expected_draws_k, rtol=0, atol=1e-12)
    assert posteriors.log_likelihood == pytest.approx(log_likelihood, rel=0, abs=1e-12)
    if prune_state:
        assert np.all(posteriors.state_posteriors_nk[:, 1] == 0)

    # shifting the log emissions of each site, even far enough that their exponentials underflow, shifts the log
    # likelihood and leaves the posteriors alone
    shifts_n = -1000.0 * np.arange(1, len(jump_probs_n) + 1)
    shifted = forward_backward(log_emissions_nk + shifts_n[:, None], jump_probs_n, prior_k, LENGTHS_C)
    assert np.allclose(shifted.state_posteriors_nk, state_posteriors_nk, rtol=0, atol=1e-12)
    assert shifted.log_likelihood == pytest.approx(log_likelihood + np.sum(shifts_n), rel=0, abs=1e-9)


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("prune_state", [False, True])
def test_viterbi_matches_brute_force(seed, prune_state):
    log_emissions_nk, jump_probs_n, prior_k = random_hmm(seed, prune_state)
    best_path_n = brute_force(log_emissions_nk, jump_probs_n, prior_k)[4]
    jump_probs_n[STARTS_C] = 0.5
    assert np.array_equal(viterbi(log_emissions_nk, jump_probs_n, prior_k, LENGTHS_C), best_path_n)


def test_jump_probabilities():
    distances_n = site_distances(np.array([100, 150, 400, 7, 20]), np.array([3, 2]))
    assert np.array_equal(distances_n, [np.inf, 50, 250, np.inf, 13])
    # the first site of each chain always jumps, ie is drawn from the prior
    expected_n = [1, 1 - np.exp(-0.05), 1 - np.exp(-0.25), 1, 1 - np.exp(-0.013)]
    assert np.allclose(jump_probabilities(distances_n, 1e-3), expected_n, rtol=1e-12, atol=0)
    with pytest.raises(AssertionError):
        site_distances(np.array([100, 100]), np.array([2]))


def test_fit_jump_rate():
    rng = np.random.default_rng(0)
    distances_n = site_distances(np.cumsum(rng.integers(1, 2000, 10000)), np.array([4000, 6000]))
    # if the posterior of each jump is its probability at some rate, the expected log likelihood peaks at that rate
    jump_posteriors_n = jump_probabilities(distances_n, 1e-5)
    assert fit_jump_rate(jump_posteriors_n, distances_n, max_jump_rate=1e-3) == pytest.approx(1e-5, rel=1e-9)
    # unless that's above the maximum rate
    assert fit_jump_rate(jump_posteriors_n, distances_n, max_jump_rate=1e-6) == 1e-6
    # without any jumps except at chain starts, the rate is the minimum
    no_jumps_n = np.where(np.isfinite(distances_n), 0.0, 1.0)
    assert fit_jump_rate(no_jumps_n, distances_n, max_jump_rate=1e-3) == MIN_JUMP_RATE


def test_fit_prior():
    # the mode of the Dirichlet posterior
    assert np.allclose(fit_prior(np.array([3.0, 1.0, 0.0]), concentration=2.0), np.array([4, 2, 1]) / 7)
    # a concentration below 1 prunes states with fewer than 1 - concentration expected draws, whose draws go to the
    # nearest remaining state, or of two equally near, the one with more draws
    assert np.allclose(fit_prior(np.array([6.0, 0.5, 3.0]), concentration=0.1), np.array([5.6, 0, 2.1]) / 7.7)
    # a segment whose MAF is between two states splits its draw between them, and keeps one of them.  The pruned
    # states stay pruned.
    draws_k = np.array([0.0, 0.6, 0.4, 0.0, 2.0])
    assert np.allclose(fit_prior(draws_k, concentration=0.1), np.array([0, 0.1, 0, 0, 1.1]) / 1.2)
    # if every state would be pruned, the one with the most draws is kept
    assert np.array_equal(fit_prior(np.array([0.2, 0.5, 0.3]), concentration=0.1), [0, 1, 0])


def test_overrelax_prior():
    prior_k = np.array([0.4, 0.3, 0.2, 0.1, 0.1])
    # the M-step pruned the last state
    fitted_prior_k = np.array([0.5, 0.3, 0.15, 0.05, 0.0])
    # a step size of 1 gives the M-step's prior
    assert np.array_equal(overrelax_prior(prior_k, fitted_prior_k, 1.0), fitted_prior_k)
    # a step size of 2 moves the log prior twice as far, and the pruned state stays pruned
    extrapolated_k = np.array([0.625, 0.3, 0.1125, 0.025, 0.0])
    assert np.allclose(overrelax_prior(prior_k, fitted_prior_k, 2.0), extrapolated_k / np.sum(extrapolated_k))
