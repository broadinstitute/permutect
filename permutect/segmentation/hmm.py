"""
The hidden Markov model of allele fraction segmentation, apart from its emissions: forward-backward, Viterbi, and the
M-step updates of the transition parameters.

Sites come in chains, one per contig.  The hidden state of each site is one of K states, in segmentation a grid of minor
allele fractions.  Between consecutive sites n - 1 and n of a chain the state jumps with probability jump_n, which grows
with the distance between the sites: breakpoints are a Poisson process with rate lambda per base, so
jump_n = 1 - exp(-lambda * distance_n).  A jump redraws the state from a prior pi shared by all states, so it may land
back on the same state; without a jump the state stays the same.  The first site of each chain is drawn from pi.

The transition matrix (1 - jump_n) I + jump_n 1 pi^T is diagonal plus rank one, so the recursions never build it and
cost O(K) per site rather than O(K^2).  Chains are independent, so each recursion steps through all of them at once:
step t handles the t-th site of every chain with more than t sites.

Arrays are indexed by site n, over the concatenated sites of all chains, by chain c, and by state k.
"""

import math
from dataclasses import dataclass

import numpy as np

# lower bound of the fitted jump rate, in jumps per base, so that jumps never become impossible
MIN_JUMP_RATE = 1e-12


@dataclass
class Posteriors:
    # P(state of site n = k | data)
    state_posteriors_nk: np.ndarray
    # P(the state jumps between site n - 1 and site n | data), which is 1 at the first site of each chain
    jump_posteriors_n: np.ndarray
    # expected number of draws from the prior, at chain starts and jumps, that land in state k
    expected_draws_k: np.ndarray
    log_likelihood: float


class _ChainSteps:
    """
    The sites of each step of a recursion over chains.  Sorting the chains by decreasing length makes the chains with
    more than t sites a prefix of the sorted order, so their t-th sites are sorted_starts_c[:num_chains_t[t]] + t.
    """

    def __init__(self, lengths_c):
        lengths_c = np.asarray(lengths_c, dtype=np.int64)
        assert lengths_c.ndim == 1 and len(lengths_c) > 0 and np.all(lengths_c > 0)
        self.num_sites = int(np.sum(lengths_c))
        self.starts_c = np.cumsum(lengths_c) - lengths_c
        order_c = np.argsort(-lengths_c, kind="stable")
        self.sorted_starts_c = self.starts_c[order_c]
        self.sorted_lengths_c = lengths_c[order_c]
        self.num_steps = int(self.sorted_lengths_c[0])
        # the number of chains with more than t sites
        self.num_chains_t = np.searchsorted(-self.sorted_lengths_c, -np.arange(self.num_steps), side="left")

    def sites(self, t: int) -> np.ndarray:
        """the t-th site of each chain with more than t sites, in sorted chain order"""
        return self.sorted_starts_c[: self.num_chains_t[t]] + t


def site_distances(positions_n: np.ndarray, lengths_c) -> np.ndarray:
    """
    The distance from each site to the previous site of its chain, and infinity at the first site of each chain, whose
    jump probability is therefore 1.  Positions must increase within each chain.
    """
    lengths_c = np.asarray(lengths_c, dtype=np.int64)
    assert len(positions_n) == np.sum(lengths_c)
    distances_n = np.empty(len(positions_n), dtype=np.float64)
    distances_n[1:] = np.diff(positions_n)
    distances_n[np.cumsum(lengths_c) - lengths_c] = np.inf
    assert np.all(distances_n > 0), "positions must increase within each chain"
    return distances_n


def jump_probabilities(distances_n: np.ndarray, jump_rate: float) -> np.ndarray:
    """the probability of at least one breakpoint of a Poisson process with rate jump_rate per base over each distance"""
    return -np.expm1(-jump_rate * distances_n)


def forward_backward(
    log_emissions_nk: np.ndarray, jump_probs_n: np.ndarray, prior_k: np.ndarray, lengths_c
) -> Posteriors:
    """
    The E-step: posteriors of the states and jumps given the data.  The jump probabilities of the first site of each
    chain are ignored, because its state is drawn from the prior.
    """
    steps = _ChainSteps(lengths_c)
    assert log_emissions_nk.shape == (steps.num_sites, len(prior_k)) and jump_probs_n.shape == (steps.num_sites,)

    # emissions relative to each site's largest, so that they don't underflow; the log likelihood adds back the shifts
    max_log_emissions_n = np.max(log_emissions_nk, axis=1)
    emissions_nk = np.exp(log_emissions_nk - max_log_emissions_n[:, None])
    jump_n = jump_probs_n.copy()
    jump_n[steps.starts_c] = 1.0
    stay_n = 1 - jump_n

    # alpha_nk = P(state k at site n | data up to site n) and normalizers_n = P(data at site n | data before site n),
    # both within the chain and with the emissions shifted
    alpha_nk = np.empty_like(emissions_nk)
    normalizers_n = np.empty(steps.num_sites)
    for t in range(steps.num_steps):
        n = steps.sites(t)
        predicted_nk = prior_k if t == 0 else stay_n[n, None] * alpha_nk[n - 1] + jump_n[n, None] * prior_k
        unnormalized_nk = emissions_nk[n] * predicted_nk
        normalizers_n[n] = np.sum(unnormalized_nk, axis=1)
        alpha_nk[n] = unnormalized_nk / normalizers_n[n, None]

    # beta_nk = P(data after site n | state k at site n) / P(data after site n | data up to site n), within the chain
    beta_nk = np.ones_like(emissions_nk)
    for t in range(steps.num_steps - 1, 0, -1):
        n = steps.sites(t)
        emitted_nk = emissions_nk[n] * beta_nk[n]
        after_jump_n = emitted_nk @ prior_k
        beta_nk[n - 1] = (stay_n[n, None] * emitted_nk + (jump_n[n] * after_jump_n)[:, None]) / normalizers_n[n, None]

    # P(jump at site n that lands in state k | data), summing over the state at site n - 1, whose alpha sums to 1.  At
    # the first site of each chain, where jump_n = 1, this is the state posterior.
    draws_nk = jump_n[:, None] * prior_k * emissions_nk * beta_nk / normalizers_n[:, None]
    return Posteriors(
        state_posteriors_nk=alpha_nk * beta_nk,
        jump_posteriors_n=np.sum(draws_nk, axis=1),
        expected_draws_k=np.sum(draws_nk, axis=0),
        log_likelihood=float(np.sum(np.log(normalizers_n)) + np.sum(max_log_emissions_n)),
    )


def viterbi(log_emissions_nk: np.ndarray, jump_probs_n: np.ndarray, prior_k: np.ndarray, lengths_c) -> np.ndarray:
    """
    The most likely path of states, ie the state of each site.  The jump probabilities of the first site of each chain
    are ignored, because its state is drawn from the prior.
    """
    steps = _ChainSteps(lengths_c)
    num_states = len(prior_k)
    assert log_emissions_nk.shape == (steps.num_sites, num_states) and jump_probs_n.shape == (steps.num_sites,)
    with np.errstate(divide="ignore"):
        log_prior_k = np.log(prior_k)
        log_jump_n = np.log(jump_probs_n)

    # log probability of the most likely path to each state at the current site of each chain, in sorted chain order
    best_ck = np.empty((len(steps.sorted_starts_c), num_states))
    # whether the most likely path to state k at site n stays in k rather than jumping, and the state at site n - 1 of
    # the most likely path that jumps
    stayed_nk = np.zeros((steps.num_sites, num_states), dtype=bool)
    jumped_from_n = np.zeros(steps.num_sites, dtype=np.int64)
    for t in range(steps.num_steps):
        n = steps.sites(t)
        if t == 0:
            best_ck[: len(n)] = log_prior_k + log_emissions_nk[n]
            continue
        previous_ck = best_ck[: len(n)]
        # a path that stays in k may also jump back to it
        with np.errstate(divide="ignore"):
            stay_scores_ck = previous_ck + np.log((1 - jump_probs_n[n, None]) + jump_probs_n[n, None] * prior_k)
        jumped_from_c = np.argmax(previous_ck, axis=1)
        jump_scores_ck = (previous_ck[np.arange(len(n)), jumped_from_c] + log_jump_n[n])[:, None] + log_prior_k
        stayed_nk[n] = stay_scores_ck >= jump_scores_ck
        jumped_from_n[n] = jumped_from_c
        best_ck[: len(n)] = np.where(stayed_nk[n], stay_scores_ck, jump_scores_ck) + log_emissions_nk[n]

    # trace back from the most likely final state of each chain
    states_n = np.empty(steps.num_sites, dtype=np.int64)
    states_n[steps.sorted_starts_c + steps.sorted_lengths_c - 1] = np.argmax(best_ck, axis=1)
    for t in range(steps.num_steps - 1, 0, -1):
        n = steps.sites(t)
        states_c = states_n[n]
        states_n[n - 1] = np.where(stayed_nk[n, states_c], states_c, jumped_from_n[n])
    return states_n


def fit_jump_rate(jump_posteriors_n: np.ndarray, distances_n: np.ndarray, max_jump_rate: float) -> float:
    """
    The M-step for the jump rate lambda: maximize the expected log likelihood of the jumps,
    sum_n J_n log(1 - exp(-lambda d_n)) - (1 - J_n) lambda d_n, over the sites that aren't the first of their chain,
    for lambda between MIN_JUMP_RATE and max_jump_rate.  The objective is concave, so bisection on the sign of its
    derivative finds the maximum, and if that's outside the bounds the nearer bound is the constrained maximum.
    """
    within_n = np.isfinite(distances_n)
    jumps_n, distances_n = jump_posteriors_n[within_n], distances_n[within_n]
    stay_term = np.sum((1 - jumps_n) * distances_n)

    def derivative(rate: float) -> float:
        return np.sum(jumps_n * distances_n / np.expm1(rate * distances_n)) - stay_term

    if derivative(max_jump_rate) >= 0:
        return max_jump_rate
    if derivative(MIN_JUMP_RATE) <= 0:
        return MIN_JUMP_RATE
    low, high = math.log(MIN_JUMP_RATE), math.log(max_jump_rate)
    for _ in range(60):
        middle = (low + high) / 2
        if derivative(math.exp(middle)) > 0:
            low = middle
        else:
            high = middle
    return math.exp((low + high) / 2)


def fit_prior(expected_draws_k: np.ndarray, concentration: float) -> np.ndarray:
    """
    The M-step for the prior pi: its mode under a symmetric Dirichlet(concentration) prior, proportional to
    max(expected draws + concentration - 1, 0).  A concentration below 1 favors sparse priors by pruning the states with
    fewer than 1 - concentration expected draws.  Pruning is permanent, because a state with prior 0 gets no draws.

    The states are in order of MAF, and draws come only from breakpoints and contig starts, so a segment has about one
    draw.  A segment whose MAF is between two states splits its draw between them, which could prune both.  Instead,
    states are pruned one at a time, fewest draws first, and a pruned state's draws go to the nearest remaining state,
    where the next E-step would mostly move its sites.  If every state would be pruned, the one with the most draws
    after this merging is kept.
    """
    draws_k = np.array(expected_draws_k, dtype=np.float64)
    if concentration < 1:
        remaining_k = draws_k > 0
        while np.count_nonzero(remaining_k) > 1:
            prunable = np.flatnonzero(remaining_k & (draws_k + concentration - 1 <= 0))
            if len(prunable) == 0:
                break
            pruned = prunable[np.argmin(draws_k[prunable])]
            remaining_k[pruned] = False
            # the nearest remaining state, or of two equally near, the one with more draws
            others = np.flatnonzero(remaining_k)
            distances = np.abs(others - pruned)
            nearest = others[distances == np.min(distances)]
            draws_k[nearest[np.argmax(draws_k[nearest])]] += draws_k[pruned]
            draws_k[pruned] = 0
    unnormalized_k = np.maximum(draws_k + concentration - 1, 0)
    if not np.any(unnormalized_k > 0):
        unnormalized_k = (np.arange(len(draws_k)) == np.argmax(draws_k)).astype(np.float64)
    return unnormalized_k / np.sum(unnormalized_k)


def overrelax_prior(prior_k: np.ndarray, fitted_prior_k: np.ndarray, step_size: float) -> np.ndarray:
    """
    Over-relaxation of the M-step for the prior: extrapolate in log space from the current prior past the one that
    fit_prior gave, by a factor of step_size.  Only fit_prior prunes states: the ones it pruned stay pruned, and the
    extrapolation keeps the others.
    """
    if step_size == 1:
        return fitted_prior_k
    with np.errstate(divide="ignore", invalid="ignore"):
        extrapolated_k = np.log(prior_k) + step_size * (np.log(fitted_prior_k) - np.log(prior_k))
    log_prior_k = np.where((prior_k > 0) & (fitted_prior_k > 0), extrapolated_k, -np.inf)
    result_k = np.exp(log_prior_k - np.max(log_prior_k))
    return result_k / np.sum(result_k)
