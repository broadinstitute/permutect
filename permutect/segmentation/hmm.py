"""
The hidden Markov model of allele fraction segmentation, apart from its emissions: forward-backward, Viterbi, and the
M-step updates of the transition parameters.

Sites come in chains, one per contig.  The hidden state of each site is one of K states, in segmentation a grid of minor
allele fractions.  Between consecutive sites n - 1 and n of a chain the state jumps with probability jump_n, which grows
with the distance between the sites: breakpoints are a Poisson process with rate lambda per base, so
jump_n = 1 - exp(-lambda * distance_n).  A jump redraws the state from a prior pi shared by all states, so it may land
back on the same state; without a jump the state stays the same.  The first site of each chain is drawn from pi.

The transition matrix (1 - jump_n) I + jump_n 1 pi^T is diagonal plus rank one, so the recursions never build it and
cost O(K) per site rather than O(K^2).  They step through each chain one site at a time, which in Python would cost far
more per site than the arithmetic, so numba compiles them, and since chains are independent, it runs them in parallel.

Arrays are indexed by site n, over the concatenated sites of all chains, by chain c, and by state k.
"""

import math
from dataclasses import dataclass

import numpy as np
from numba import njit
from numba import prange

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


def _chain_starts(lengths_c) -> tuple[np.ndarray, np.ndarray]:
    """the first site of each chain, and the lengths of the chains as int64"""
    lengths_c = np.asarray(lengths_c, dtype=np.int64)
    assert lengths_c.ndim == 1 and len(lengths_c) > 0 and np.all(lengths_c > 0)
    return np.cumsum(lengths_c) - lengths_c, lengths_c


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
    log_emissions_nk: np.ndarray,
    jump_probs_n: np.ndarray,
    prior_k: np.ndarray,
    lengths_c,
    overwrite_log_emissions: bool = False,
) -> Posteriors:
    """
    The E-step: posteriors of the states and jumps given the data.  The jump probabilities of the first site of each
    chain are ignored, because its state is drawn from the prior.

    Memory: a whole genome has millions of sites, so this keeps only two site-by-state arrays.  One holds the emissions,
    with the dtype of the log emissions, and overwrites them if overwrite_log_emissions.  The other holds the forward
    probabilities, which the backward pass overwrites with the state posteriors as it goes.  The backward probabilities
    and the posteriors of jumps are needed for one site of each chain at a time, so they are never stored for every site.

    Precision: only the emissions may be float32, which is safe because they are relative to each site's largest, and
    the emission model's outliers keep them from being tiny.  Everything else is float64, including the forward
    probabilities.  Before EM prunes a state that the data don't support, its prior can be far smaller than the smallest
    float32, so where such a state fits the sites ahead much better than the current mixture, its forward probability is
    tiny and its backward probability huge, and their product, its posterior, needn't be small.  In float32 the forward
    probability would underflow to 0 and the backward one overflow, giving NaNs.
    """
    starts_c, lengths_c = _chain_starts(lengths_c)
    num_sites, num_states = log_emissions_nk.shape
    assert num_sites == np.sum(lengths_c) and num_states == len(prior_k) and jump_probs_n.shape == (num_sites,)
    emissions_nk = log_emissions_nk if overwrite_log_emissions else log_emissions_nk.copy()
    jump_n = jump_probs_n.astype(np.float64)
    jump_n[starts_c] = 1.0
    max_log_emissions_n, normalizers_n, jump_posteriors_n = (np.empty(num_sites) for _ in range(3))
    posteriors_nk = np.empty((num_sites, num_states))
    draws_ck = np.zeros((len(starts_c), num_states))
    _forward_backward_kernel(
        emissions_nk,
        jump_n,
        prior_k.astype(np.float64),
        starts_c,
        lengths_c,
        max_log_emissions_n,
        normalizers_n,
        posteriors_nk,
        jump_posteriors_n,
        draws_ck,
    )
    return Posteriors(
        state_posteriors_nk=posteriors_nk,
        jump_posteriors_n=jump_posteriors_n,
        expected_draws_k=np.sum(draws_ck, axis=0),
        log_likelihood=float(np.sum(np.log(normalizers_n)) + np.sum(max_log_emissions_n)),
    )


@njit(parallel=True)
def _forward_backward_kernel(
    emissions_nk,
    jump_n,
    prior_k,
    starts_c,
    lengths_c,
    max_log_emissions_n,
    normalizers_n,
    posteriors_nk,
    jump_posteriors_n,
    draws_ck,
):
    """
    forward_backward for each chain in parallel.  The log emissions in emissions_nk become the emissions relative to each
    site's largest, so that they don't underflow, and the log likelihood adds back the shifts, max_log_emissions_n.  Each
    chain's expected draws are summed separately, since chains run in parallel.
    """
    num_states = len(prior_k)
    for c in prange(len(starts_c)):
        start, end = starts_c[c], starts_c[c] + lengths_c[c]

        # alpha_nk = P(state k at site n | data up to site n) and normalizers_n = P(data at site n | data before site n),
        # both within the chain and with the emissions shifted.  posteriors_nk holds alpha until the backward pass.
        for n in range(start, end):
            largest = emissions_nk[n, 0]
            for k in range(1, num_states):
                largest = max(largest, emissions_nk[n, k])
            max_log_emissions_n[n] = largest
            normalizer = 0.0
            for k in range(num_states):
                emissions_nk[n, k] = math.exp(emissions_nk[n, k] - largest)
                stay = 0.0 if n == start else (1 - jump_n[n]) * posteriors_nk[n - 1, k]
                posteriors_nk[n, k] = emissions_nk[n, k] * (stay + jump_n[n] * prior_k[k])
                normalizer += posteriors_nk[n, k]
            normalizers_n[n] = normalizer
            for k in range(num_states):
                posteriors_nk[n, k] /= normalizer

        # beta_k = P(data after site n | state k at site n) / P(data after site n | data up to site n), within the chain,
        # at the current site, which is 1 at the chain's last site.  The state posteriors alpha * beta overwrite alpha
        # site by site, after which alpha isn't needed.
        beta_k = np.ones(num_states)
        emitted_k = np.empty(num_states)
        for n in range(end - 1, start - 1, -1):
            # P(jump at site n that lands in state k | data), summing over the state at site n - 1, whose alpha sums to 1.
            # At the first site of the chain, where jump_n = 1, this is the state posterior.
            draw_scale = jump_n[n] / normalizers_n[n]
            jump_posterior, after_jump = 0.0, 0.0
            for k in range(num_states):
                emitted_k[k] = emissions_nk[n, k] * beta_k[k]
                draws = draw_scale * prior_k[k] * emitted_k[k]
                jump_posterior += draws
                draws_ck[c, k] += draws
                posteriors_nk[n, k] *= beta_k[k]
                after_jump += prior_k[k] * emitted_k[k]
            jump_posteriors_n[n] = jump_posterior

            # beta at site n - 1
            if n > start:
                for k in range(num_states):
                    beta_k[k] = ((1 - jump_n[n]) * emitted_k[k] + jump_n[n] * after_jump) / normalizers_n[n]


def viterbi(log_emissions_nk: np.ndarray, jump_probs_n: np.ndarray, prior_k: np.ndarray, lengths_c) -> np.ndarray:
    """
    The most likely path of states, ie the state of each site.  The jump probabilities of the first site of each chain
    are ignored, because its state is drawn from the prior.
    """
    starts_c, lengths_c = _chain_starts(lengths_c)
    num_sites, num_states = log_emissions_nk.shape
    assert num_sites == np.sum(lengths_c) and num_states == len(prior_k) and jump_probs_n.shape == (num_sites,)
    with np.errstate(divide="ignore"):
        log_prior_k = np.log(prior_k.astype(np.float64))
        log_jump_n = np.log(jump_probs_n.astype(np.float64))
    states_n = np.empty(num_sites, dtype=np.int64)
    _viterbi_kernel(
        log_emissions_nk,
        jump_probs_n.astype(np.float64),
        log_jump_n,
        prior_k.astype(np.float64),
        log_prior_k,
        starts_c,
        lengths_c,
        states_n,
    )
    return states_n


@njit(parallel=True)
def _viterbi_kernel(log_emissions_nk, jump_n, log_jump_n, prior_k, log_prior_k, starts_c, lengths_c, states_n):
    """viterbi for each chain in parallel"""
    num_states = len(prior_k)
    for c in prange(len(starts_c)):
        start, length = starts_c[c], lengths_c[c]
        # log probability of the most likely path to each state at the current site
        best_k = log_prior_k + log_emissions_nk[start]
        next_best_k = np.empty(num_states)
        # whether the most likely path to state k at the chain's site i stays in k rather than jumping, and the state at
        # site i - 1 of the most likely path that jumps
        stayed_ik = np.zeros((length, num_states), dtype=np.bool_)
        jumped_from_i = np.zeros(length, dtype=np.int64)
        for i in range(1, length):
            n = start + i
            jumped_from = np.argmax(best_k)
            jumped_from_i[i] = jumped_from
            for k in range(num_states):
                # a path that stays in k may also jump back to it
                stay_score = best_k[k] + math.log((1 - jump_n[n]) + jump_n[n] * prior_k[k])
                jump_score = best_k[jumped_from] + log_jump_n[n] + log_prior_k[k]
                stayed_ik[i, k] = stay_score >= jump_score
                next_best_k[k] = (stay_score if stayed_ik[i, k] else jump_score) + log_emissions_nk[n, k]
            best_k[:] = next_best_k

        # trace back from the most likely final state
        states_n[start + length - 1] = np.argmax(best_k)
        for i in range(length - 1, 0, -1):
            state = states_n[start + i]
            states_n[start + i - 1] = state if stayed_ik[i, state] else jumped_from_i[i]


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
