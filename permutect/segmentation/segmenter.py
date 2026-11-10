"""
Segmentation of one sample's sites by minor allele fraction (MAF), with the hidden Markov model of hmm.py and the
emission model of emissions.py fit by expectation maximization (EM).

The hidden states are a fixed grid of MAFs from 0 to 1/2.  Each EM iteration runs forward-backward with the current
parameters (the E-step), then updates the prior distribution of MAF states, the jump rate and the emission parameters
(the M-step).  The sparse Dirichlet prior on the MAF prior takes effect only after the first few iterations, because it
prunes states permanently.  After EM, Viterbi gives the most likely state of each site, and each run of consecutive
sites in the same state is a segment.

Plain EM converges slowly here, because it prunes states one at a time and the emission parameters creep toward their
optima.  Adaptive over-relaxation (Salakhutdinov and Roweis, 2003) speeds it up: once pruning starts, the M-step's
changes to the MAF prior and the emission parameters are multiplied by a step size that grows after every iteration
that raises the log likelihood and returns to 1 after one that lowers it.  On a real exome this cut EM from 45
iterations to 21, with the same segments.
"""

import math
from dataclasses import dataclass

import numpy as np
import torch

from permutect.segmentation.emissions import EmissionModel
from permutect.segmentation.emissions import Pileups
from permutect.segmentation.hmm import fit_jump_rate
from permutect.segmentation.hmm import fit_prior
from permutect.segmentation.hmm import forward_backward
from permutect.segmentation.hmm import jump_probabilities
from permutect.segmentation.hmm import overrelax_prior
from permutect.segmentation.hmm import site_distances
from permutect.segmentation.hmm import viterbi

# EM iterations before the sparse prior on the MAF prior takes effect
NUM_ITERATIONS_BEFORE_PRUNING = 3
INITIAL_JUMP_RATE = 1e-8
# the emission M-step fits a random subset of at most this many sites, which is plenty for its five parameters
MAX_FITTING_SITES = 100_000
# L-BFGS iterations of each emission M-step, which starts from the parameters of the previous one
EMISSION_FIT_ITERATIONS = 5
# EM stops when the log likelihood changes by less than this per site
CONVERGENCE_THRESHOLD = 1e-6
# the over-relaxation step size grows by this factor after each iteration that raises the log likelihood, up to the max
STEP_SIZE_GROWTH = 1.5
MAX_STEP_SIZE = 10.0
# sites per chunk when computing emissions, to bound the memory of intermediate tensors
CHUNK_SIZE = 100_000
# the posterior of each segment's MAF is computed on this many MAFs evenly spaced from 0 to 1/2, ie every 0.005, which
# is finer than the MAFs of segments can be told apart and keeps this affordable for a genome's millions of sites
NUM_POSTERIOR_MAFS = 101
# each segment's MAF has an interval of the narrowest MAFs with this much of its posterior, ie a highest density interval
MAF_INTERVAL_MASS = 0.8


@dataclass
class Segmentation:
    # the MAF of each state, a grid from 0 to 1/2
    mafs_k: np.ndarray
    # the most likely state of each site, jointly ie by Viterbi
    states_n: np.ndarray
    # the posterior mean MAF of each site
    posterior_mafs_n: np.ndarray
    # the first site of each run of consecutive sites of a contig in the same state, ie of each segment
    run_starts_r: np.ndarray
    # for each run, the lower bound of its MAF's interval, its MAF, and the upper bound
    run_mafs_r3: np.ndarray
    prior_k: np.ndarray
    jump_rate: float
    emission_model: EmissionModel
    log_likelihood: float


@dataclass
class Segment:
    contig: str
    # 1-based, like the end, which is inclusive
    start: int
    end: int
    maf: float
    # in a gap between segments, the MAF of the segment after the gap; otherwise the same as the MAF
    secondary_maf: float
    # the bounds of the MAF's interval, which has MAF_INTERVAL_MASS of its posterior, or NaN for a gap between segments
    maf_low: float = math.nan
    maf_high: float = math.nan


def log_emissions(emission_model: EmissionModel, pileups: Pileups, mafs_k: np.ndarray, dtype=np.float32) -> np.ndarray:
    """
    The log emission probability of each site in each MAF state, computed in chunks of sites, which bounds the memory
    of the emission model's intermediate tensors, into an array of the given dtype.  float32, the default, halves the
    memory of one of the HMM's two largest arrays without changing its posteriors or Viterbi path appreciably.
    """
    num_sites = len(pileups.alt_counts_n)
    log_emissions_nk = np.empty((num_sites, len(mafs_k)), dtype=dtype)
    with torch.no_grad():
        for start in range(0, num_sites, CHUNK_SIZE):
            chunk = slice(start, start + CHUNK_SIZE)
            log_emissions_nk[chunk] = emission_model.log_likelihoods(pileups.subset(chunk), torch.from_numpy(mafs_k))
    return log_emissions_nk


def segment(
    pileups: Pileups,
    positions_n: np.ndarray,
    lengths_c: np.ndarray,
    num_maf_states: int,
    contamination: float,
    max_jump_rate: float,
    prior_concentration: float,
    max_iterations: int,
) -> Segmentation:
    """
    Segment sites given their pileups and positions, where the sites of each contig are consecutive, with lengths_c[c]
    sites in contig c, and in order of position
    """
    num_sites = len(positions_n)
    mafs_k = np.linspace(0, 0.5, num_maf_states)
    distances_n = site_distances(positions_n, lengths_c)
    prior_k = np.full(num_maf_states, 1 / num_maf_states)
    jump_rate = min(INITIAL_JUMP_RATE, max_jump_rate)
    emission_model = EmissionModel(contamination=contamination)
    rng = np.random.default_rng(0)
    fitting_sites = np.sort(rng.choice(num_sites, min(num_sites, MAX_FITTING_SITES), replace=False))
    fitting_pileups = pileups.subset(torch.from_numpy(fitting_sites))

    previous_log_likelihood, previous_num_states, step_size = -math.inf, num_maf_states, 1.0
    for iteration in range(max_iterations):
        # E-step, restricted to the states that haven't been pruned
        active_k = np.flatnonzero(prior_k)
        log_emissions_nk = log_emissions(emission_model, pileups, mafs_k[active_k])
        jump_probs_n = jump_probabilities(distances_n, jump_rate)
        # the emissions overwrite the log emissions, to save memory
        posteriors = forward_backward(
            log_emissions_nk, jump_probs_n, prior_k[active_k], lengths_c, overwrite_log_emissions=True
        )
        print(
            f"EM iteration {iteration}: log likelihood {posteriors.log_likelihood:.2f}, {len(active_k)} MAF states, "
            f"jump rate {jump_rate:.3g} per base, step size {step_size:.3g}"
        )

        # the sparse prior affects the log likelihood from iteration NUM_ITERATIONS_BEFORE_PRUNING + 1 on.  Convergence
        # also requires an iteration without pruning, since the log likelihood can level off while pruning continues.
        change = abs(posteriors.log_likelihood - previous_log_likelihood)
        converged = (
            iteration >= NUM_ITERATIONS_BEFORE_PRUNING + 2
            and change < CONVERGENCE_THRESHOLD * num_sites
            and len(active_k) == previous_num_states
        )
        if converged or iteration == max_iterations - 1:
            break
        if iteration > NUM_ITERATIONS_BEFORE_PRUNING:
            improved = posteriors.log_likelihood >= previous_log_likelihood
            step_size = min(STEP_SIZE_GROWTH * step_size, MAX_STEP_SIZE) if improved else 1.0
        previous_log_likelihood, previous_num_states = posteriors.log_likelihood, len(active_k)

        # M-step
        expected_draws_k = np.zeros(num_maf_states)
        expected_draws_k[active_k] = posteriors.expected_draws_k
        concentration = prior_concentration if iteration >= NUM_ITERATIONS_BEFORE_PRUNING else 1
        fitted_prior_k = fit_prior(expected_draws_k, concentration)
        prior_k = overrelax_prior(prior_k, fitted_prior_k, step_size)
        jump_rate = fit_jump_rate(posteriors.jump_posteriors_n, distances_n, max_jump_rate)
        fitting_posteriors_nk = torch.from_numpy(posteriors.state_posteriors_nk[fitting_sites])
        fitting_mafs_k = torch.from_numpy(mafs_k[active_k])
        emission_model.fit(fitting_pileups, fitting_posteriors_nk, fitting_mafs_k, EMISSION_FIT_ITERATIONS, step_size)
        # release this iteration's site-by-state arrays before the next E-step allocates its own
        log_emissions_nk = posteriors = None

    model = emission_model
    print(
        f"error rate {model.error_rate.item():.3g}, hom concentration {model.hom_concentration.item():.3g}, "
        f"het concentration {model.het_concentration.item():.3g}, reference bias {model.reference_bias.item():.3g}, "
        f"outlier weight {model.outlier_weight.item():.3g}"
    )
    # EM stops right after an E-step, so these are the posteriors of the final parameters
    posterior_mafs_n = posteriors.state_posteriors_nk @ mafs_k[active_k]
    log_likelihood = posteriors.log_likelihood
    # forward_backward overwrote the log emissions with the emissions, so Viterbi recomputes them, after the old arrays
    # are released.  The recomputed ones are released when Viterbi returns.
    log_emissions_nk = posteriors = None
    active_states_n = viterbi(
        log_emissions(emission_model, pileups, mafs_k[active_k]), jump_probs_n, prior_k[active_k], lengths_c
    )
    states_n = active_k[active_states_n]
    run_starts_r = find_runs(states_n, lengths_c)
    return Segmentation(
        mafs_k=mafs_k,
        states_n=states_n,
        posterior_mafs_n=posterior_mafs_n,
        run_starts_r=run_starts_r,
        run_mafs_r3=run_mafs(emission_model, pileups, run_starts_r),
        prior_k=prior_k,
        jump_rate=jump_rate,
        emission_model=emission_model,
        log_likelihood=log_likelihood,
    )


def find_runs(states_n: np.ndarray, lengths_c: np.ndarray) -> np.ndarray:
    """the first site of each run of consecutive sites of a contig in the same state"""
    new_run_n = np.ones(len(states_n), dtype=bool)
    new_run_n[1:] = states_n[1:] != states_n[:-1]
    new_run_n[np.cumsum(lengths_c) - lengths_c] = True
    return np.flatnonzero(new_run_n)


def run_mafs(emission_model: EmissionModel, pileups: Pileups, run_starts_r: np.ndarray) -> np.ndarray:
    """
    For each run, the lower bound, mode and upper bound of the posterior of its MAF, given its sites and the fitted
    emission model, with a uniform prior on the MAF.  The bounds are those of the narrowest interval containing the mode
    with MAF_INTERVAL_MASS of the posterior.  Posteriors are computed on NUM_POSTERIOR_MAFS evenly spaced MAFs, which
    sets their precision.  Unlike the HMM's posteriors, they don't depend on which states EM pruned.

    The mode, not the median, is the MAF, because the posterior of a balanced segment piles up against 1/2, the largest
    MAF, which pulls its median and percentiles below 1/2.  Even so, a balanced segment's MAF is weakly identified: a
    little more spread among its hets' allele fractions than the model expects looks like a MAF slightly below 1/2, so
    its MAF can come out as low as about 0.47.
    """
    mafs_k = np.linspace(0, 0.5, NUM_POSTERIOR_MAFS)
    num_sites = len(pileups.alt_counts_n)
    runs_n = np.repeat(np.arange(len(run_starts_r)), np.diff(np.append(run_starts_r, num_sites)))
    log_likelihoods_rk = np.zeros((len(run_starts_r), NUM_POSTERIOR_MAFS))
    for start in range(0, num_sites, CHUNK_SIZE):
        sites = slice(start, start + CHUNK_SIZE)
        # float64, because a run's log likelihoods sum over many sites
        log_emissions_nk = log_emissions(emission_model, pileups.subset(sites), mafs_k, dtype=np.float64)
        # the chunk's sites in each run, which are consecutive and in different runs from the other blocks'
        block_starts = np.flatnonzero(np.diff(runs_n[sites], prepend=-1))
        log_likelihoods_rk[runs_n[sites][block_starts]] += np.add.reduceat(log_emissions_nk, block_starts, axis=0)

    posteriors_rk = np.exp(log_likelihoods_rk - np.max(log_likelihoods_rk, axis=1, keepdims=True))
    posteriors_rk /= np.sum(posteriors_rk, axis=1, keepdims=True)

    # the highest density interval: the most probable MAFs, in decreasing order, until they have the interval's mass
    order_rk = np.argsort(-posteriors_rk, axis=1)
    sorted_cumulative_rk = np.cumsum(np.take_along_axis(posteriors_rk, order_rk, axis=1), axis=1)
    num_in_interval_r = 1 + np.sum(sorted_cumulative_rk < MAF_INTERVAL_MASS, axis=1)
    ranks_rk = np.argsort(order_rk, axis=1)
    in_interval_rk = ranks_rk < num_in_interval_r[:, None]
    lows_r = mafs_k[np.argmax(in_interval_rk, axis=1)]
    highs_r = mafs_k[NUM_POSTERIOR_MAFS - 1 - np.argmax(in_interval_rk[:, ::-1], axis=1)]
    return np.stack((lows_r, mafs_k[order_rk[:, 0]], highs_r), axis=1)


def make_segments(
    contigs_c: list[str],
    lengths_c: np.ndarray,
    positions_n: np.ndarray,
    run_starts_r: np.ndarray,
    run_mafs_r3: np.ndarray,
) -> list[Segment]:
    """
    Each run of consecutive sites of a contig is a segment from its first site to its last.  Runs start at
    run_starts_r and don't cross contigs; run_mafs_r3 has the lower bound of the MAF of each run, the MAF and the
    upper bound.  Between consecutive segments of a contig, unless their sites are adjacent, there's a gap segment whose
    MAF is that of the segment before it and whose secondary MAF is that of the segment after it.  The ends of a
    contig, before its first site and after its last, aren't in any segment.
    """
    run_ends_r = np.append(run_starts_r[1:], len(positions_n))
    contigs_r = np.searchsorted(np.cumsum(lengths_c), run_starts_r, side="right")
    segments = []
    for run, (start, end) in enumerate(zip(run_starts_r, run_ends_r)):
        contig = contigs_c[contigs_r[run]]
        low, maf, high = (float(value) for value in run_mafs_r3[run])
        if run > 0 and contigs_r[run - 1] == contigs_r[run] and positions_n[start] > positions_n[start - 1] + 1:
            previous_maf = float(run_mafs_r3[run - 1, 1])
            gap_start, gap_end = int(positions_n[start - 1]) + 1, int(positions_n[start]) - 1
            segments.append(Segment(contig, gap_start, gap_end, previous_maf, maf))
        segments.append(Segment(contig, int(positions_n[start]), int(positions_n[end - 1]), maf, maf, low, high))
    return segments
