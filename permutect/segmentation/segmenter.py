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


@dataclass
class Segmentation:
    # the MAF of each state, a grid from 0 to 1/2
    mafs_k: np.ndarray
    # the most likely state of each site, jointly ie by Viterbi
    states_n: np.ndarray
    # the posterior mean MAF of each site
    posterior_mafs_n: np.ndarray
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


def log_emissions(emission_model: EmissionModel, pileups: Pileups, mafs_k: np.ndarray) -> np.ndarray:
    num_sites = len(pileups.alt_counts_n)
    with torch.no_grad():
        chunks = [
            emission_model.log_likelihoods(pileups.subset(slice(start, start + CHUNK_SIZE)), torch.from_numpy(mafs_k))
            for start in range(0, num_sites, CHUNK_SIZE)
        ]
    return torch.cat(chunks).numpy()


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
        posteriors = forward_backward(log_emissions_nk, jump_probs_n, prior_k[active_k], lengths_c)
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

    model = emission_model
    print(
        f"error rate {model.error_rate.item():.3g}, hom concentration {model.hom_concentration.item():.3g}, "
        f"het concentration {model.het_concentration.item():.3g}, reference bias {model.reference_bias.item():.3g}, "
        f"outlier weight {model.outlier_weight.item():.3g}"
    )
    return Segmentation(
        mafs_k=mafs_k,
        states_n=active_k[viterbi(log_emissions_nk, jump_probs_n, prior_k[active_k], lengths_c)],
        posterior_mafs_n=posteriors.state_posteriors_nk @ mafs_k[active_k],
        prior_k=prior_k,
        jump_rate=jump_rate,
        emission_model=emission_model,
        log_likelihood=posteriors.log_likelihood,
    )


def make_segments(
    contigs_c: list[str], lengths_c: np.ndarray, positions_n: np.ndarray, states_n: np.ndarray, mafs_n: np.ndarray
) -> list[Segment]:
    """
    Each run of consecutive sites of a contig in the same state is a segment from its first site to its last, whose MAF
    is the mean MAF of its sites.  Between consecutive segments of a contig, unless their sites are adjacent, there's a
    gap segment whose MAF is that of the segment before it and whose secondary MAF is that of the segment after it.
    The ends of a contig, before its first site and after its last, aren't in any segment.
    """
    segments = []
    for contig, start, length in zip(contigs_c, np.cumsum(lengths_c) - lengths_c, lengths_c):
        positions = positions_n[start : start + length]
        run_starts = np.flatnonzero(np.diff(states_n[start : start + length], prepend=-1))
        run_ends = np.append(run_starts[1:], length)
        run_mafs = np.add.reduceat(mafs_n[start : start + length], run_starts) / (run_ends - run_starts)
        for run, (run_start, run_end) in enumerate(zip(run_starts, run_ends)):
            if run > 0 and positions[run_start] > positions[run_start - 1] + 1:
                gap_start, gap_end = int(positions[run_start - 1]) + 1, int(positions[run_start]) - 1
                segments.append(Segment(contig, gap_start, gap_end, float(run_mafs[run - 1]), float(run_mafs[run])))
            maf = float(run_mafs[run])
            segments.append(Segment(contig, int(positions[run_start]), int(positions[run_end - 1]), maf, maf))
    return segments
