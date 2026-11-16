import numpy as np
import pytest
import torch
from scipy.stats import betabinom

from permutect.read_orientation.f1r2_counts_reader import BASES
from permutect.read_orientation.f1r2_counts_reader import CONTEXTS
from permutect.read_orientation.f1r2_counts_reader import F1R2Counts
from permutect.read_orientation.f1r2_counts_reader import Sites
from permutect.read_orientation.read_orientation_model import ALLELE_FRACTION_PSEUDOCOUNTS_K2
from permutect.read_orientation.read_orientation_model import ARTIFACT_STATES
from permutect.read_orientation.read_orientation_model import CANONICAL_CONTEXTS
from permutect.read_orientation.read_orientation_model import F1R2_FRACTION_PSEUDOCOUNTS_K2
from permutect.read_orientation.read_orientation_model import HOM_REF
from permutect.read_orientation.read_orientation_model import NUM_STATES
from permutect.read_orientation.read_orientation_model import REVERSE_COMPLEMENT_STATES_K
from permutect.read_orientation.read_orientation_model import STATES
from permutect.read_orientation.read_orientation_model import canonical_sites
from permutect.read_orientation.read_orientation_model import flat_prior
from permutect.read_orientation.read_orientation_model import learn_prior
from permutect.read_orientation.read_orientation_model import learn_priors
from permutect.read_orientation.read_orientation_model import log_likelihoods
from permutect.read_orientation.read_orientation_model import posteriors
from permutect.read_orientation.read_orientation_model import reverse_complement

A, C, G, T = range(4)


def empty_counts() -> F1R2Counts:
    return F1R2Counts("SAMPLE", max_depth=1000, sites={context: Sites.empty() for context in CONTEXTS})


def add_sites(counts: F1R2Counts, context: str, alt_bases_n, alt_counts_n, alt_f1r2_counts_n, depths_n):
    keys_n4 = np.column_stack([alt_bases_n, alt_counts_n, alt_f1r2_counts_n, depths_n])
    counts.sites[context] += Sites.of(keys_n4, np.ones(len(keys_n4)))


def test_canonical_contexts():
    assert len(CANONICAL_CONTEXTS) == 32
    assert all(context < reverse_complement(context) for context in CANONICAL_CONTEXTS)
    assert set(CANONICAL_CONTEXTS) | {reverse_complement(context) for context in CANONICAL_CONTEXTS} == set(CONTEXTS)
    assert reverse_complement("ACG") == "CGT"


def test_reverse_complement_states():
    assert torch.equal(REVERSE_COMPLEMENT_STATES_K[REVERSE_COMPLEMENT_STATES_K], torch.arange(NUM_STATES))
    assert STATES[REVERSE_COMPLEMENT_STATES_K[STATES.index("F1R2_A")]] == "F2R1_T"
    assert STATES[REVERSE_COMPLEMENT_STATES_K[STATES.index("F2R1_G")]] == "F1R2_C"
    assert all(REVERSE_COMPLEMENT_STATES_K[k] == k for k in range(HOM_REF, NUM_STATES))


def test_log_likelihoods_are_beta_binomial():
    # a site without alt reads, then sites with alt T, T, A and G in a context with ref C
    keys = [[C, 0, 0, 40], [T, 5, 5, 50], [T, 1, 0, 7], [A, 12, 6, 100], [G, 30, 2, 60]]
    sites = Sites.of(keys, [1] * len(keys))
    result_nk = log_likelihoods(sites, ref_base=C)

    af_alpha_k, af_beta_k = ALLELE_FRACTION_PSEUDOCOUNTS_K2.numpy().T
    f1r2_alpha_k, f1r2_beta_k = F1R2_FRACTION_PSEUDOCOUNTS_K2.numpy().T
    for (alt_base, alt_count, alt_f1r2_count, depth), result_k in zip(sites.keys_n4.tolist(), result_nk):
        expected_k = betabinom.logpmf(alt_count, depth, af_alpha_k, af_beta_k)
        expected_k += betabinom.logpmf(alt_f1r2_count, alt_count, f1r2_alpha_k, f1r2_beta_k)
        # artifact states are possible only for their alt base, which can't be the ref base
        possible_k = torch.tensor(
            [state not in ARTIFACT_STATES or (state[-1] == BASES[alt_base] and alt_base != C) for state in STATES]
        )
        assert torch.all(torch.isinf(result_k[~possible_k]))
        assert np.allclose(result_k[possible_k].numpy(), expected_k[possible_k.numpy()], rtol=1e-9)


def test_posteriors():
    log_likelihoods_nk = torch.log(torch.rand(5, NUM_STATES, dtype=torch.float64))
    result_nk = posteriors(log_likelihoods_nk, flat_prior(A))
    assert torch.allclose(result_nk.sum(dim=-1), torch.ones(5, dtype=torch.float64))
    # the impossible ref-to-ref artifact states, F1R2_A and F2R1_A for ref base A
    assert torch.all(result_nk[:, [STATES.index("F1R2_A"), STATES.index("F2R1_A")]] == 0)


def test_flat_prior():
    prior_k = flat_prior(G)
    assert prior_k[STATES.index("F1R2_G")] == prior_k[STATES.index("F2R1_G")] == 0
    assert torch.allclose(prior_k[prior_k > 0], torch.full((10,), 0.1, dtype=torch.float64))


def test_canonical_sites():
    # ACA's reverse complement is TGT, whose alt C reads in F1R2 orientation are alt G reads of ACA in F2R1 orientation.
    # Sites without alt reads get ACA's ref base C as their alt base.
    counts = empty_counts()
    add_sites(counts, "TGT", [G, C, C], [0, 1, 5], [0, 1, 4], [30, 30, 40])
    add_sites(counts, "ACA", [C, G], [0, 5], [0, 1], [30, 40])
    sites = canonical_sites(counts, "ACA")
    assert sites.keys_n4.tolist() == [[C, 0, 0, 30], [G, 1, 0, 30], [G, 5, 1, 40]]
    assert sites.counts_n.tolist() == [2, 1, 2]


def simulate(counts: F1R2Counts, context: str, prior_k: np.ndarray, num_sites: int, depth: int, rng):
    """
    Add sites of a context, drawn from the model with the given prior, at the given depth.  The data fit the model
    only approximately.  Its likelihoods don't depend on the alt base of a non-artifact state, as if it were certain, so
    every alt read here has the same base, T, and the prior must give no other artifact states.  And its artifacts have
    at least one alt read, since an artifact's alt base can't be the ref base, which is the alt base of a site without
    alt reads.  An artifact's allele fraction distribution, Beta(1, 9), gives none at depth d with probability
    9 / (9 + d), which biases its prior down that much: 8% at depth 100 and 1% at depth 1000.
    """
    assert context[1] != "T" and not any(p > 0 and state[-1] != "T" for state, p in zip(ARTIFACT_STATES, prior_k))
    states_n = rng.choice(NUM_STATES, size=num_sites, p=prior_k)
    af_alpha_n, af_beta_n = ALLELE_FRACTION_PSEUDOCOUNTS_K2.numpy()[states_n].T
    alt_counts_n = rng.binomial(depth, rng.beta(af_alpha_n, af_beta_n))
    f1r2_alpha_n, f1r2_beta_n = F1R2_FRACTION_PSEUDOCOUNTS_K2.numpy()[states_n].T
    alt_f1r2_counts_n = rng.binomial(alt_counts_n, rng.beta(f1r2_alpha_n, f1r2_beta_n))
    # sites without alt reads have the ref base as their alt base
    alt_bases_n = np.where(alt_counts_n > 0, T, BASES.index(context[1]))
    add_sites(counts, context, alt_bases_n, alt_counts_n, alt_f1r2_counts_n, np.full(num_sites, depth))


def test_em_recovers_the_prior_of_simulated_data():
    # At depth 1000, where the bias is small, the largest relative error in 10 simulations of a million sites was 7%
    rng = np.random.default_rng(42)
    true_prior_k = np.zeros(NUM_STATES)
    states = ("F1R2_T", "F2R1_T", "GERMLINE_HET", "SOMATIC_HET", "HOM_VAR")
    true_prior_k[[STATES.index(s) for s in states]] = [0.01, 0.003, 0.01, 0.002, 0.005]
    true_prior_k[HOM_REF] = 1 - true_prior_k.sum()
    counts = empty_counts()
    simulate(counts, "ACG", true_prior_k, num_sites=1_000_000, depth=1000, rng=rng)

    em = learn_prior(canonical_sites(counts, "ACG"), ref_base=C, convergence_threshold=1e-8, max_iterations=500)
    assert em.converged
    for state in states:
        k = STATES.index(state)
        assert em.prior_k[k].item() == pytest.approx(true_prior_k[k], rel=0.15), state
    for state in ("F1R2_A", "F1R2_G", "F2R1_A", "F2R1_G"):
        assert em.prior_k[STATES.index(state)].item() < 1e-5, state


def test_contexts_without_enough_data_keep_the_flat_prior():
    counts = empty_counts()
    rng = np.random.default_rng(0)
    prior_k = np.zeros(NUM_STATES)
    prior_k[[STATES.index(s) for s in ("F1R2_T", "GERMLINE_HET", "HOM_VAR")]] = 0.01
    prior_k[HOM_REF] = 1 - prior_k.sum()
    simulate(counts, "ACG", prior_k, num_sites=20000, depth=60, rng=rng)
    # the canonical GGA has sites without alt reads but none with two or more
    counts.sites["GGA"] = Sites.of([[G, 0, 0, 51]], [1000])

    priors = learn_priors(counts)
    assert set(priors) == set(CONTEXTS)
    for context in ("TCC", "GGA", "AAA", "TTT"):
        assert torch.equal(priors[context].prior_k, flat_prior(BASES.index(context[1])))
        assert priors[context].num_sites == priors[context].num_alt_sites == 0
    # the prior of a reverse complement context has the equivalent states
    acg, cgt = priors["ACG"], priors["CGT"]
    assert acg.num_sites == cgt.num_sites == 20000
    assert acg.prior_k[STATES.index("F1R2_T")] == cgt.prior_k[STATES.index("F2R1_A")]
    assert torch.equal(acg.prior_k[HOM_REF:], cgt.prior_k[HOM_REF:])
