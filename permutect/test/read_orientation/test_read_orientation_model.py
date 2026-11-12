import numpy as np
import pytest
import torch
from scipy.stats import betabinom

from permutect.read_orientation.picard_histogram_reader import BASES
from permutect.read_orientation.picard_histogram_reader import CONTEXTS
from permutect.read_orientation.picard_histogram_reader import ORIENTATIONS
from permutect.read_orientation.picard_histogram_reader import AltSites
from permutect.read_orientation.picard_histogram_reader import F1R2Counts
from permutect.read_orientation.read_orientation_model import ALLELE_FRACTION_PSEUDOCOUNTS_K2
from permutect.read_orientation.read_orientation_model import CANONICAL_CONTEXTS
from permutect.read_orientation.read_orientation_model import F1R2_FRACTION_PSEUDOCOUNTS_K2
from permutect.read_orientation.read_orientation_model import HOM_REF
from permutect.read_orientation.read_orientation_model import NUM_STATES
from permutect.read_orientation.read_orientation_model import REVERSE_COMPLEMENT_STATES_K
from permutect.read_orientation.read_orientation_model import STATE_ALT_BASES_K
from permutect.read_orientation.read_orientation_model import STATES
from permutect.read_orientation.read_orientation_model import context_data
from permutect.read_orientation.read_orientation_model import flat_prior
from permutect.read_orientation.read_orientation_model import learn_prior
from permutect.read_orientation.read_orientation_model import learn_priors
from permutect.read_orientation.read_orientation_model import log_likelihoods
from permutect.read_orientation.read_orientation_model import responsibilities
from permutect.read_orientation.read_orientation_model import reverse_complement

# the maximum depth of the histograms, which is 200 by default in Mutect2, high enough for simulations at depth 1000
MAX_DEPTH = 1000


def empty_counts() -> F1R2Counts:
    zeros = np.zeros(MAX_DEPTH, dtype=np.int64)
    ref = {context: zeros.copy() for context in CONTEXTS}
    alt = {
        (context, alt_base, orientation): zeros.copy()
        for context in CONTEXTS
        for alt_base in BASES
        if alt_base != context[1]
        for orientation in ORIENTATIONS
    }
    return F1R2Counts("SAMPLE", MAX_DEPTH, ref, alt, {context: AltSites.empty() for context in CONTEXTS})


def add_sites(counts: F1R2Counts, context: str, alt_bases_n, alt_counts_n, alt_f1r2_counts_n, depths_n):
    """add sites to F1R2 counts as GATK collects them: into histograms if they have fewer than two alt reads"""
    alt_bases_n, alt_counts_n, alt_f1r2_counts_n, depths_n = (
        np.asarray(x) for x in (alt_bases_n, alt_counts_n, alt_f1r2_counts_n, depths_n)
    )
    bins_n = np.minimum(depths_n, MAX_DEPTH) - 1
    np.add.at(counts.ref_histograms[context], bins_n[alt_counts_n == 0], 1)
    for alt_base in BASES:
        for orientation, alt_f1r2_count in (("F1R2", 1), ("F2R1", 0)):
            sites = (alt_counts_n == 1) & (alt_bases_n == BASES.index(alt_base)) & (alt_f1r2_counts_n == alt_f1r2_count)
            if np.any(sites):
                np.add.at(counts.alt_depth_one_histograms[(context, alt_base, orientation)], bins_n[sites], 1)
    multiple = alt_counts_n >= 2
    columns = (np.asarray(column)[multiple] for column in (alt_bases_n, alt_counts_n, alt_f1r2_counts_n, depths_n))
    new_sites = AltSites.collapse(*columns, np.ones(int(multiple.sum()), dtype=np.int64))
    counts.alt_sites[context] = counts.alt_sites[context] + new_sites


def test_canonical_contexts():
    assert len(CANONICAL_CONTEXTS) == 32
    assert all(context < reverse_complement(context) for context in CANONICAL_CONTEXTS)
    assert set(CANONICAL_CONTEXTS) | {reverse_complement(context) for context in CANONICAL_CONTEXTS} == set(CONTEXTS)
    assert reverse_complement("ACG") == "CGT"


def test_reverse_complement_states():
    assert torch.equal(REVERSE_COMPLEMENT_STATES_K[REVERSE_COMPLEMENT_STATES_K], torch.arange(NUM_STATES))
    assert STATES[REVERSE_COMPLEMENT_STATES_K[STATES.index("F1R2_A")]] == "F2R1_T"
    assert STATES[REVERSE_COMPLEMENT_STATES_K[STATES.index("F2R1_G")]] == "F1R2_C"
    assert all(REVERSE_COMPLEMENT_STATES_K[k] == k for k in range(8, NUM_STATES))


def test_log_likelihoods_are_beta_binomial():
    ref_base = BASES.index("C")
    alt_bases_n = torch.tensor([1, 3, 3, 0, 2])  # a ref site, then alt T, T, A and G
    alt_counts_n = torch.tensor([0, 5, 1, 12, 30])
    alt_f1r2_counts_n = torch.tensor([0, 5, 0, 6, 2])
    depths_n = torch.tensor([40, 50, 7, 100, 60])
    result_nk = log_likelihoods(ref_base, alt_bases_n, alt_counts_n, alt_f1r2_counts_n, depths_n)

    af_alpha_k, af_beta_k = ALLELE_FRACTION_PSEUDOCOUNTS_K2.numpy().T
    f1r2_alpha_k, f1r2_beta_k = F1R2_FRACTION_PSEUDOCOUNTS_K2.numpy().T
    for n in range(len(depths_n)):
        expected_k = betabinom.logpmf(
            alt_counts_n[n].item(), depths_n[n].item(), af_alpha_k, af_beta_k
        ) + betabinom.logpmf(alt_f1r2_counts_n[n].item(), alt_counts_n[n].item(), f1r2_alpha_k, f1r2_beta_k)
        # artifact states are possible only for their alt base, which can't be the ref base
        possible_k = (STATE_ALT_BASES_K == alt_bases_n[n]) & (STATE_ALT_BASES_K != ref_base) | (STATE_ALT_BASES_K < 0)
        assert torch.all(torch.isinf(result_nk[n, ~possible_k]))
        assert np.allclose(result_nk[n, possible_k].numpy(), expected_k[possible_k.numpy()], rtol=1e-9)
    # a ref site can't be an artifact
    assert torch.all(torch.isinf(result_nk[0, :8]))


def test_responsibilities():
    log_likelihoods_nk = torch.log(torch.rand(5, NUM_STATES, dtype=torch.float64))
    prior_k = flat_prior(BASES.index("A"))
    result_nk = responsibilities(log_likelihoods_nk, prior_k)
    assert torch.allclose(result_nk.sum(dim=-1), torch.ones(5, dtype=torch.float64))
    # the impossible ref-to-ref artifact states, F1R2_A and F2R1_A for ref base A
    assert torch.all(result_nk[:, [0, 4]] == 0)
    not_hom_ref_nk = responsibilities(log_likelihoods_nk, prior_k, given_not_hom_ref=True)
    assert torch.all(not_hom_ref_nk[:, HOM_REF] == 0)
    assert torch.allclose(not_hom_ref_nk.sum(dim=-1), torch.ones(5, dtype=torch.float64))


def test_flat_prior():
    prior_k = flat_prior(BASES.index("G"))
    assert prior_k[STATES.index("F1R2_G")] == prior_k[STATES.index("F2R1_G")] == 0
    assert torch.allclose(prior_k[prior_k > 0], torch.full((10,), 0.1, dtype=torch.float64))


def test_reverse_complement_data_go_to_the_canonical_context():
    # ACA's reverse complement is TGT, whose alt C reads in F1R2 orientation are alt G reads of ACA in F2R1 orientation.
    # Sites without alt reads get ACA's ref base C as their alt base.
    counts = empty_counts()
    c, g = BASES.index("C"), BASES.index("G")
    add_sites(counts, "TGT", [c, c, c], [0, 1, 5], [0, 1, 4], [30, 30, 40])
    add_sites(counts, "ACA", [g, g], [0, 5], [0, 1], [30, 40])
    data = context_data(counts, "ACA")
    assert (data.num_ref_sites, data.num_alt_depth_one_sites, data.num_multiple_alt_sites) == (2, 1, 2)
    sites = sorted(
        zip(
            *(
                x.tolist()
                for x in (data.alt_bases_n, data.alt_counts_n, data.alt_f1r2_counts_n, data.depths_n, data.num_sites_n)
            )
        )
    )
    assert sites == [(c, 0, 0, 30, 2.0), (g, 1, 0, 30, 1.0), (g, 5, 1, 40, 2.0)]


def simulate(counts: F1R2Counts, context: str, prior_k: np.ndarray, num_sites: int, depth: int, rng):
    """
    Add sites of a context, drawn from the model with the given prior, at the given depth.  The data fit the model
    only approximately.  Its likelihoods don't depend on the alt base of a non-artifact state, as if it were certain, so
    every alt read here has the same base, T, and the prior must give no other artifact states.  And its artifacts have
    at least one alt read, since an artifact's alt base can't be the ref base, which is the alt base of a site without
    alt reads.  An artifact's allele fraction distribution, Beta(1, 9), gives none at depth d with probability
    9 / (9 + d), which biases its prior down that much: 8% at depth 100 and 1% at depth 1000.
    """
    alt_base = BASES.index("T")
    assert context[1] != "T" and all(prior_k[k] == 0 for k in range(8) if STATE_ALT_BASES_K[k] != alt_base)
    states_n = rng.choice(NUM_STATES, size=num_sites, p=prior_k)
    af_alpha_n, af_beta_n = ALLELE_FRACTION_PSEUDOCOUNTS_K2.numpy()[states_n].T
    alt_counts_n = rng.binomial(depth, rng.beta(af_alpha_n, af_beta_n))
    f1r2_alpha_n, f1r2_beta_n = F1R2_FRACTION_PSEUDOCOUNTS_K2.numpy()[states_n].T
    alt_f1r2_counts_n = rng.binomial(alt_counts_n, rng.beta(f1r2_alpha_n, f1r2_beta_n))
    # sites without alt reads have the ref base as their alt base
    alt_bases_n = np.where(alt_counts_n > 0, alt_base, BASES.index(context[1]))
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

    em = learn_prior(context_data(counts, "ACG"), convergence_threshold=1e-8, max_iterations=500)
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
    counts.ref_histograms["GGA"][50] = 1000

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
