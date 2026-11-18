import math

import numpy as np
import pytest
import torch
from scipy.stats import betabinom

from permutect.read_orientation.f1r2_counts_reader import BASES
from permutect.read_orientation.f1r2_counts_reader import CONTEXTS
from permutect.read_orientation.f1r2_counts_reader import F1R2Counts
from permutect.read_orientation.f1r2_counts_reader import Sites
from permutect.read_orientation.maf_mixtures import MAF_GRID
from permutect.read_orientation.read_orientation_model import ALLELE_FRACTION_PSEUDOCOUNTS_K2
from permutect.read_orientation.read_orientation_model import ARTIFACT_STATES
from permutect.read_orientation.read_orientation_model import CANONICAL_CONTEXTS
from permutect.read_orientation.read_orientation_model import F1R2_FRACTION_PSEUDOCOUNTS_K2
from permutect.read_orientation.read_orientation_model import GERMLINE_HET
from permutect.read_orientation.read_orientation_model import HOM_REF
from permutect.read_orientation.read_orientation_model import NUM_ARTIFACT_STATES
from permutect.read_orientation.read_orientation_model import NUM_STATES
from permutect.read_orientation.read_orientation_model import REVERSE_COMPLEMENT_STATES_K
from permutect.read_orientation.read_orientation_model import SOMATIC_HET
from permutect.read_orientation.read_orientation_model import STATES
from permutect.read_orientation.read_orientation_model import ContextPrior
from permutect.read_orientation.read_orientation_model import GermlineHetLikelihoods
from permutect.read_orientation.read_orientation_model import ReadOrientationModel
from permutect.read_orientation.read_orientation_model import artifact_states_bo
from permutect.read_orientation.read_orientation_model import canonical_sites
from permutect.read_orientation.read_orientation_model import flat_prior
from permutect.read_orientation.read_orientation_model import learn_prior
from permutect.read_orientation.read_orientation_model import learn_priors
from permutect.read_orientation.read_orientation_model import log_likelihoods
from permutect.read_orientation.read_orientation_model import non_artifact_f1r2_pseudocounts_k2
from permutect.read_orientation.read_orientation_model import posteriors
from permutect.read_orientation.read_orientation_model import reverse_complement
from permutect.segmentation.emissions import EmissionModel

A, C, G, T = range(4)


def empty_counts() -> F1R2Counts:
    return F1R2Counts("SAMPLE", max_depth=1000, sites={context: Sites.empty() for context in CONTEXTS})


def add_sites(counts: F1R2Counts, context: str, alt_bases_n, alt_counts_n, alt_f1r2_counts_n, depths_n):
    keys_n4 = np.column_stack([alt_bases_n, alt_counts_n, alt_f1r2_counts_n, depths_n])
    counts.sites[context] += Sites.of(keys_n4, np.ones(len(keys_n4)))


def emission_het_log_likelihood(emission_model: EmissionModel, alt_count: int, depth: int, maf: float) -> float:
    """a germline het's log likelihood from an emission model: its alt allele is on either haplotype, with probability 1/2"""
    with torch.no_grad():
        _, minor, major, _ = emission_model.phased_genotype_log_likelihoods(
            *(torch.tensor(float(x), dtype=torch.float64) for x in (alt_count, depth, 0, maf))
        )
    return math.log(0.5 * math.exp(minor.item()) + 0.5 * math.exp(major.item()))


def test_germline_het_likelihoods():
    emission_model = EmissionModel(contamination=0.02, het_concentration=50.0, reference_bias=1.1)
    # the grid MAFs alone, then a mixture of 30% MAF 0.1 and 70% MAF 0.5
    mixture = torch.zeros(1, len(MAF_GRID), dtype=torch.float64)
    mixture[0, [MAF_GRID.index(0.1), MAF_GRID.index(0.5)]] = torch.tensor([0.3, 0.7], dtype=torch.float64)
    germline_het = GermlineHetLikelihoods(
        emission_model, torch.cat([torch.eye(len(MAF_GRID), dtype=torch.float64), mixture])
    )
    keys = [[C, 0, 0, 30], [T, 3, 1, 30], [T, 15, 8, 30], [T, 3, 1, 30]]
    sites = Sites.of(
        keys, [1] * len(keys), [MAF_GRID.index(0.01), MAF_GRID.index(0.1), MAF_GRID.index(0.5), len(MAF_GRID)]
    )

    for (_, alt_count, _, depth), mixture_index, result in zip(
        sites.keys_n4.tolist(), sites.maf_mixtures_n.tolist(), germline_het(sites).tolist()
    ):
        if mixture_index < len(MAF_GRID):
            expected = emission_het_log_likelihood(emission_model, alt_count, depth, MAF_GRID[mixture_index])
        else:
            likelihood_01, likelihood_05 = (
                math.exp(emission_het_log_likelihood(emission_model, alt_count, depth, maf)) for maf in (0.1, 0.5)
            )
            expected = math.log(0.3 * likelihood_01 + 0.7 * likelihood_05)
        assert result == pytest.approx(expected, rel=1e-9)

    # in loss of heterozygosity, a het without alt reads is likely, its alt allele on the lost haplotype, unlike at MAF 1/2
    assert (
        emission_het_log_likelihood(emission_model, 0, 30, 0.01)
        > emission_het_log_likelihood(emission_model, 0, 30, 0.5) + 5
    )


def test_germline_het_likelihoods_replace_only_the_germline_het_allele_fraction():
    sites = Sites.of([[C, 0, 0, 40], [T, 5, 5, 50], [G, 30, 2, 60]], [1, 1, 1], [0, 13, 7])
    germline_het = GermlineHetLikelihoods(EmissionModel(), torch.eye(len(MAF_GRID), dtype=torch.float64))
    without_nk = log_likelihoods(sites, ref_base=C)
    with_nk = log_likelihoods(sites, ref_base=C, germline_het=germline_het)
    others = [k for k in range(NUM_STATES) if k != GERMLINE_HET]
    assert torch.equal(with_nk[:, others], without_nk[:, others])
    # GATK's beta-binomial allele fraction gives way, and the F1R2 term stays
    alpha, beta = ALLELE_FRACTION_PSEUDOCOUNTS_K2[GERMLINE_HET].tolist()
    gatk_n = torch.tensor(betabinom.logpmf(sites.alt_counts_n.numpy(), sites.depths_n.numpy(), alpha, beta))
    assert torch.allclose(with_nk[:, GERMLINE_HET], without_nk[:, GERMLINE_HET] - gatk_n + germline_het(sites))


def test_non_artifact_f1r2_pseudocounts(tmp_path):
    pseudocounts_k2 = non_artifact_f1r2_pseudocounts_k2(3.0, 17.0)
    assert torch.equal(pseudocounts_k2[:NUM_ARTIFACT_STATES], F1R2_FRACTION_PSEUDOCOUNTS_K2[:NUM_ARTIFACT_STATES])
    assert pseudocounts_k2[NUM_ARTIFACT_STATES:].tolist() == [[3.0, 17.0]] * (NUM_STATES - NUM_ARTIFACT_STATES)

    # a model saves them, and gives every non-artifact variant their likelihoods
    path = tmp_path / "model.pt"
    ReadOrientationModel("SAMPLE", learn_priors(empty_counts()), f1r2_fraction_pseudocounts_k2=pseudocounts_k2).save(
        path
    )
    alt_counts_b, alt_f1r2_counts_b = torch.tensor([10, 10, 3]), torch.tensor([1, 5, 3])
    result_b = ReadOrientationModel.load(path).non_artifact_f1r2_log_likelihoods_b(alt_counts_b, alt_f1r2_counts_b)
    assert np.allclose(result_b.numpy(), betabinom.logpmf(alt_f1r2_counts_b.numpy(), alt_counts_b.numpy(), 3.0, 17.0))


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


def random_priors(rng) -> dict[str, ContextPrior]:
    """a random prior of each canonical context, and the equivalent prior of its reverse complement"""
    priors = {}
    for context in CANONICAL_CONTEXTS:
        prior_k = flat_prior(BASES.index(context[1])) * torch.from_numpy(rng.uniform(0.1, 1.0, NUM_STATES))
        num_sites, num_alt_sites = rng.integers(1000), rng.integers(100)
        prior = ContextPrior(context, prior_k / prior_k.sum(), int(num_sites), int(num_alt_sites))
        priors[context], priors[reverse_complement(context)] = prior, prior.reverse_complement()
    return priors


def test_model_save_and_load(tmp_path):
    model = ReadOrientationModel("TUMOR", random_priors(np.random.default_rng(0)))
    model.save(tmp_path / "model.pt")
    loaded = ReadOrientationModel.load(tmp_path / "model.pt")
    assert loaded.sample == "TUMOR"
    for name, buffer in model.named_buffers():
        assert torch.equal(buffer, loaded.get_buffer(name)), name


def test_model_needs_the_priors_of_every_context():
    priors = random_priors(np.random.default_rng(0))
    del priors["ACG"]
    with pytest.raises(ValueError, match="all 64 contexts"):
        ReadOrientationModel("TUMOR", priors)


def test_model_log_priors():
    priors = random_priors(np.random.default_rng(1))
    model = ReadOrientationModel("TUMOR", priors)
    # ACG with alt T, then its reverse complement CGT with alt A, whose F1R2 artifacts are ACG's F2R1 artifacts and vice
    # versa, then ACG with alt C, its ref base, and a deletion, neither of which can be a read orientation artifact
    left_b, ref_b, right_b, alt_b = torch.tensor([[A, C, G, T], [C, G, T, A], [A, C, G, C], [A, C, G, 4]]).T
    result_bo = model.log_priors_bo(left_b, ref_b, right_b, alt_b)
    expected_o = priors["ACG"].prior_k[[STATES.index("F1R2_T"), STATES.index("F2R1_T")]].log()
    assert torch.allclose(result_bo[0], expected_o)
    assert torch.allclose(result_bo[1], expected_o.flip(0))
    assert torch.all(result_bo[2:] == -math.inf)


def test_model_likelihoods_are_those_of_the_em():
    # the log likelihoods of an artifact state are the sum of the model's spectrum and F1R2 log likelihoods, and those of
    # a somatic het include its non-artifact F1R2 log likelihoods
    keys = [[T, 5, 5, 50], [T, 1, 0, 7], [A, 12, 6, 100], [G, 30, 2, 60]]
    sites = Sites.of(keys, [1] * len(keys))
    em_nk = log_likelihoods(sites, ref_base=C)
    model = ReadOrientationModel("TUMOR", random_priors(np.random.default_rng(2)))
    alt_counts_n, alt_f1r2_counts_n, depths_n = (
        x.double() for x in (sites.alt_counts_n, sites.alt_f1r2_counts_n, sites.depths_n)
    )
    artifact_no = model.artifact_spectrum_log_likelihoods_bo(sites.alt_bases_n, depths_n, alt_counts_n)
    artifact_no += model.artifact_f1r2_log_likelihoods_bo(sites.alt_bases_n, alt_counts_n, alt_f1r2_counts_n)
    assert torch.allclose(artifact_no, em_nk.gather(1, artifact_states_bo(sites.alt_bases_n)))

    non_artifact_f1r2_n = model.non_artifact_f1r2_log_likelihoods_b(alt_counts_n, alt_f1r2_counts_n)
    alpha, beta = ALLELE_FRACTION_PSEUDOCOUNTS_K2[SOMATIC_HET].tolist()
    somatic_spectrum_n = betabinom.logpmf(alt_counts_n.numpy(), depths_n.numpy(), alpha, beta)
    assert np.allclose(em_nk[:, SOMATIC_HET].numpy(), somatic_spectrum_n + non_artifact_f1r2_n.numpy())
