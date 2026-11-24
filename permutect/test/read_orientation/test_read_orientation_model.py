import math

import numpy as np
import pytest
import torch
from scipy.integrate import quad
from scipy.stats import beta as beta_distribution
from scipy.stats import betabinom
from scipy.stats import binom
from scipy.stats import hypergeom

from permutect.read_orientation.f1r2_counts_reader import BASES
from permutect.read_orientation.f1r2_counts_reader import CONTEXTS
from permutect.read_orientation.f1r2_counts_reader import UNKNOWN_REF_F1R2
from permutect.read_orientation.f1r2_counts_reader import F1R2Counts
from permutect.read_orientation.f1r2_counts_reader import Sites
from permutect.read_orientation.maf_mixtures import MAF_GRID
from permutect.read_orientation.read_orientation_model import CONTEXT_INDICES
from permutect.read_orientation.read_orientation_model import FLAT_PRIOR
from permutect.read_orientation.read_orientation_model import GERMLINE_HET
from permutect.read_orientation.read_orientation_model import NEWTON_TOLERANCE
from permutect.read_orientation.read_orientation_model import NUM_BASES
from permutect.read_orientation.read_orientation_model import NUM_CONTEXTS
from permutect.read_orientation.read_orientation_model import NUM_GENOTYPES
from permutect.read_orientation.read_orientation_model import OTHER_STRAND_PROCESSES_P
from permutect.read_orientation.read_orientation_model import PAIR_CONTEXTS
from permutect.read_orientation.read_orientation_model import PAIRS_C
from permutect.read_orientation.read_orientation_model import POSSIBLE_PROCESSES_P
from permutect.read_orientation.read_orientation_model import SOMATIC_HET
from permutect.read_orientation.read_orientation_model import SQUAREM_OBJECTIVE_TOLERANCE
from permutect.read_orientation.read_orientation_model import SQUAREM_TOLERANCE
from permutect.read_orientation.read_orientation_model import VARIANT_ALLELE_FRACTION_PSEUDOCOUNTS_V2
from permutect.read_orientation.read_orientation_model import EMState
from permutect.read_orientation.read_orientation_model import GermlineHetLikelihoods
from permutect.read_orientation.read_orientation_model import HistogramMarginals
from permutect.read_orientation.read_orientation_model import OrientationEM
from permutect.read_orientation.read_orientation_model import ReadOrientationModel
from permutect.read_orientation.read_orientation_model import SiteTable
from permutect.read_orientation.read_orientation_model import damage_pseudocounts
from permutect.read_orientation.read_orientation_model import fit_read_orientation_model
from permutect.read_orientation.read_orientation_model import learned_pairs_w
from permutect.read_orientation.read_orientation_model import log_hypergeometric
from permutect.read_orientation.read_orientation_model import process_name
from permutect.read_orientation.read_orientation_model import reverse_complement
from permutect.read_orientation.read_orientation_model import unconstrained_damage
from permutect.read_orientation.read_orientation_model import variant_log_likelihoods_nv
from permutect.segmentation.emissions import EmissionModel

A, C, G, T = range(4)


def process(context: str, base: str) -> int:
    return NUM_BASES * CONTEXT_INDICES[context] + BASES.index(base)


def empty_counts() -> F1R2Counts:
    return F1R2Counts("SAMPLE", max_depth=1000, sites={context: Sites.empty() for context in CONTEXTS})


def table_of(keys, ref_f1r2_counts=None, maf_mixtures=None, context="ACG") -> SiteTable:
    """a SiteTable of sites of one context, each with its (alt base, alt count, alt F1R2 count, depth)"""
    return SiteTable.of({context: Sites.of(keys, [1] * len(keys), maf_mixtures, ref_f1r2_counts)})


def test_contexts_pairs_and_processes():
    assert reverse_complement("ACG") == "CGT"
    assert len(PAIR_CONTEXTS) == 32
    for context in CONTEXTS:
        assert PAIRS_C[CONTEXT_INDICES[context]] == PAIRS_C[CONTEXT_INDICES[reverse_complement(context)]]
    # the other strand's process of each process, which is an involution that keeps whether a process can happen
    assert torch.equal(OTHER_STRAND_PROCESSES_P[OTHER_STRAND_PROCESSES_P], torch.arange(len(OTHER_STRAND_PROCESSES_P)))
    assert torch.equal(POSSIBLE_PROCESSES_P[OTHER_STRAND_PROCESSES_P], POSSIBLE_PROCESSES_P)
    assert POSSIBLE_PROCESSES_P.sum() == 192
    # at a site of ACG with alt T, damage to the + strand is ACG -> ATG, and to the - strand CGT -> CAT
    assert process_name(process("ACG", "T")) == "ACG -> ATG"
    assert process_name(OTHER_STRAND_PROCESSES_P[process("ACG", "T")].item()) == "CGT -> CAT"
    # at a site of CGT with alt A, damage to the - strand, which reads ACG, is ACG -> ATG
    assert OTHER_STRAND_PROCESSES_P[process("CGT", "A")] == process("ACG", "T")


def test_log_hypergeometric():
    for a1, a2, n1, n2 in [(3, 2, 20, 30), (0, 5, 10, 40), (6, 0, 53, 47), (1, 0, 1, 0)]:
        result = log_hypergeometric(*(torch.tensor(x, dtype=torch.float64) for x in (a1, a2, n1, n2))).item()
        assert result == pytest.approx(hypergeom.logpmf(a1, n1 + n2, n1, a1 + a2), rel=1e-9)


def test_variant_likelihoods_share_one_allele_fraction():
    # given n1, a real variant's likelihood is that of one allele fraction for the reads of both strands
    capture = (3.0, 17.0)
    a1, a2, n1, n2 = 4, 1, 15, 45
    sites = table_of([[T, a1 + a2, a1, n1 + n2]], ref_f1r2_counts=[n1 - a1])
    result_v = variant_log_likelihoods_nv(sites, capture)[0]
    alpha, beta = VARIANT_ALLELE_FRACTION_PSEUDOCOUNTS_V2[SOMATIC_HET - 1].tolist()

    def integrand(f):
        return beta_distribution.pdf(f, alpha, beta) * binom.pmf(a1, n1, f) * binom.pmf(a2, n2, f)

    expected = math.log(quad(integrand, 0, 1, epsabs=1e-14)[0])
    assert result_v[SOMATIC_HET - 1].item() == pytest.approx(expected, rel=1e-6)

    # without n1, it's the marginal over n1 under the capture bias
    unknown = table_of([[T, a1 + a2, a1, n1 + n2]])
    marginal = sum(
        betabinom.pmf(m, n1 + n2, *capture)
        * math.exp(variant_log_likelihoods_nv(table_of([[T, a1 + a2, a1, n1 + n2]], [m - a1]), capture)[0, 1])
        for m in range(a1, n1 + n2 - a2 + 1)
    )
    assert variant_log_likelihoods_nv(unknown, capture)[0, SOMATIC_HET - 1].item() == pytest.approx(
        math.log(marginal), rel=1e-6
    )


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
    alt_counts, depths = torch.tensor([0, 3, 15, 3]), torch.tensor([30, 30, 30, 30])
    mixtures = torch.tensor([MAF_GRID.index(0.01), MAF_GRID.index(0.1), MAF_GRID.index(0.5), len(MAF_GRID)])
    for alt_count, depth, mixture_index, result in zip(
        alt_counts.tolist(), depths.tolist(), mixtures.tolist(), germline_het(alt_counts, depths, mixtures).tolist()
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
    sites = table_of([[C, 0, 0, 40], [T, 5, 5, 50], [G, 30, 2, 60]], maf_mixtures=[0, 13, 7])
    germline_het = GermlineHetLikelihoods(EmissionModel(), torch.eye(len(MAF_GRID), dtype=torch.float64))
    without_nv = variant_log_likelihoods_nv(sites, (10.0, 10.0))
    with_nv = variant_log_likelihoods_nv(sites, (10.0, 10.0), germline_het)
    assert torch.equal(with_nv[:, 1:], without_nv[:, 1:])
    alpha, beta = VARIANT_ALLELE_FRACTION_PSEUDOCOUNTS_V2[GERMLINE_HET - 1].tolist()
    gatk_n = torch.tensor(betabinom.logpmf(sites.alt_counts_n.numpy(), sites.depths_n.numpy(), alpha, beta))
    expected_n = germline_het(sites.alt_counts_n, sites.depths_n, sites.maf_mixtures_n)
    assert torch.allclose(with_nv[:, 0], without_nv[:, 0] - gatk_n + expected_n)


def test_histogram_marginals_are_sums_over_n1():
    capture = (2.0, 8.0)
    depths = torch.tensor([1, 7, 30])
    plus, minus = (0.5, 40.0), (0.2, 300.0)
    marginals = HistogramMarginals(depths, capture)
    result_tqd = marginals.log_likelihoods_tqd(*(torch.tensor([x], dtype=torch.float64) for x in (*plus, *minus)))
    for t, (k1, k2) in enumerate(HistogramMarginals.PATTERNS):
        for d_index, d in enumerate(depths.tolist()):
            expected = sum(
                betabinom.pmf(n1, d, *capture) * betabinom.pmf(k1, n1, *plus) * betabinom.pmf(k2, d - n1, *minus)
                for n1 in range(d + 1)
            )
            assert result_tqd[t, 0, d_index].item() == pytest.approx(math.log(expected), rel=1e-9)


# genotypes of simulated sites: HOM_REF, GERMLINE_HET, SOMATIC_HET, HOM_VAR
SIMULATED_GENOTYPE_PRIORS = (0.985, 0.01, 0.004, 0.001)
SIMULATED_ALLELE_FRACTIONS = tuple(map(tuple, VARIANT_ALLELE_FRACTION_PSEUDOCOUNTS_V2.tolist()))  # the model's


def simulate_sites(rng, context: str, num_sites: int, depth: int, capture, damage: dict, background) -> Sites:
    """
    Sites of a context drawn from the generative model, as Mutect2 would count them: n1 from the capture bias
    (alpha, beta); a genotype; at a real variant, alt reads of the context's first other base at its allele fraction; and
    damage, in every genotype, reads of each other base from the rates of the + strand process on the F1R2 reads and the
    - strand process on the F2R1 reads, Beta(mean, concentration) of the processes in damage, or of background.  A site
    without alt reads goes to the ref histogram, a site with one read of its most common alt base to the alt histogram,
    and others to an alt table, with that base's counts.
    """
    ref = BASES.index(context[1])
    others = [b for b in range(NUM_BASES) if b != ref]
    n1 = rng.binomial(depth, rng.beta(*capture, size=num_sites))
    n2 = depth - n1
    genotypes = rng.choice(NUM_GENOTYPES, size=num_sites, p=SIMULATED_GENOTYPE_PRIORS)
    variant_af = np.zeros(num_sites)
    for g, pseudocounts in enumerate(SIMULATED_ALLELE_FRACTIONS, start=1):
        variant_af[genotypes == g] = rng.beta(*pseudocounts, size=(genotypes == g).sum())
    reads1, reads2 = np.zeros((num_sites, NUM_BASES), dtype=int), np.zeros((num_sites, NUM_BASES), dtype=int)
    reads1[:, others[0]] = rng.binomial(n1, variant_af)
    reads2[:, others[0]] = rng.binomial(n2, variant_af)
    rc_context = reverse_complement(context)
    for base in others:
        for reads, strand_context, strand_base, n in (
            (reads1, context, BASES[base], n1),
            (reads2, rc_context, BASES[NUM_BASES - 1 - base], n2),
        ):
            mean, concentration = damage.get((strand_context, strand_base), background)
            rates = rng.beta(mean * concentration, (1 - mean) * concentration, size=num_sites)
            reads[:, base] += rng.binomial(
                n - reads1.sum(1) * (reads is reads1) - reads2.sum(1) * (reads is reads2), rates
            )
    totals = reads1 + reads2
    alt_bases = totals.argmax(axis=1)
    alt_counts = totals[np.arange(num_sites), alt_bases]
    alt_f1r2 = reads1[np.arange(num_sites), alt_bases]
    alt_bases = np.where(alt_counts == 0, ref, alt_bases)
    # an alt table site's depth is its ref reads plus those of its alt base
    other_reads1 = reads1.sum(1) - alt_f1r2
    other_reads2 = reads2.sum(1) - (alt_counts - alt_f1r2)
    in_table = alt_counts >= 2
    depths = np.where(in_table, depth - other_reads1 - other_reads2, depth)
    ref_f1r2 = np.where(in_table, n1 - reads1.sum(1), UNKNOWN_REF_F1R2)
    keys = np.column_stack([alt_bases, alt_counts, alt_f1r2, depths])
    return Sites.of(keys, np.ones(num_sites), None, ref_f1r2)


def simulated_counts(rng, contexts, num_sites: int, depth: int, capture, damage: dict, background) -> F1R2Counts:
    counts = empty_counts()
    for context in contexts:
        counts.sites[context] = simulate_sites(rng, context, num_sites, depth, capture, damage, background)
    return counts


def damage_means_and_concentrations(model: ReadOrientationModel):
    alpha_ca, beta_ca = model.damage_pseudocounts_ca2.unbind(-1)
    return alpha_ca / (alpha_ca + beta_ca), alpha_ca + beta_ca


def mean_of(model: ReadOrientationModel, context: str, base: str) -> float:
    means_ca, _ = damage_means_and_concentrations(model)
    return means_ca[CONTEXT_INDICES[context], BASES.index(base)].item()


# FFPE-like capture bias and deamination of the C of ACG, at rates of mean 0.01, on a background of mean 1e-4
CAPTURE = (0.2 * 30, 0.8 * 30)
DAMAGE = {("ACG", "T"): (0.01, 30.0)}
BACKGROUND = (1e-4, 1000.0)


def test_em_recovers_simulated_damage():
    rng = np.random.default_rng(42)
    counts = simulated_counts(rng, ("ACG", "CGT", "ACA", "TGT"), 100_000, 100, CAPTURE, DAMAGE, BACKGROUND)
    model = fit_read_orientation_model(counts, CAPTURE, convergence_threshold=1e-6, max_iterations=50)
    means_ca, concentrations_ca = damage_means_and_concentrations(model)
    # ACG -> ATG, which damages + strands at sites of ACG and - strands at sites of CGT
    assert mean_of(model, "ACG", "T") == pytest.approx(0.01, rel=0.2)
    assert concentrations_ca[CONTEXT_INDICES["ACG"], T].item() == pytest.approx(30, rel=0.5)
    # its reverse complement, a different process, and the other processes are background
    for context, base in (("CGT", "A"), ("ACG", "A"), ("CGT", "G"), ("ACA", "T"), ("TGT", "A")):
        assert mean_of(model, context, base) == pytest.approx(1e-4, rel=0.5), (context, base)
    # the genotype priors of the learned pairs
    for context in ("ACG", "ACA"):
        priors_s = model.genotype_priors_cs[CONTEXT_INDICES[context]]
        assert priors_s.tolist() == pytest.approx(SIMULATED_GENOTYPE_PRIORS, rel=0.25, abs=3e-4), context


def simulated_em(seed: int) -> OrientationEM:
    counts = simulated_counts(np.random.default_rng(seed), ("ACG", "ACA"), 20_000, 60, CAPTURE, DAMAGE, BACKGROUND)
    sites = SiteTable.of(counts.sites)
    return OrientationEM(sites, learned_pairs_w(sites), CAPTURE)


def test_damage_m_step_reaches_each_pairs_minimum():
    # From EM's starting point, far from the optimum, Newton's method leaves each pair of a process and its other
    # strand's process at a minimum of its objective: the Hessian is positive definite there, Newton's step is within
    # its tolerance, and no pair's objective is worse than where it started
    em = simulated_em(11)
    state = em.initial_state()
    hyper = (state.hyper_mean_2, state.hyper_variance_2)
    posteriors = em.e_step(state.priors_ws, state.unconstrained_q2)
    u_q2, _ = em.damage_m_step(state.unconstrained_q2, posteriors, *hyper)
    objectives_w, gradient_q2, block_w44 = em.objectives_gradient_and_hessian_blocks(u_q2, posteriors, *hyper)
    assert torch.all(torch.linalg.eigvalsh(block_w44)[:, 0] > 0)
    gradient_w4 = torch.cat([gradient_q2[em.firsts_w], gradient_q2[em.seconds_w]], dim=-1)
    newton_step_w4 = torch.linalg.solve(block_w44, gradient_w4.unsqueeze(-1)).squeeze(-1)
    assert newton_step_w4.abs().max() <= NEWTON_TOLERANCE
    assert torch.all(objectives_w <= em.pair_objectives(state.unconstrained_q2, posteriors, *hyper))


def test_squarem_reaches_the_fixed_point_of_em():
    # SQUAREM's result is where plain EM stays, and it fits at least as well as as many iterations of plain EM, which
    # haven't converged by then
    em = simulated_em(12)
    result = em.run(convergence_threshold=1e-6, max_iterations=100)
    assert result.converged
    state = EMState(
        result.priors_ws,
        result.damage_unconstrained_p2[em.learned_processes_q],
        result.hyper_mean_2,
        result.hyper_variance_2,
    )
    next_state = em.em_map(state)
    assert torch.allclose(next_state.priors_ws, state.priors_ws, rtol=0, atol=1e-6)
    assert abs(em.log_posterior(next_state) - em.log_posterior(state)) <= SQUAREM_OBJECTIVE_TOLERANCE
    plain_state = em.initial_state()
    for _ in range(result.num_iterations):
        plain_state = em.em_map(plain_state)
    assert em.log_posterior(state) >= em.log_posterior(plain_state) - SQUAREM_TOLERANCE


def mirrored(counts: F1R2Counts) -> F1R2Counts:
    """
    The counts as if the reference were the other strand: each site of c is a site of rc(c), its alt base the
    complement, and its F1R2 reads its F2R1 reads
    """
    result = empty_counts()
    for context, sites in counts.sites.items():
        alt_bases_n, alt_counts_n, alt_f1r2_n, depths_n = sites.keys_n4.unbind(-1)
        keys_n4 = torch.stack([NUM_BASES - 1 - alt_bases_n, alt_counts_n, alt_counts_n - alt_f1r2_n, depths_n], -1)
        known_n = sites.ref_f1r2_counts_n != UNKNOWN_REF_F1R2
        ref_f1r2_n = torch.where(known_n, depths_n - alt_counts_n - sites.ref_f1r2_counts_n, UNKNOWN_REF_F1R2)
        result.sites[reverse_complement(context)] = Sites.of(keys_n4, sites.counts_n, sites.maf_mixtures_n, ref_f1r2_n)
    return result


def test_physical_processes_do_not_depend_on_which_strand_is_the_reference():
    # the regression test of reverse complements: a process's parameters come from its damage on + strands at sites
    # of its context and on - strands at sites of the reverse complement, whichever strand the reference is
    rng = np.random.default_rng(7)
    counts = simulated_counts(rng, ("ACG", "CGT"), 50_000, 60, CAPTURE, DAMAGE, BACKGROUND)
    model = fit_read_orientation_model(counts, CAPTURE, convergence_threshold=1e-8, max_iterations=100)
    mirrored_model = fit_read_orientation_model(
        mirrored(counts), CAPTURE[::-1], convergence_threshold=1e-8, max_iterations=100
    )
    # Every process's mean rate, and the concentration of the one with damage, to within EM's tolerance.  The
    # concentrations of the background processes, whose rates are near 0, barely change their likelihoods, and EM
    # leaves them where it stops.
    (means_ca, concentrations_ca), (mirrored_means_ca, mirrored_concentrations_ca) = (
        damage_means_and_concentrations(m) for m in (model, mirrored_model)
    )
    assert torch.allclose(means_ca, mirrored_means_ca, rtol=1e-2)
    acg = CONTEXT_INDICES["ACG"]
    assert concentrations_ca[acg, T].item() == pytest.approx(mirrored_concentrations_ca[acg, T].item(), rel=1e-2)
    assert torch.allclose(model.genotype_priors_cs, mirrored_model.genotype_priors_cs, rtol=1e-3, atol=1e-6)
    # each context's sites are its reverse complement's in the mirror
    assert torch.equal(model.num_sites_c, mirrored_model.num_sites_c[REVERSE_COMPLEMENTS])


REVERSE_COMPLEMENTS = [CONTEXT_INDICES[reverse_complement(context)] for context in CONTEXTS]


def test_capture_bias_alone_is_not_damage():
    # strong capture bias, as in FFPE exomes, puts most alt reads of real variants in F2R1 orientation, but in both
    # strands' damage the same, background
    rng = np.random.default_rng(3)
    capture = (0.15 * 30, 0.85 * 30)
    counts = simulated_counts(rng, ("ACG", "CGT"), 50_000, 100, capture, {}, BACKGROUND)
    model = fit_read_orientation_model(counts, capture, convergence_threshold=1e-6, max_iterations=50)
    means_ca, _ = damage_means_and_concentrations(model)
    learned = [process("ACG", b) for b in "ATG"] + [process("CGT", b) for b in "ACT"]
    assert torch.all(means_ca.flatten()[learned] < 3e-4)


def test_pairs_without_enough_data_keep_the_flat_prior():
    rng = np.random.default_rng(0)
    counts = simulated_counts(rng, ("ACG",), 20_000, 60, CAPTURE, DAMAGE, BACKGROUND)
    # the pair of GGA has sites without alt reads but none with two or more
    counts.sites["GGA"] = Sites.of([[G, 0, 0, 51]], [1000])
    model = fit_read_orientation_model(counts, CAPTURE)
    _, concentrations_ca = damage_means_and_concentrations(model)
    for context in ("TCC", "GGA", "AAA", "TTT"):
        c = CONTEXT_INDICES[context]
        assert torch.all(model.genotype_priors_cs[c] == FLAT_PRIOR)
        assert model.num_sites_c[c] == model.num_alt_sites_c[c] == 0
        # the damage distribution of the hierarchical prior's mean, the same for every process without data
        assert torch.equal(model.damage_pseudocounts_ca2[c], model.damage_pseudocounts_ca2[CONTEXT_INDICES["AAA"]])
    acg, cgt = CONTEXT_INDICES["ACG"], CONTEXT_INDICES["CGT"]
    assert model.num_sites_c[acg] == 20_000 and model.num_sites_c[cgt] == 0
    assert torch.equal(model.genotype_priors_cs[acg], model.genotype_priors_cs[cgt])
    assert not torch.all(model.genotype_priors_cs[acg] == FLAT_PRIOR)


def random_model(rng, sample: str = "TUMOR") -> ReadOrientationModel:
    means = rng.uniform(1e-5, 0.2, size=(NUM_CONTEXTS, NUM_BASES))
    concentrations = rng.uniform(5, 500, size=(NUM_CONTEXTS, NUM_BASES))
    pseudocounts_ca2 = torch.tensor(np.stack([means * concentrations, (1 - means) * concentrations], -1))
    priors_cs = torch.tensor(rng.dirichlet(np.ones(NUM_GENOTYPES), size=NUM_CONTEXTS))
    capture = torch.tensor([3.0, 7.0])
    return ReadOrientationModel(sample, pseudocounts_ca2, priors_cs, capture, torch.tensor(rng.integers(1000, size=64)))


def test_model_save_and_load(tmp_path):
    model = random_model(np.random.default_rng(0))
    model.save(tmp_path / "model.pt")
    loaded = ReadOrientationModel.load(tmp_path / "model.pt")
    assert loaded.sample == "TUMOR"
    for name, buffer in model.named_buffers():
        assert torch.equal(buffer, loaded.get_buffer(name)), name


def test_model_needs_the_damage_of_every_process():
    with pytest.raises(ValueError, match="damage pseudocounts have shape"):
        ReadOrientationModel("TUMOR", torch.ones(NUM_CONTEXTS - 1, NUM_BASES, 2))


def test_model_processes():
    # ACG with alt T, then its reverse complement CGT with alt A, then ACG with alt C, its ref base, and a deletion
    left_b, ref_b, right_b, alt_b = torch.tensor([[A, C, G, T], [C, G, T, A], [A, C, G, C], [A, C, G, 4]]).T
    processes_bo, snvs_b = ReadOrientationModel.processes_bo(left_b, ref_b, right_b, alt_b)
    acg_atg, cgt_cat = process("ACG", "T"), process("CGT", "A")
    assert processes_bo[:2].tolist() == [[acg_atg, cgt_cat], [cgt_cat, acg_atg]]
    assert snvs_b.tolist() == [True, True, False, False]


def test_model_damage_likelihoods_are_those_of_the_em():
    # a model's likelihood of a variant's reads from damage, plus the hypergeometric factor it leaves out, is the EM's
    # likelihood of an alt table site's reads given HOM_REF, with the same damage distributions
    rng = np.random.default_rng(5)
    keys = [[T, 5, 5, 50], [T, 3, 0, 70], [A, 12, 6, 100], [G, 30, 2, 60]]
    sites = table_of(keys, ref_f1r2_counts=[20, 30, 44, 10])
    learned_w = learned_pairs_w(sites) | True
    em = OrientationEM(sites, learned_w, (10.0, 10.0))
    u_q2 = torch.tensor(rng.normal(size=(len(em.learned_processes_q), 2)))
    _, _, em_n = em.damage_log_likelihoods(u_q2)
    u_p2 = torch.zeros(NUM_CONTEXTS * NUM_BASES, 2, dtype=torch.float64)
    u_p2[em.learned_processes_q] = u_q2
    alpha_p, beta_p = damage_pseudocounts(u_p2)
    model = ReadOrientationModel("TUMOR", torch.stack([alpha_p, beta_p], -1).reshape(NUM_CONTEXTS, NUM_BASES, 2))
    a = em.alt_table
    context = torch.tensor([BASES.index(b) for b in "ACG"]).unsqueeze(-1).expand(3, len(a))
    counts = [x.double() for x in (a.alt_f1r2_counts_n, a.alt_f2r1_counts_n, a.f1r2_depths_n, a.f2r1_depths_n)]
    model_n, snvs_n = model.damage_log_likelihoods_b(*context, a.alt_bases_n, *counts)
    assert torch.all(snvs_n)
    assert torch.allclose(model_n + log_hypergeometric(*counts), em_n)


def test_model_damage_likelihoods_of_reads_of_unknown_orientation():
    # ACG -> ATG of mean rate 0.1, every other process 1e-5, and balanced capture
    pseudocounts_ca2 = torch.tensor([1e-5, 1 - 1e-5], dtype=torch.float64).expand(NUM_CONTEXTS, NUM_BASES, 2) * 1000
    pseudocounts_ca2 = pseudocounts_ca2.clone()
    pseudocounts_ca2[CONTEXT_INDICES["ACG"], T] = torch.tensor([2.0, 18.0])
    model = ReadOrientationModel("TUMOR", pseudocounts_ca2, capture_pseudocounts_2=torch.tensor([10.0, 10.0]))
    context = [torch.tensor([b] * 4) for b in (A, C, G)]
    alt = torch.tensor([T] * 4)
    # a1, a2, n1, n2, then depth and alt count: all reads oriented; 6 alt reads of 100, 4 of them F1R2 among 60
    # oriented reads; and 6 alt reads none of them oriented
    a1, a2, n1, n2 = (
        torch.tensor(x, dtype=torch.float64) for x in ([6, 4, 0, 0], [0, 0, 0, 0], [53, 30, 30, 30], [47, 30, 30, 30])
    )
    depths, alt_counts = torch.tensor([100.0, 100, 100, 60]), torch.tensor([6.0, 6, 6, 0])
    oriented_b, _ = model.damage_log_likelihoods_b(*context, alt, a1, a2, n1, n2)
    with_totals_b, _ = model.damage_log_likelihoods_b(*context, alt, a1, a2, n1, n2, depths, alt_counts)
    # without unoriented reads, the same
    assert with_totals_b[0].item() == pytest.approx(oriented_b[0].item())
    assert with_totals_b[3].item() == pytest.approx(oriented_b[3].item())
    # unoriented alt reads aren't free: 6 of 40, at the rate 0.02 expected of ACG -> ATG given that none of the
    # 60 oriented reads is damaged, cost about 3 nats, and are likelier where 4 of 30 F1R2 reads are
    assert with_totals_b[2] < oriented_b[2] - 3
    assert with_totals_b[1] > with_totals_b[2] + 3


def test_unconstrained_damage_round_trip():
    u = unconstrained_damage(0.01, 30.0)
    alpha, beta = damage_pseudocounts(u.unsqueeze(0))
    assert (alpha / (alpha + beta)).item() == pytest.approx(0.01) and (alpha + beta).item() == pytest.approx(30.0)
