import math

import numpy as np
import pytest
import torch

from permutect.architecture.posterior_model_priors import PosteriorModelPriors
from permutect.architecture.posterior_model_priors import context_dependent_snv_log_rates_sc
from permutect.architecture.posterior_model_priors import convert_rrra_tensor_to_sc
from permutect.data.batch import Batch
from permutect.data.datum import RAW_READS_ARRAY_DTYPE
from permutect.data.datum import Data
from permutect.data.datum import Datum
from permutect.utils.enums import Call
from permutect.utils.enums import Variation


def make_batch(allele_frequencies, values: dict = None) -> Batch:
    """
    a batch with the given allele frequencies, and other data fields set to the given values or else 0
    """
    data = []
    for _ in allele_frequencies:
        # ref and alt haplotypes of length 3 follow the scalar int elements
        datum = Datum(
            int_array=np.zeros(Data.HAPLOTYPES_START_IDX + 6),
            float_array=np.zeros(Data.NUM_SCALAR_FLOAT_ELEMENTS),
            reads_re=np.zeros((0, 2), dtype=RAW_READS_ARRAY_DTYPE),
        )
        for data_field, value in (values or {}).items():
            datum.set(data_field, value)
        data.append(datum)
    batch = Batch(data)
    # set the batch's float32 log allele frequencies directly, since a Datum would round them to float16
    log_allele_frequencies = torch.log(torch.tensor(allele_frequencies, dtype=torch.float32))
    batch.float_tensor[:, Data.LOG_ALLELE_FREQUENCY.idx] = log_allele_frequencies
    return batch


def test_germline_log_prior_is_accurate_for_small_allele_frequencies():
    allele_frequencies = [2e-8, 5e-8, 1e-7, 1e-6, 1e-5, 1e-3, 0.1, 0.5, 1.0]
    priors = PosteriorModelPriors(
        variant_log_prior=-10, artifact_log_prior=-10, no_germline_mode=False, device=torch.device("cpu")
    )
    log_priors_bc = priors.log_priors_bc(make_batch(allele_frequencies))

    # before the softmax normalization, the sequencing error log prior is 0 and the germline log prior is the log
    # probability of not being hom ref, 1 - (1 - af)^2
    afs = torch.tensor(allele_frequencies, dtype=torch.float32).double()
    expected = torch.log1p(-torch.square(1 - afs))
    germline_log_prior = log_priors_bc[:, Call.GERMLINE] - log_priors_bc[:, Call.SEQ_ERROR]
    torch.testing.assert_close(germline_log_prior.double(), expected, rtol=0, atol=1e-4)


@pytest.mark.parametrize("no_germline_mode", [False, True])
def test_contamination_log_prior_is_the_probability_that_the_contaminant_carries_the_allele(no_germline_mode):
    # as for germline variants, but even in no-germline mode, and not learned in the M step
    allele_frequencies = [1e-6, 1e-3, 0.1, 0.5]
    priors = PosteriorModelPriors(
        variant_log_prior=-10, artifact_log_prior=-10, no_germline_mode=no_germline_mode, device=torch.device("cpu")
    )
    priors.disable_context_dependent_snv_priors()
    totals_rrra = PosteriorModelPriors.initialize_snv_context_totals_rrra()
    priors.update_priors_m_step(torch.ones(len(Variation), len(Call)), totals_rrra, totals_rrra, 10)
    log_priors_bc = priors.log_priors_bc(make_batch(allele_frequencies))

    afs = torch.tensor(allele_frequencies, dtype=torch.float32).double()
    contamination_log_prior = log_priors_bc[:, Call.CONTAMINATION] - log_priors_bc[:, Call.SEQ_ERROR]
    expected = torch.log1p(-torch.square(1 - afs))
    torch.testing.assert_close(contamination_log_prior.double(), expected, rtol=0, atol=1e-4)


def test_mnvs_have_their_own_priors():
    priors = PosteriorModelPriors(
        variant_log_prior=-10, artifact_log_prior=-8, no_germline_mode=False, device=torch.device("cpu")
    )
    # initially the same as the SNV priors
    torch.testing.assert_close(priors.log_priors_vc[Variation.MNV], priors.log_priors_vc[Variation.SNV])

    # MNVs don't use the context-dependent somatic priors of SNVs
    with torch.no_grad():
        priors.somatic_snv_log_priors_rrra.fill_(-20)
        priors.log_priors_vc[Variation.MNV, Call.SOMATIC] = -5

    def somatic_log_prior(variant_type: Variation) -> float:
        # before the softmax normalization the sequencing error log prior is 0
        log_priors_bc = priors.log_priors_bc(make_batch([1e-6], {Data.VARIANT_TYPE: variant_type}))
        return (log_priors_bc[0, Call.SOMATIC] - log_priors_bc[0, Call.SEQ_ERROR]).item()

    assert somatic_log_prior(Variation.SNV) == pytest.approx(-20)
    assert somatic_log_prior(Variation.MNV) == pytest.approx(-5)

    # the M step learns MNV priors from the MNV posterior totals alone
    priors.disable_context_dependent_snv_priors()
    totals_vc = torch.ones(len(Variation), len(Call))
    totals_vc[Variation.MNV, Call.SOMATIC] = 100
    totals_rrra = PosteriorModelPriors.initialize_snv_context_totals_rrra()
    priors.update_priors_m_step(totals_vc, totals_rrra, totals_rrra, ignored_to_non_ignored_ratio=10)
    assert priors.log_priors_vc[Variation.MNV, Call.SOMATIC] > priors.log_priors_vc[Variation.SNV, Call.SOMATIC]


def test_context_dependent_snv_rates_recover_simulated_rates():
    # somatic SNVs simulated from the hierarchical model, about 1000 for each substitution and context
    rng = np.random.default_rng(seed=0)
    theta_s, theta_sc = rng.dirichlet(np.full(12, 20.0)), rng.dirichlet(np.full(16, 20.0), size=12)
    rates_sc = 1e-6 * 12 * theta_s[:, None] * 16 * theta_sc
    sites_sc = np.full((12, 16), 1e9)
    counts_sc = torch.from_numpy(rng.poisson(rates_sc * sites_sc)).float()
    log_rates_sc = context_dependent_snv_log_rates_sc(counts_sc, torch.from_numpy(sites_sc))
    errors_sc = (log_rates_sc - torch.from_numpy(np.log(rates_sc))).abs()
    assert errors_sc.max() < 0.2 and errors_sc.mean() < 0.05


def test_context_dependent_snv_rates_shrink_sparse_counts_and_keep_the_total():
    # three somatic SNVs, all of one substitution in one context
    counts_sc, sites_sc = torch.zeros(12, 16), torch.full((12, 16), 1e9)
    counts_sc[2, 5] = 3
    rates_sc = torch.exp(context_dependent_snv_log_rates_sc(counts_sc, sites_sc))
    # that context's rate is the highest, the substitution's other contexts share in it, and nothing is impossible
    assert rates_sc[2, 5] == rates_sc.max() and rates_sc[2, 4] > rates_sc[3, 4] and torch.all(rates_sc > 0)
    # the expected number of somatic SNVs at the sites is the count plus the overall rate prior's pseudocount
    assert torch.sum(sites_sc * rates_sc).item() == pytest.approx(3 + 1, rel=1e-4)


def test_context_dependent_snv_rates_without_somatic_snvs_are_uniform():
    sites_sc = torch.full((12, 16), 1e8)
    log_rates_sc = context_dependent_snv_log_rates_sc(torch.zeros(12, 16), sites_sc)
    expected = -math.log(sites_sc.sum().item() + 1e6 + 1)
    torch.testing.assert_close(log_rates_sc, torch.full((12, 16), expected, dtype=torch.float64))


def test_the_m_step_puts_the_context_dependent_rates_in_the_snv_priors():
    priors = PosteriorModelPriors(variant_log_prior=-10, artifact_log_prior=-10, no_germline_mode=False, device="cpu")
    rng = np.random.default_rng(seed=1)
    somatic_rrra = PosteriorModelPriors.initialize_snv_context_totals_rrra()
    somatic_rrra[:4, :4, :4, :4] = torch.from_numpy(rng.uniform(0, 5, size=(4, 4, 4, 4)))
    snvs_rrra = PosteriorModelPriors.initialize_snv_context_totals_rrra()
    snvs_rrra[:4, :4, :4, :4] = somatic_rrra[:4, :4, :4, :4] + 100
    totals_vc = torch.full((len(Variation), len(Call)), 100.0)
    priors.update_priors_m_step(totals_vc, somatic_rrra, snvs_rrra, ignored_to_non_ignored_ratio=1000)

    sites_sc = convert_rrra_tensor_to_sc(snvs_rrra) + 1000 * totals_vc.sum().item() / 64
    expected_sc = context_dependent_snv_log_rates_sc(convert_rrra_tensor_to_sc(somatic_rrra), sites_sc)
    torch.testing.assert_close(convert_rrra_tensor_to_sc(priors.somatic_snv_log_priors_rrra), expected_sc.float())
