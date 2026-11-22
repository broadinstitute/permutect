import itertools

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import Tensor
from torch import nn
from torch.nn import Parameter
from torch.nn.parameter import Buffer

from permutect import constants
from permutect.data.batch import Batch
from permutect.data.datum import Data
from permutect.metrics import plotting
from permutect.misc_utils import resolve_device
from permutect.slippage.slippage_model import SlippageModel
from permutect.utils.array_utils import add_at_index
from permutect.utils.array_utils import index_tensor
from permutect.utils.enums import Call
from permutect.utils.enums import Variation

# the shape and rate of the Gamma hyperpriors of the concentrations of the Dirichlet priors of the relative somatic SNV
# rates of the substitutions and of the contexts of each substitution
CONCENTRATION_HYPERPRIOR_SHAPE = 4.0
CONCENTRATION_HYPERPRIOR_RATE = 5.0
# the beta of the Beta(1, beta) prior of the overall somatic SNV rate per site
OVERALL_RATE_PRIOR_BETA = 1e6


def get_ref_contexts_and_alt_bases(batch: Batch) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    # each row is the ref sequence, followed by the alt sequence (hstacked) with A = 0, C = 1 . . . deletion = 4
    haplotypes_bs = batch.get_haplotypes_bs()
    seq_length = haplotypes_bs.shape[-1] // 2
    ref_center_idx = (seq_length - 1) // 2
    alt_center_idx = ref_center_idx + seq_length
    idx0 = haplotypes_bs[:, ref_center_idx - 1]
    idx1 = haplotypes_bs[:, ref_center_idx].int()
    idx2 = haplotypes_bs[:, ref_center_idx + 1]
    idx3 = haplotypes_bs[:, alt_center_idx]
    return idx0, idx1, idx2, idx3


def possible_slippage_b(slippage_model: SlippageModel, batch: Batch) -> Tensor:
    """whether each variant can be slippage, given the STR annotations that filter_variants sets"""
    repeat_lengths_b, repeat_count_changes_b = batch.get(Data.REPEAT_LENGTH), batch.get(Data.REPEAT_COUNT_CHANGE)
    return slippage_model.possible_b(batch.get(Data.VARIANT_TYPE), repeat_lengths_b, repeat_count_changes_b)


def convert_rrra_tensor_to_sc(input_rrra: torch.Tensor) -> torch.Tensor:
    """
    Input is a tensor indexed by the 3-base ref sequence rrr = (left flank, ref base of substitution, right flank)
    and alt base a, including the non-SNV index 4 for deletion and the unused elements where ref and alt bases
    are the same.  Output is indexed by 's' the flattened substitution taking on 4x3=12 values and by 'c', the
    flattened 4x4=16 values of the two flanking bases.
    """
    # step 1: left flank, right flank, ref base, alt base and discard non-SNV index 4
    tensor_ffra = torch.transpose(input_rrra, 1, 2)[0:4, 0:4, 0:4, 0:4]

    # step 2: flatten the flanking bases 'ff' into a single dimension 'c'
    tensor_cra = torch.flatten(tensor_ffra, start_dim=0, end_dim=1)

    # step 3: flatten the ref base 'r' and alt base 'a' into a single dimension 's' and omit r=a
    nontrivial_s_indices = torch.IntTensor([1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14])
    tensor_cs = torch.flatten(tensor_cra, start_dim=1, end_dim=2)[:, nontrivial_s_indices]

    # step 4: switch s and c
    return torch.transpose(tensor_cs, 0, 1)


def context_dependent_snv_log_rates_sc(somatic_sc: Tensor, sites_sc: Tensor) -> Tensor:
    """
    The log somatic SNV rate per site of each substitution 's' and pair of flanking bases 'c', indexed as by
    convert_rrra_tensor_to_sc, given each one's expected somatic SNVs, from the E step, and its sites, by empirical Bayes
    in a hierarchical model.  The rate of (s, c) is r * 12 theta_s * 16 theta_sc.  The overall rate r has a
    Beta(1, OVERALL_RATE_PRIOR_BETA) prior, the substitutions' relative rates theta_s a symmetric Dirichlet prior of
    concentration kappa_s, the relative rates theta_sc of each substitution's contexts symmetric Dirichlet priors of
    concentration kappa_c, and the concentrations Gamma hyperpriors.

    The rates are tiny and every (s, c) has about the same number of sites, most of them ignored sites, so given their
    total the somatic SNVs are multinomial with probabilities theta_s theta_sc, whose Dirichlet priors are conjugate.
    The concentrations are the mode of their posterior, the Dirichlet-multinomial likelihood of the counts times the
    hyperpriors, and the thetas are their posterior means given the concentrations.  It's deterministic and takes
    milliseconds, and r is its posterior mean.
    """
    counts_sc = somatic_sc.detach().double().cpu()
    num_substitutions, num_contexts = counts_sc.shape
    counts_s = counts_sc.sum(dim=1)
    total = counts_s.sum()
    log_concentrations = torch.zeros(2, dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_concentrations], max_iter=100, line_search_fn="strong_wolfe")

    def negative_log_posterior():
        optimizer.zero_grad()
        kappa_s, kappa_c = torch.exp(log_concentrations)
        lgamma = torch.lgamma
        substitutions = lgamma(num_substitutions * kappa_s) - lgamma(total + num_substitutions * kappa_s)
        substitutions = substitutions + torch.sum(lgamma(counts_s + kappa_s) - lgamma(kappa_s))
        contexts = torch.sum(lgamma(num_contexts * kappa_c) - lgamma(counts_s + num_contexts * kappa_c))
        contexts = contexts + torch.sum(lgamma(counts_sc + kappa_c) - lgamma(kappa_c))
        # the Gamma hyperpriors' log densities, plus the log Jacobian of optimizing the logs of the concentrations
        hyperpriors = torch.sum(
            CONCENTRATION_HYPERPRIOR_SHAPE * log_concentrations
            - CONCENTRATION_HYPERPRIOR_RATE * torch.exp(log_concentrations)
        )
        loss = -(substitutions + contexts + hyperpriors)
        loss.backward()
        return loss

    optimizer.step(negative_log_posterior)
    kappa_s, kappa_c = torch.exp(log_concentrations.detach())
    theta_s = (counts_s + kappa_s) / (total + num_substitutions * kappa_s)
    theta_sc = (counts_sc + kappa_c) / (counts_s[:, None] + num_contexts * kappa_c)
    overall_rate = (total + 1) / (sites_sc.detach().double().cpu().sum() + OVERALL_RATE_PRIOR_BETA + 1)
    return torch.log(overall_rate * num_substitutions * theta_s[:, None] * num_contexts * theta_sc)


class PosteriorModelPriors(nn.Module):
    """
    Important technical point: the priors here really only apply to somatic variants, artifacts, and
    normal artifacts.
    The priors for germline variants and contamination are not learned but depend on the population allele frequency of
    each particular variant.  Therefore the stored log germline and contamination priors are set to zero (except in
    no_germline mode, where the germline prior is set to IMPOSSIBLE_LOG_PROB to effectively turn off germline calls) and
    are replaced in log_priors_bc by the per-variant log probability that the sample or the contaminant, respectively,
    is not hom ref.
    The sequencing error log prior is also set to zero because sequencing error essentially *always* happens!  More
    precisely, the sequencing error *generative process* of the sequencer having some stochastic process of misread
    bases is always occurring.  Usually this process does not cause error.  The question is not whether sequencing error
    exists, but whether sequencing error explains the amount of alt allele reads observed.  Thus the mathematically
    correct thing is to set the prior to one (log prior to zero) and allow the likelihoods (basically the TLOD from
    Mutect2) to distinguish.
    The read orientation artifact prior isn't learned either.  It comes from the read orientation model, which
    learn_read_orientation_model learns, so its stored value is also zero and is replaced in log_priors_bc.
    The slippage prior is zero for the same reason as the sequencing error prior: slippage can always happen in a long
    enough STR.  It is IMPOSSIBLE_LOG_PROB for a variant that can't be slippage.
    """

    def __init__(
        self,
        variant_log_prior: float,
        artifact_log_prior: float,
        no_germline_mode: bool,
        device=None,
        slippage_model: SlippageModel = None,
    ):
        """
        :param slippage_model: which variants can be slippage, by default as in FilterMutectCalls
        """
        super(PosteriorModelPriors, self).__init__()
        self.no_germline_mode = no_germline_mode
        self.slippage_model = SlippageModel() if slippage_model is None else slippage_model
        self._device = resolve_device(device)
        self.use_context_dependent_snv_priors = True

        # pre-softmax priors of different call types [log P(variant), log P(artifact), log P(seq error)] for each variant type
        # although these vectors are defined for all variant types, somatic SNVs are handled separately
        self.log_priors_vc = torch.nn.Parameter(torch.zeros(len(Variation), len(Call)))
        with torch.no_grad():
            self.log_priors_vc[:, Call.SOMATIC] = variant_log_prior
            self.log_priors_vc[:, Call.ARTIFACT] = artifact_log_prior
            self.log_priors_vc[:, Call.GERMLINE] = constants.IMPOSSIBLE_LOG_PROB if self.no_germline_mode else 0
            self.log_priors_vc[:, Call.NORMAL_ARTIFACT] = artifact_log_prior

        # context and substitution dependent (the 3 r's stand for ref context, the 'a' is for alt base)
        # we have 5 values just in case there is a D in the ref context (an indel next to a SNV, perhaps)
        self.somatic_snv_log_priors_rrra = Parameter(variant_log_prior * torch.ones((5, 5, 5, 5), device=self._device))

        # a buffer so that it moves with the module; not persistent because it is a constant
        self.NONTRIVIAL_CONTEXTS_rrra = Buffer(torch.ones((5, 5, 5, 5), device=self._device), persistent=False)
        self.NONTRIVIAL_CONTEXTS_rrra[4, :, :, :] = 0
        self.NONTRIVIAL_CONTEXTS_rrra[:, 4, :, :] = 0
        self.NONTRIVIAL_CONTEXTS_rrra[:, :, 4, :] = 0
        self.NONTRIVIAL_CONTEXTS_rrra[:, :, :, 4] = 0
        for base in range(4):
            self.NONTRIVIAL_CONTEXTS_rrra[:, base, :, base] = 0
        self.NUM_NONTRIVIAL_CONTEXTS = 4 * 4 * 4 * 3

    def enable_context_dependent_snv_priors(self) -> None:
        self.use_context_dependent_snv_priors = True

    def disable_context_dependent_snv_priors(self) -> None:
        self.use_context_dependent_snv_priors = False

    @classmethod
    def initialize_snv_context_totals_rrra(cls, device=None):
        return torch.zeros(5, 5, 5, 5, device=resolve_device(device))

    @classmethod
    def increment_somatic_snv_context_totals_rrra(
        cls, snv_totals_rrra, context_totals_rrra, batch: Batch, posteriors_bc
    ):
        is_snv = (batch.get(Data.VARIANT_TYPE) == Variation.SNV).float()
        somatic_snv_posteriors_b = posteriors_bc[:, Call.SOMATIC] * is_snv
        idx0, idx1, idx2, idx3 = get_ref_contexts_and_alt_bases(batch)
        add_at_index(tens=snv_totals_rrra, idx=(idx0, idx1, idx2, idx3), values=somatic_snv_posteriors_b)
        add_at_index(tens=context_totals_rrra, idx=(idx0, idx1, idx2, idx3), values=is_snv)

    def somatic_snv_log_priors(self, batch: Batch) -> torch.Tensor:
        idx0, idx1, idx2, idx3 = get_ref_contexts_and_alt_bases(batch)
        return index_tensor(self.somatic_snv_log_priors_rrra, (idx0, idx1, idx2, idx3))

    def log_priors_bc(self, batch: Batch) -> torch.Tensor:
        variant_types_b = batch.get(Data.VARIANT_TYPE).long()
        log_allele_frequencies_b = batch.get(Data.LOG_ALLELE_FREQUENCY)
        is_snv_b = variant_types_b == Variation.SNV

        # seq error and germline initialized to 0 or IMPOSSIBLE_LOG_PROB as discussed above
        log_priors_bc = self.log_priors_vc[variant_types_b.long(), :]
        log_priors_bc[:, Call.SEQ_ERROR] = 0
        # 1 minus hom ref probability: 1 - (1 - AF)^2 = AF (2 - AF)
        log_carrier_priors_b = log_allele_frequencies_b + torch.log(2 - torch.exp(log_allele_frequencies_b))
        log_priors_bc[:, Call.GERMLINE] = (
            constants.IMPOSSIBLE_LOG_PROB if self.no_germline_mode else log_carrier_priors_b
        )
        log_priors_bc[:, Call.CONTAMINATION] = log_carrier_priors_b
        log_priors_bc[:, Call.SLIPPAGE] = torch.where(
            possible_slippage_b(self.slippage_model, batch), 0, constants.IMPOSSIBLE_LOG_PROB
        )

        if self.use_context_dependent_snv_priors:
            # torch.where rather than mask arithmetic, which gives 0 * -inf = NaN when a log prior is -inf, eg the SNV
            # priors of a dataset without SNVs
            log_priors_bc[:, Call.SOMATIC] = torch.where(
                is_snv_b, self.somatic_snv_log_priors(batch), log_priors_bc[:, Call.SOMATIC]
            )
        return torch.nn.functional.log_softmax(log_priors_bc, dim=-1)

    def update_priors_m_step(
        self,
        posterior_totals_vc,
        somatic_snv_totals_rrra,
        snv_context_totals_rrra,
        ignored_to_non_ignored_ratio,
    ):
        # update the priors in an EM-style M step.  We'll need the counts of each call type vs variant type
        # We need to correct for all the sites that didn't enter the training data, sites with neither somatic variants
        # nor artifacts.  For example, suppose our entire test dataset in a million-base genome consists of a single
        # somatic variant.  We want to somatic prior to be one in a million, not one in one!
        total_nonignored = torch.sum(posterior_totals_vc).item()
        total_ignored = ignored_to_non_ignored_ratio * total_nonignored
        overall_total = total_ignored + total_nonignored
        # coarse assumption that every context is equally likely
        total_ignored_per_context = total_ignored / 64

        with torch.no_grad():
            self.log_priors_vc.copy_(torch.log(posterior_totals_vc / (posterior_totals_vc + overall_total)))
            self.log_priors_vc[:, Call.SEQ_ERROR] = 0
            self.log_priors_vc[:, Call.GERMLINE] = constants.IMPOSSIBLE_LOG_PROB if self.no_germline_mode else 0
            self.log_priors_vc[:, Call.CONTAMINATION] = 0
            self.log_priors_vc[:, Call.SLIPPAGE] = 0

        if self.use_context_dependent_snv_priors:
            # indices 's' for the 4x3=12 substitutions, A->C = 0, A->G = 1. . .T->G = 11, and 'c' for the 4x4=16 pairs
            # of flanking bases
            sites_sc = convert_rrra_tensor_to_sc(snv_context_totals_rrra) + total_ignored_per_context
            somatic_sc = convert_rrra_tensor_to_sc(somatic_snv_totals_rrra)
            # on the device of the priors, so that the loop below copies nothing from the host
            log_rates_sc = context_dependent_snv_log_rates_sc(somatic_sc, sites_sc).to(self.somatic_snv_log_priors_rrra)

            with torch.no_grad():
                # lf, rf are left and right flanking bases, ref and alt are ref and alt bases of the substitution.
                for lf, rf in itertools.product(range(4), range(4)):
                    for ref, alt in itertools.product(range(4), range(4)):
                        if ref != alt:
                            context = lf * 4 + rf
                            substitution_with_trivial = ref * 4 + alt
                            substitution = substitution_with_trivial - (substitution_with_trivial // 5) - 1
                            self.somatic_snv_log_priors_rrra[lf, ref, rf, alt] = log_rates_sc[substitution, context]
        else:  # if not using context-dependent SNV priors
            with torch.no_grad():
                self.somatic_snv_log_priors_rrra.fill_(self.log_priors_vc[Variation.SNV, Call.SOMATIC])

    def make_priors_bar_plot(self, snv_context_totals_rrra):
        # bar plot of log priors -- data is indexed by call type name, and x ticks are variant types

        log_prior_bar_plot_data = {
            call.name: self.log_priors_vc[:, call].cpu().detach().numpy()
            for call in [Call.SOMATIC, Call.ARTIFACT, Call.NORMAL_ARTIFACT]
        }

        somatic_snv_rates_rrra = self.NONTRIVIAL_CONTEXTS_rrra * torch.exp(self.somatic_snv_log_priors_rrra).detach()
        average_somatic_snv_rate = torch.sum(somatic_snv_rates_rrra) / self.NUM_NONTRIVIAL_CONTEXTS
        log_prior_bar_plot_data[Call.SOMATIC.name][Variation.SNV] = torch.log(average_somatic_snv_rate).item()

        prior_fig, prior_ax = plotting.grouped_bar_plot(
            log_prior_bar_plot_data, [v_type.name for v_type in Variation], "log priors"
        )
        return prior_fig, prior_ax

    def make_context_priors_plot(self):
        # the main structure is 4 rows A, C, G, T and 4 columns A, C, G, T defining the ref and alt bases of the SNV
        prior_fig, prior_ax = plt.subplots(4, 4, sharex="all", sharey="all", squeeze=False)
        row_names = ["A", "C", "G", "T"]
        col_names = ["A", "C", "G", "T"]
        bounds = np.array([0, 1, 2, 3, 4])

        common_colormesh = None
        for row, label in enumerate(row_names):
            for col, var_type in enumerate(col_names):
                if row != col:
                    # for each substitution, we have a 2D sub-array for left flank (l) and right flank (r)
                    values_lr = self.somatic_snv_log_priors_rrra[:, row, :, col].detach().cpu()
                    common_colormesh = plotting.color_plot_2d_on_axis(
                        prior_ax[row, col], bounds, bounds, values_lr, None, None
                    )

        prior_fig.colorbar(common_colormesh)
        plotting.tidy_subplots(
            prior_fig,
            prior_ax,
            x_label="alt base",
            y_label="ref base",
            row_labels=row_names,
            col_labels=col_names,
        )
        return prior_fig, prior_ax
