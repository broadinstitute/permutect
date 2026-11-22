import math

import torch
from matplotlib import pyplot as plt
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from tqdm.autonotebook import tqdm
from tqdm.autonotebook import trange

from permutect import constants
from permutect.architecture.posterior_model_priors import PosteriorModelPriors
from permutect.architecture.spectra.posterior_model_spectra import PosteriorModelSpectra
from permutect.data.batch import Batch
from permutect.data.batch import BatchIndices
from permutect.data.batches_on_device import batches_on_device
from permutect.data.count_binning import NUM_ALT_COUNT_BINS
from permutect.data.count_binning import count_from_alt_bin_index
from permutect.data.datum import Data
from permutect.metrics import plotting
from permutect.metrics.loss_metrics import AccuracyMetrics
from permutect.misc_utils import backpropagate
from permutect.misc_utils import float_dtype
from permutect.misc_utils import resolve_device
from permutect.read_orientation.read_orientation_model import ReadOrientationModel
from permutect.segmentation.germline_model import GermlineModel
from permutect.slippage.slippage_model import SlippageModel
from permutect.utils.enums import Call
from permutect.utils.enums import Label
from permutect.utils.enums import Variation
from permutect.utils.math_utils import LOGIT_EPS
from permutect.utils.math_utils import f_score
from permutect.utils.math_utils import prob_to_logit

# Fitting the spectra and priors: Adam steps at a high learning rate warm-start a full-batch EM, which stops when an
# iteration changes the mean log evidence per variant by less than the convergence threshold.  On DREAM chr20, 300
# warm-start steps and 2 EM iterations came within 0.0006 nats per variant of the converged evidence.
DEFAULT_SPECTRUM_LEARNING_RATE = 0.1
DEFAULT_NUM_SPECTRUM_WARMUP_STEPS = 300
DEFAULT_SPECTRUM_CONVERGENCE_THRESHOLD = 1e-4
# the L-BFGS iterations on the spectra in each EM iteration, a generalized EM's partial maximization
LBFGS_ITERATIONS_PER_EM_ITERATION = 5


def best_error_probability_threshold(error_probs, recall_weight: float = 1.0) -> float:
    """
    The error probability threshold that maximizes the expected F score if the error probabilities are calibrated.
    Variants pass if their error probability is at most the threshold, and recall_weight is the beta of the F_beta
    score, the importance of recall (sensitivity) relative to precision.
    """
    sorted_probs = sorted(error_probs)  # from least to greatest error probability
    total_error = sum(sorted_probs) + 0.0001
    total_non_error = len(sorted_probs) - total_error + 0.0002

    # start at threshold = 0; that is, everything fails, and pass one variant at a time by increasing the threshold
    error_found, non_error_found = total_error, 0
    best_threshold, best_f_score = 0, 0
    for prob in sorted_probs:
        error_found -= prob  # lose a fractional error
        non_error_found += 1 - prob  # gain a fractional non-error

        tp = non_error_found  # non-errors that pass the threshold are true positives
        fp = total_error - error_found  # errors that pass the threshold are false positives
        sensitivity = tp / total_non_error
        precision = tp / (tp + fp)

        score = f_score(precision, sensitivity, beta=recall_weight)
        if score > best_f_score:
            best_f_score, best_threshold = score, prob
    return best_threshold


class EStepTotals:
    """
    The E step's totals over batches, for the priors' M step, and the total log evidence, kept on the device so that
    recording them doesn't make the CPU wait for the GPU
    """

    def __init__(self, device):
        # posterior totals by variant type and call, and the somatic SNV posteriors and SNVs by context and alt base
        self.posterior_totals_vc = torch.zeros((len(Variation), len(Call)), device=device)
        self.somatic_snv_totals_rrra = PosteriorModelPriors.initialize_snv_context_totals_rrra(device=device)
        self.snv_context_totals_rrra = PosteriorModelPriors.initialize_snv_context_totals_rrra(device=device)
        self.log_evidence = torch.zeros((), device=device)
        self.count = 0

    def record(self, batch: Batch, log_posteriors_bc: Tensor):
        posteriors_bc = torch.softmax(log_posteriors_bc, dim=-1).detach()
        PosteriorModelPriors.increment_somatic_snv_context_totals_rrra(
            self.somatic_snv_totals_rrra, self.snv_context_totals_rrra, batch, posteriors_bc
        )
        self.posterior_totals_vc.index_add_(dim=0, index=batch.get(Data.VARIANT_TYPE), source=posteriors_bc)
        self.log_evidence += torch.logsumexp(log_posteriors_bc.detach(), dim=1).sum()
        self.count += batch.size()

    def update_priors(self, priors: PosteriorModelPriors, ignored_to_non_ignored_ratio: float):
        priors.update_priors_m_step(
            self.posterior_totals_vc,
            self.somatic_snv_totals_rrra,
            self.snv_context_totals_rrra,
            ignored_to_non_ignored_ratio,
        )

    def mean_log_evidence(self) -> float:
        return (self.log_evidence / max(self.count, 1)).item()


class PosteriorModel(torch.nn.Module):
    """ """

    def __init__(
        self,
        variant_log_prior: float,
        artifact_log_prior: float,
        no_germline_mode: bool = False,
        device=None,
        germline_model: GermlineModel = None,
        read_orientation_model: ReadOrientationModel = None,
        slippage_model: SlippageModel = None,
    ):
        """
        :param germline_model: from segment_pileups, for germline likelihoods.  Without one, the MAF is 1/2
        everywhere and the germline likelihoods have default parameters.
        :param read_orientation_model: from learn_read_orientation_model, for the likelihood of an SNV's alt reads from
        damage, which is part of SEQ_ERROR's likelihood.  Without one, SEQ_ERROR's likelihood is Mutect2's alone.
        :param slippage_model: which variants can be slippage and their likelihoods, by default as in FilterMutectCalls
        """
        super(PosteriorModel, self).__init__()

        self._device = resolve_device(device)
        self._dtype = float_dtype(self._device)
        self.no_germline_mode = no_germline_mode

        slippage_model = SlippageModel() if slippage_model is None else slippage_model
        self.spectra = PosteriorModelSpectra(germline_model, self._device, read_orientation_model, slippage_model)
        self.priors = PosteriorModelPriors(
            variant_log_prior,
            artifact_log_prior,
            no_germline_mode,
            self._device,
            slippage_model,
        )

        self.to(device=self._device, dtype=self._dtype)

    def posterior_probabilities_bc(self, batch: Batch) -> Tensor:
        """
        :param batch:
        :return: non-log probabilities as a 2D tensor, indexed by batch 'b', Call type 'c'
        """
        return torch.nn.functional.softmax(self.log_relative_posteriors_bc(batch), dim=1)

    def error_probabilities_b(self, batch: Batch, germline_mode: bool = False) -> Tensor:
        """
        :param germline_mode: if True, germline classification is not considered an error mode
        :param batch:
        :return: non-log error probabilities as a 1D tensor with length batch size
        """
        return self.error_probabilities_from_posteriors_b(self.posterior_probabilities_bc(batch), germline_mode)

    def error_probabilities_from_posteriors_b(self, posterior_probs_bc: Tensor, germline_mode: bool = False) -> Tensor:
        """
        :param posterior_probs_bc: non-log posterior probabilities, as from posterior_probabilities_bc
        :param germline_mode: if True, germline classification is not considered an error mode
        :return: non-log error probabilities as a 1D tensor with length batch size
        """
        assert not (germline_mode and self.no_germline_mode), "germline mode and no-germline mode are incompatible"
        return 1 - posterior_probs_bc[:, Call.GERMLINE if germline_mode else Call.SOMATIC]  # 0th column is variant

    def log_posterior_and_ingredients(self, batch: Batch) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """
        :param batch:
        :batch.seq_error_log_likelihoods() is the probability that these *particular* reads exhibit the alt allele given a
        sequencing error ie an error explainable in terms of base qualities.  For example if we have two alt reads with error
        probability of 0.1 and 0.2, and two ref reads with error probabilities 0.05 and 0.06 this quantity would be
        log(0.1*0.2*0.95*0.94).  This is an annotation emitted by the GATK and by the time it reaches here is a 1D tensor
        of length batch_size.
        :return:
        """
        # All log likelihood/relative posterior tensors below have shape batch.size() x len(CallType)
        # spectra tensors contain the likelihood that these *particular* reads (that is, not just the read count) are alt
        # normal log likelihoods contain everything going on in the matched normal sample
        log_priors_bc = self.priors.log_priors_bc(batch)
        spectra_log_lks_bc, normal_log_lks_bc = self.spectra.spectra_log_likelihoods_bc(batch)

        log_posteriors_bc = log_priors_bc + spectra_log_lks_bc + normal_log_lks_bc

        # an infinite artifact logit marks a known artifact, eg a variant with a trusted upstream artifact filter, which
        # can only be an artifact or a normal artifact.  Since both get the same logit, any finite value works for them.
        artifact_logits_b = batch.get(Data.CACHED_ARTIFACT_LOGIT)
        known_artifacts_b = torch.isposinf(artifact_logits_b)
        artifact_logits_b = torch.where(known_artifacts_b, 0, artifact_logits_b)
        log_posteriors_bc[:, Call.ARTIFACT] += artifact_logits_b
        log_posteriors_bc[:, Call.NORMAL_ARTIFACT] += artifact_logits_b
        # slippage, like any sequencing error, gets no artifact logit: its reads look like a non-artifact's

        # a known artifact can only be an artifact or a normal artifact
        for call in set(Call) - {Call.ARTIFACT, Call.NORMAL_ARTIFACT}:
            log_posteriors_bc[:, call] = torch.where(
                known_artifacts_b, constants.IMPOSSIBLE_LOG_PROB, log_posteriors_bc[:, call]
            )

        # TODO: HACK / EXPERIMENT: make it impossible to call an artifact when the artifact logits are negative
        log_posteriors_bc[:, Call.ARTIFACT] = torch.where(
            batch.get(Data.CACHED_ARTIFACT_LOGIT) < 0,
            constants.IMPOSSIBLE_LOG_PROB,
            log_posteriors_bc[:, Call.ARTIFACT],
        )
        # TODO: END OF HACK / EXPERIMENT

        return log_priors_bc, spectra_log_lks_bc, normal_log_lks_bc, log_posteriors_bc

    def log_relative_posteriors_bc(self, batch: Batch) -> Tensor:
        return self.log_posterior_and_ingredients(batch)[-1]

    def learn_priors_and_spectra(
        self,
        posterior_loader,
        num_iterations,
        ignored_to_non_ignored_ratio: float,
        summary_writer: SummaryWriter = None,
        learning_rate: float = DEFAULT_SPECTRUM_LEARNING_RATE,
        num_warmup_steps: int = DEFAULT_NUM_SPECTRUM_WARMUP_STEPS,
        convergence_threshold: float = DEFAULT_SPECTRUM_CONVERGENCE_THRESHOLD,
        full_pass_loader=None,
    ) -> tuple[int, bool]:
        """
        Fit the spectra, whose parameters maximize the evidence, and the priors, whose M step uses the E step's posterior
        totals, with the context-dependent SNV priors throughout, in two phases:

        1. A warm start: Adam steps on the minibatches of posterior_loader, at least num_warmup_steps of them, in whole
           epochs, each followed by the priors' M step.  These quickly get near the optimum, but don't settle there.
        2. A full-batch EM.  Each iteration takes a few L-BFGS iterations on the spectra given the priors, each needing a
           pass over all the data, then a pass for the E step and the priors' M step.  It stops when an iteration changes
           the mean log evidence per variant by less than convergence_threshold, or after num_iterations iterations.  The
           change is absolute, since the evidence leaves out the ignored sites and needn't increase.

        :param posterior_loader: an iterable of batches with a length, such as a CompactPosteriorData
        :param full_pass_loader: the batches of a pass over all the data for the EM, in a fixed order and as big as fit
            in memory, by default posterior_loader
        :param ignored_to_non_ignored_ratio: ratio of sites in which no evidence of variation was found to sites in which
        sufficient evidence was found to emit test data.  Without this parameter (i.e. if it were set to zero) we would
        underestimate the frequency of sequencing error, hence overestimate the prior probability of variation.
        :return: the number of EM iterations and whether the EM converged
        """
        # the germline, read orientation and slippage models' parameters are frozen
        parameters = [parameter for parameter in self.spectra.parameters() if parameter.requires_grad]
        full_pass_loader = posterior_loader if full_pass_loader is None else full_pass_loader
        self.priors.enable_context_dependent_snv_priors()

        optimizer = torch.optim.Adam(parameters, lr=learning_rate)
        num_epochs = max(1, math.ceil(num_warmup_steps / max(len(posterior_loader), 1)))
        for epoch in trange(1, num_epochs + 1, desc="AF spectra warm-start epoch"):
            totals = EStepTotals(self._device)
            batch: Batch
            batches = batches_on_device(posterior_loader, device=self._device)
            for batch in tqdm(batches, mininterval=10, total=len(posterior_loader)):
                log_posteriors_bc = self.log_relative_posteriors_bc(batch)
                totals.record(batch, log_posteriors_bc)
                backpropagate(optimizer, -torch.mean(torch.logsumexp(log_posteriors_bc, dim=1)))
            totals.update_priors(self.priors, ignored_to_non_ignored_ratio)
            if summary_writer is not None:
                summary_writer.add_scalar("spectra warm-start mean log evidence", totals.mean_log_evidence(), epoch)
        num_data = totals.count

        lbfgs = torch.optim.LBFGS(
            parameters, lr=1, max_iter=LBFGS_ITERATIONS_PER_EM_ITERATION, history_size=20, line_search_fn="strong_wolfe"
        )

        def negative_mean_log_evidence() -> Tensor:
            # the gradient accumulates over the batches, of which only one at a time is in memory
            lbfgs.zero_grad()
            total = torch.zeros((), device=self._device)
            for full_pass_batch in batches_on_device(full_pass_loader, device=self._device):
                log_posteriors_bc = self.log_relative_posteriors_bc(full_pass_batch)
                loss = -torch.sum(torch.logsumexp(log_posteriors_bc, dim=1)) / num_data
                loss.backward()
                total += loss.detach()
            return total

        iteration, converged, previous_mean_log_evidence = 0, False, None
        for iteration in trange(1, num_iterations + 1, desc="AF spectra EM iteration"):
            lbfgs.step(negative_mean_log_evidence)
            totals = EStepTotals(self._device)
            with torch.no_grad():
                for full_pass_batch in batches_on_device(full_pass_loader, device=self._device):
                    totals.record(full_pass_batch, self.log_relative_posteriors_bc(full_pass_batch))
            totals.update_priors(self.priors, ignored_to_non_ignored_ratio)
            mean_log_evidence = totals.mean_log_evidence()
            if summary_writer is not None:
                summary_writer.add_scalar("spectra EM mean log evidence", mean_log_evidence, iteration)
            change = (
                math.inf if previous_mean_log_evidence is None else abs(mean_log_evidence - previous_mean_log_evidence)
            )
            converged = change < convergence_threshold
            if converged:
                break
            previous_mean_log_evidence = mean_log_evidence

        outcome = "converged" if converged else f"stopped without converging at the maximum of {num_iterations}"
        print(f"AF spectra: {num_epochs} warm-start epochs, then {iteration} EM iterations, {outcome}")
        if summary_writer is not None:
            self.plot_spectra_and_priors(summary_writer, totals.snv_context_totals_rrra, num_epochs + iteration)
        return iteration, converged

    def plot_spectra_and_priors(self, summary_writer: SummaryWriter, snv_context_totals_rrra: Tensor, step: int):
        # once, after fitting, since the figures take about 2 seconds, as long as an epoch of a small dataset
        for depth in [9, 19, 30, 50, 100]:
            art_spectra_fig, art_spectra_axs = self.spectra.artifact_spectra.plot_artifact_spectra(depth)
            summary_writer.add_figure("Artifact AF Spectra at depth = " + str(depth), art_spectra_fig, step)

        # normal_artifact_spectra_fig, normal_artifact_spectra_axs = plot_artifact_spectra(self.normal_artifact_spectra)
        # summary_writer.add_figure("Normal Artifact AF Spectra", normal_artifact_spectra_fig, epoch)

        var_spectra_fig, _ = self.spectra.somatic_spectrum.plot_spectrum("Variant AF Spectrum")
        summary_writer.add_figure("Variant AF Spectra", var_spectra_fig, step)

        prior_fig, prior_ax = self.priors.make_priors_bar_plot(snv_context_totals_rrra)
        context_prior_fig, context_prior_ax = self.priors.make_context_priors_plot()
        summary_writer.add_figure("log priors", prior_fig, step)
        summary_writer.add_figure("log SNV priors", context_prior_fig, step)

        # normal artifact joint tumor-normal spectra
        # na_fig, na_axes = plt.subplots(1, len(Variation), sharex='all', sharey='all', squeeze=False)
        # for variant_index, variant_type in enumerate(Variation):
        #    self.normal_artifact_spectra[variant_index].density_plot_on_axis(na_axes[0, variant_index])
        # plotting.tidy_subplots(na_fig, na_axes, x_label="tumor fraction", y_label="normal fraction",
        #                       row_labels=[""], column_labels=[var_type.name for var_type in Variation])
        # summary_writer.add_figure("normal artifact spectra", na_fig, epoch)

    # map of Variant type to probability threshold that maximizes F1 score
    # loader is a Dataloader whose collate_fn is the PosteriorBatch constructor
    # in inference mode: each batch's error probabilities are kept until the end, and with gradients they would keep
    # their autograd graphs, about 7 GB per million variants
    @torch.inference_mode()
    def calculate_probability_thresholds(
        self,
        posterior_loader,
        summary_writer: SummaryWriter = None,
        germline_mode: bool = False,
        recall_weight: float = 1.0,
    ):
        self.train(False)
        # of each batch, on the device until the end; errors are artifacts and seq errors
        error_probs, var_types = [], []

        # expected counts for the theoretical ROC curves, ie the ROC curves if the error probabilities were perfectly
        # calibrated: a variant with error probability p counts as 1 - p non-errors (VARIANT) and p errors (ARTIFACT)
        theoretical_metrics = AccuracyMetrics.create(num_sources=1, device=self._device)

        batch: Batch
        batches = batches_on_device(posterior_loader, device=self._device)
        for batch in tqdm(batches, mininterval=10, total=len(posterior_loader)):
            # 0th column is true variant, subtract it from 1 to get error prob
            error_probs_b = self.error_probabilities_b(batch, germline_mode)
            var_types_b = batch.get(Data.VARIANT_TYPE)
            error_probs.append(error_probs_b)
            # a copy: the column is a view of the batch's data, which would stay in memory until the end
            var_types.append(var_types_b.clone())

            indices = BatchIndices(
                sources=torch.zeros_like(var_types_b),
                labels=torch.full_like(var_types_b, Label.VARIANT),
                var_types=var_types_b,
                ref_counts=torch.zeros_like(var_types_b),
                alt_counts=batch.get(Data.ORIGINAL_ALT_COUNT),
            )
            error_logits_b = torch.logit(error_probs_b, eps=LOGIT_EPS)
            indices.increment_tensor(theoretical_metrics, values=1 - error_probs_b, logits=error_logits_b)
            artifact_labels_b = torch.full_like(var_types_b, Label.ARTIFACT)
            indices.increment_tensor(
                theoretical_metrics, values=error_probs_b, labels=artifact_labels_b, logits=error_logits_b
            )

        # copied to the CPU once, rather than for every batch, which would make the CPU wait for the GPU each time
        error_probs_n = torch.cat(error_probs).cpu() if error_probs else torch.zeros(0)
        var_types_n = torch.cat(var_types).cpu() if var_types else torch.zeros(0, dtype=torch.long)
        theoretical_metrics = theoretical_metrics.cpu()
        thresholds_by_type = {
            var_type: best_error_probability_threshold(error_probs_n[var_types_n == var_type].tolist(), recall_weight)
            for var_type in Variation
        }
        roc_fig, roc_axes = plt.subplots(1, len(Variation), sharex="all", sharey="all", squeeze=False)
        roc_by_cnt_fig, roc_by_cnt_axes = plt.subplots(
            1, len(Variation), sharex="all", sharey="all", squeeze=False, figsize=(10, 6), dpi=100
        )
        count_bin_labels = [str(count_from_alt_bin_index(count_bin)) for count_bin in range(NUM_ALT_COUNT_BINS)]
        for var_type in Variation:
            # every curve marks the chosen threshold, and the default error probability of 0.5 (logit 0)
            threshold_logit = prob_to_logit(thresholds_by_type[var_type])

            # plot all count ROC curves for this variant type
            roc_data_by_count = [
                theoretical_metrics.make_roc_data(None, count_bin, None, var_type, sens_prec=True)
                for count_bin in range(NUM_ALT_COUNT_BINS)
            ]
            thresholds = [threshold_logit] * NUM_ALT_COUNT_BINS
            plotting.plot_roc_on_axis(
                roc_data_by_count, count_bin_labels, roc_by_cnt_axes[0, var_type], True, thresholds
            )

            roc_data = theoretical_metrics.make_roc_data(None, None, None, var_type, sens_prec=True)
            plotting.plot_roc_on_axis([roc_data], [""], roc_axes[0, var_type], True, [threshold_logit])

        variation_types = [var_type.name for var_type in Variation]
        for fig, axes in ((roc_by_cnt_fig, roc_by_cnt_axes), (roc_fig, roc_axes)):
            plotting.tidy_subplots(
                fig, axes, x_label="sensitivity", y_label="precision", row_labels=[""], col_labels=variation_types
            )
        if summary_writer is not None:
            summary_writer.add_figure("theoretical ROC by variant type ", roc_fig)
            summary_writer.add_figure("theoretical ROC by variant type and alt count ", roc_by_cnt_fig)

        return thresholds_by_type
