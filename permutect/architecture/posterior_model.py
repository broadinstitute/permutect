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
from permutect.misc_utils import StreamingAverage
from permutect.misc_utils import backpropagate
from permutect.misc_utils import float_dtype
from permutect.misc_utils import resolve_device
from permutect.utils.enums import Call
from permutect.utils.enums import Label
from permutect.utils.enums import Variation
from permutect.utils.math_utils import LOGIT_EPS
from permutect.utils.math_utils import f_score
from permutect.utils.math_utils import prob_to_logit


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


class PosteriorModel(torch.nn.Module):
    """ """

    def __init__(
        self,
        variant_log_prior: float,
        artifact_log_prior: float,
        no_germline_mode: bool = False,
        device=None,
        het_beta: float = None,
    ):
        super(PosteriorModel, self).__init__()

        self._device = resolve_device(device)
        self._dtype = float_dtype(self._device)
        self.no_germline_mode = no_germline_mode
        self.het_beta = het_beta

        self.spectra = PosteriorModelSpectra(device=self._device, het_beta=het_beta)
        self.priors = PosteriorModelPriors(variant_log_prior, artifact_log_prior, no_germline_mode, self._device)

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
        for non_artifact_call in (Call.SOMATIC, Call.SEQ_ERROR, Call.GERMLINE):
            log_posteriors_bc[:, non_artifact_call] = torch.where(
                known_artifacts_b, constants.IMPOSSIBLE_LOG_PROB, log_posteriors_bc[:, non_artifact_call]
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
        learning_rate: float = 0.001,
    ):
        """
        :param summary_writer:
        :param num_iterations:
        :param posterior_loader:
        :param ignored_to_non_ignored_ratio: ratio of sites in which no evidence of variation was found to sites in which
        sufficient evidence was found to emit test data.  Without this parameter (i.e. if it were set to zero) we would
        underestimate the frequency of sequencing error, hence overestimate the prior probability of variation.
        """
        # spectra parameters are learned via batched stochastic gradient descent during the e step of each epoch.
        # the loss is the negative log evidence.  Prior parameters are adjusted in the M steo.
        optimizer = torch.optim.Adam(self.spectra.parameters(), lr=learning_rate)

        # we begin learning with context-dependent priors turned off
        self.priors.disable_context_dependent_snv_priors()
        for epoch in trange(1, num_iterations + 1, desc="AF spectra epoch"):
            use_context_dependence = epoch > (num_iterations / 2)
            if use_context_dependence:
                self.priors.enable_context_dependent_snv_priors()
            epoch_loss = StreamingAverage()

            # data for M step
            # E-step totals indexed by variant type and call type
            # for somatic SNVs, totals indexed by 3-base context and substitution alt base
            posterior_totals_tc = torch.zeros((len(Variation), len(Call)), device=self._device)
            somatic_snv_totals_rrra = PosteriorModelPriors.initialize_snv_context_totals_rrra(device=self._device)
            snv_context_totals_rrra = PosteriorModelPriors.initialize_snv_context_totals_rrra(device=self._device)

            batch: Batch
            batches = batches_on_device(posterior_loader, device=self._device)
            for batch in tqdm(batches, mininterval=10, total=len(posterior_loader)):
                relative_posteriors = self.log_relative_posteriors_bc(batch)
                log_evidence = torch.logsumexp(relative_posteriors, dim=1)

                # the next line performs "for b in batch: posterior_total_tc[var_types[b],:] += posteriors_bc[b, :]"
                posteriors_bc = torch.softmax(relative_posteriors, dim=-1).detach()

                # collect the somatic SNV E step counts for subsequent M step
                PosteriorModelPriors.increment_somatic_snv_context_totals_rrra(
                    somatic_snv_totals_rrra, snv_context_totals_rrra, batch, posteriors_bc
                )
                posterior_totals_tc.index_add_(dim=0, index=batch.get(Data.VARIANT_TYPE), source=posteriors_bc)

                # confidence_mask = torch.logical_or(batch.get_artifact_logits() < 0, batch.get_artifact_logits() > 3)
                loss = -torch.mean(log_evidence)
                # loss = - torch.sum(confidence_mask * log_evidence) / (torch.sum(confidence_mask) + 0.000001)

                backpropagate(optimizer, loss)

                # the sum stays on the device, so that recording it doesn't make the CPU wait for the GPU
                epoch_loss.record_sum(batch.size() * loss.detach(), batch.size())
            # iteration over posterior dataloader finished

            self.priors.update_priors_m_step(
                posterior_totals_tc,
                somatic_snv_totals_rrra,
                snv_context_totals_rrra,
                ignored_to_non_ignored_ratio,
            )

            if summary_writer is not None:
                summary_writer.add_scalar("spectrum negative log evidence", epoch_loss.get(), epoch)

                for depth in [9, 19, 30, 50, 100]:
                    art_spectra_fig, art_spectra_axs = self.spectra.artifact_spectra.plot_artifact_spectra(depth)
                    summary_writer.add_figure("Artifact AF Spectra at depth = " + str(depth), art_spectra_fig, epoch)

                # normal_artifact_spectra_fig, normal_artifact_spectra_axs = plot_artifact_spectra(self.normal_artifact_spectra)
                # summary_writer.add_figure("Normal Artifact AF Spectra", normal_artifact_spectra_fig, epoch)

                var_spectra_fig, _ = self.spectra.somatic_spectrum.plot_spectrum("Variant AF Spectrum")
                summary_writer.add_figure("Variant AF Spectra", var_spectra_fig, epoch)

                prior_fig, prior_ax = self.priors.make_priors_bar_plot(snv_context_totals_rrra)
                context_prior_fig, context_prior_ax = self.priors.make_context_priors_plot()
                summary_writer.add_figure("log priors", prior_fig, epoch)
                summary_writer.add_figure("log SNV priors", context_prior_fig, epoch)

                # normal artifact joint tumor-normal spectra
                # na_fig, na_axes = plt.subplots(1, len(Variation), sharex='all', sharey='all', squeeze=False)
                # for variant_index, variant_type in enumerate(Variation):
                #    self.normal_artifact_spectra[variant_index].density_plot_on_axis(na_axes[0, variant_index])
                # plotting.tidy_subplots(na_fig, na_axes, x_label="tumor fraction", y_label="normal fraction",
                #                       row_labels=[""], column_labels=[var_type.name for var_type in Variation])
                # summary_writer.add_figure("normal artifact spectra", na_fig, epoch)

    # map of Variant type to probability threshold that maximizes F1 score
    # loader is a Dataloader whose collate_fn is the PosteriorBatch constructor
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
            var_types.append(var_types_b)

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
