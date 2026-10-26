import torch

from permutect.architecture.artifact_model import BatchLosses
from permutect.architecture.artifact_model import BatchOutput
from permutect.data.batch import Batch
from permutect.data.batch import BatchProperty
from permutect.metrics.loss_metrics import LossMetrics


class LossRecorder:
    def __init__(self, device, num_sources: int):
        self.primary_metrics = LossMetrics(num_sources=num_sources, device=device)
        self.count_metrics = LossMetrics(num_sources=num_sources, device=device)
        self.num_sources = num_sources

    def record(self, output: BatchOutput, losses: BatchLosses, batch: Batch):
        is_labeled_b = batch.get_is_labeled_mask()
        labeled_weights = is_labeled_b * output.weights
        unlabeled_weights = (1 - is_labeled_b) * output.weights

        self.primary_metrics.record(batch, losses.supervised_losses_b, labeled_weights)
        self.primary_metrics.record(batch, losses.unsupervised_losses_b, unlabeled_weights)
        self.count_metrics.record(batch, losses.alt_count_losses_b, output.weights)

    def mean_loss(self) -> float:
        """
        Mean over labels of each label's average semi-supervised loss.  Labels without data, such as artifact and
        variant when all data are unlabeled, are skipped instead of contributing 0/0 = NaN.  NaN only if there are no data.
        """
        totals_l = self.primary_metrics.totals_slvra.get_marginal(BatchProperty.LABEL)
        counts_l = self.primary_metrics.counts_slvra.get_marginal(BatchProperty.LABEL)
        has_data_l = counts_l > 0
        return torch.mean(totals_l[has_data_l] / counts_l[has_data_l]).item()

    def output_results(self, epoch_type, epoch, summary_writer, generate_plots):
        self.primary_metrics.put_on_cpu()
        self.count_metrics.put_on_cpu()
        self.primary_metrics.write_marginals(epoch_type, epoch, summary_writer, prefix="semisupervised-loss")
        self.count_metrics.write_marginals(epoch_type, epoch, summary_writer, prefix="alt-count-loss")
        self.primary_metrics.print_marginals(f"Semisupervised loss, {epoch_type.name} epoch {epoch}.")

        if generate_plots:
            self.primary_metrics.make_plots(summary_writer, "semisupervised loss", epoch_type, epoch)
            self.primary_metrics.make_plots(summary_writer, "total weight", epoch_type, epoch, plot_type="counts")
            self.count_metrics.make_plots(summary_writer, "alt count prediction loss", epoch_type, epoch)
