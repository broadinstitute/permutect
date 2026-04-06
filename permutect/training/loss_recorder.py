from permutect.architecture.artifact_model import BatchLosses
from permutect.architecture.artifact_model import BatchOutput
from permutect.data.batch import Batch
from permutect.metrics.loss_metrics import LossMetrics


class LossRecorder:
    def __init__(self, device, num_sources: int):
        self.primary_metrics = LossMetrics(num_sources=num_sources, device=device)
        self.count_metrics = LossMetrics(num_sources=num_sources, device=device)
        self.source_metrics = LossMetrics(num_sources=num_sources, device=device)
        self.num_sources = num_sources

    def record(self, output: BatchOutput, losses: BatchLosses, batch: Batch):
        is_labeled_b = batch.get_is_labeled_mask()
        labeled_weights = is_labeled_b * output.weights
        unlabeled_weights = (1 - is_labeled_b) * output.weights

        self.primary_metrics.record(batch, losses.supervised_losses_b, labeled_weights)
        self.primary_metrics.record(batch, losses.unsupervised_losses_b, unlabeled_weights)
        self.source_metrics.record(batch, losses.source_prediction_losses_b, output.source_weights)
        self.count_metrics.record(batch, losses.alt_count_losses_b, output.weights)

    def output_results(self, epoch_type, epoch, summary_writer, generate_plots):
        self.primary_metrics.put_on_cpu()
        self.count_metrics.put_on_cpu()
        self.source_metrics.put_on_cpu()
        self.primary_metrics.write_to_summary_writer(
            epoch_type, epoch, summary_writer, prefix="semisupervised-loss"
        )
        self.count_metrics.write_to_summary_writer(epoch_type, epoch, summary_writer, prefix="alt-count-loss")
        self.source_metrics.write_to_summary_writer(
            epoch_type, epoch, summary_writer, prefix="source-loss"
        )
        self.primary_metrics.report_marginals(f"Semisupervised loss, {epoch_type.name} epoch {epoch}.")
        if self.num_sources > 1:
            self.source_metrics.report_marginals(f"Source loss, {epoch_type.name} epoch {epoch}.")

        if generate_plots:
            self.primary_metrics.make_plots(summary_writer, "semisupervised loss", epoch_type, epoch)
            self.primary_metrics.make_plots(
                summary_writer,
                "total weight of data vs alt and ref counts",
                epoch_type,
                epoch,
                type_of_plot="counts",
            )
            self.count_metrics.make_plots(summary_writer, "alt count prediction loss", epoch_type, epoch)
            if self.num_sources > 1:
                self.source_metrics.make_plots(
                    summary_writer, "source prediction loss", epoch_type, epoch
                )
