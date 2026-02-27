from permutect.architecture.artifact_model import BatchOutput, BatchLosses
from permutect.data.batch import Batch
from permutect.metrics.loss_metrics import LossMetrics


class LossRecorder:
    def __init__(self, device, num_sources: int):
        self.semisupervised_loss_metrics = LossMetrics(num_sources=num_sources, device=device)
        self.alt_count_loss_metrics = LossMetrics(num_sources=num_sources, device=device)
        self.source_prediction_loss_metrics = LossMetrics(num_sources=num_sources, device=device)
        self.num_sources = num_sources

    def record(self, output: BatchOutput, losses: BatchLosses, batch: Batch):
        is_labeled_b = batch.get_is_labeled_mask()
        self.semisupervised_loss_metrics.record(batch, losses.supervised_losses_b, is_labeled_b * output.weights)
        self.semisupervised_loss_metrics.record(batch, losses.unsupervised_losses_b, output.weights)
        self.source_prediction_loss_metrics.record(batch, losses.source_prediction_losses_b, output.source_weights)
        self.alt_count_loss_metrics.record(batch, losses.alt_count_losses_b, output.weights)

    def output_results(self, epoch_type, epoch, summary_writer, generate_plots):
        self.semisupervised_loss_metrics.put_on_cpu()
        self.alt_count_loss_metrics.put_on_cpu()
        self.source_prediction_loss_metrics.put_on_cpu()
        self.semisupervised_loss_metrics.write_to_summary_writer(epoch_type, epoch, summary_writer, prefix="semisupervised-loss")
        self.alt_count_loss_metrics.write_to_summary_writer(epoch_type, epoch, summary_writer, prefix="alt-count-loss")
        self.source_prediction_loss_metrics.write_to_summary_writer(epoch_type, epoch, summary_writer, prefix="source-loss")
        self.semisupervised_loss_metrics.report_marginals(f"Semisupervised loss for {epoch_type.name} epoch {epoch}.")
        if self.num_sources > 1:
            self.source_prediction_loss_metrics.report_marginals(f"Source prediction loss for {epoch_type.name} epoch {epoch}.")

        if generate_plots:
            self.semisupervised_loss_metrics.make_plots(summary_writer, "semisupervised loss", epoch_type, epoch)
            self.semisupervised_loss_metrics.make_plots(summary_writer, "total weight of data vs alt and ref counts", epoch_type, epoch,
                                    type_of_plot="counts")
            self.alt_count_loss_metrics.make_plots(summary_writer, "alt count prediction loss", epoch_type, epoch)
            if self.num_sources > 1:
                self.source_prediction_loss_metrics.make_plots(summary_writer, "source prediction loss", epoch_type, epoch)
