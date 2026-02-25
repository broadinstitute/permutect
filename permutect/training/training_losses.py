from permutect.architecture.artifact_model import BatchOutput
from permutect.data.batch import Batch
from permutect.metrics.loss_metrics import LossMetrics


class TrainingLosses:
    def __init__(self, num_sources: int, device):
        self.supervised_loss_metrics = LossMetrics(num_sources=num_sources, device=device)
        self.unsupervised_loss_metrics = LossMetrics(num_sources=num_sources, device=device)
        self.alt_count_loss_metrics = LossMetrics(num_sources=num_sources, device=device)
        self.source_prediction_loss_metrics = None if num_sources < 2 else LossMetrics(num_sources=num_sources, device=device)

        self.all_metrics = (self.supervised_loss_metrics, self.unsupervised_loss_metrics, self.alt_count_loss_metrics, self.source_prediction_loss_metrics)
        self.names = ("supervised-loss", "unsupervised-loss", "alt_count-loss", "source_prediction-loss")

    def record(self, batch: Batch, supervised_losses_b, unsupervised_losses_b,
               source_losses_b, alt_count_losses_b, output: BatchOutput):
        is_labeled_b = batch.get_is_labeled_mask()
        self.supervised_loss_metrics.record(batch, supervised_losses_b, is_labeled_b * output.weights)
        self.unsupervised_loss_metrics.record(batch, unsupervised_losses_b, (1 - is_labeled_b) * output.weights)
        self.source_prediction_loss_metrics.record(batch, source_losses_b, output.source_weights)
        self.alt_count_loss_metrics.record(batch, alt_count_losses_b, output.weights)

    def write_to_summary_writer(self, epoch_type, epoch, summary_writer, make_plots: bool):
        for (metric, name) in zip(self.all_metrics, self.names):
            if metric is not None:
                metric.put_on_cpu()
                metric.write_to_summary_writer(epoch_type, epoch, summary_writer, prefix=name)
                if make_plots:
                    metric.make_plots(summary_writer, name, epoch_type, epoch)

        self.supervised_loss_metrics.report_marginals(f"Supervised loss for {epoch_type.name} epoch {epoch}.")
        self.unsupervised_loss_metrics.report_marginals(f"Unsupervised loss for {epoch_type.name} epoch {epoch}.")

        if make_plots:
            self.supervised_loss_metrics.make_plots(summary_writer, "total weight of data vs alt and ref counts", epoch_type, epoch, type_of_plot="counts")
