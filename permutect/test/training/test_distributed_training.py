import itertools

import pytest
import torch
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter

from permutect.architecture.artifact_model import load_model
from permutect.data.batch import Batch
from permutect.data.memory_mapped_data import MemoryMappedData
from permutect.data.reads_dataset import ReadsDataset
from permutect.metrics.evaluation_metrics import EvaluationMetrics
from permutect.parameters import TrainingParameters
from permutect.test.process_group import run_in_process_group
from permutect.test.test_file_names import PERMUTECT_V0_9_2_ARTIFACT_MODEL
from permutect.test.test_file_names import PREPROCESSED_DATA
from permutect.training import model_training
from permutect.training.balancer import Balancer
from permutect.training.downsampler import Downsampler
from permutect.training.loss_recorder import LossRecorder
from permutect.utils.enums import Epoch
from permutect.utils.enums import ParameterSet

BATCH_SIZE = 16


@pytest.fixture(scope="module")
def data():
    return list(itertools.islice(MemoryMappedData.load_from_tarfile(PREPROCESSED_DATA).generate(), 2 * BATCH_SIZE))


def process_batches(balancer: Balancer, batches):
    for batch in batches:
        balancer.process_batch_and_compute_weights(batch, artifact_probs_b=torch.linspace(0, 1, batch.size()))


def balancer_state_after_one_batch_per_process(data) -> dict:
    Balancer.DATA_BEFORE_RECOMPUTE = BATCH_SIZE + 1  # so that the weights are recomputed after both batches
    balancer = Balancer(num_sources=1, device=torch.device("cpu"))
    rank = dist.get_rank()
    process_batches(balancer, [Batch(data[rank * BATCH_SIZE : (rank + 1) * BATCH_SIZE])])
    return balancer.state_dict()


def test_balancer_sums_counts_over_processes(data, monkeypatch):
    # every process has the counts and weights of a single process that sees the batches of both
    monkeypatch.setattr(Balancer, "DATA_BEFORE_RECOMPUTE", BATCH_SIZE + 1)
    single_process = Balancer(num_sources=1, device=torch.device("cpu"))
    process_batches(single_process, [Batch(data[:BATCH_SIZE]), Batch(data[BATCH_SIZE:])])
    assert not torch.equal(single_process.weights_slvra, torch.ones_like(single_process.weights_slvra))

    for state in run_in_process_group(2, balancer_state_after_one_batch_per_process, data):
        for name, tensor in single_process.state_dict().items():
            assert torch.allclose(state[name], tensor), name


def loss_recorder_sums_after_all_reduce():
    rank = dist.get_rank()
    recorder = LossRecorder(torch.device("cpu"), num_sources=1)
    for metrics in (recorder.primary_metrics, recorder.count_metrics):
        metrics.totals_slvra += rank + 1
        metrics.counts_slvra += 10 * (rank + 1)
    recorder.all_reduce()
    all_metrics = (recorder.primary_metrics, recorder.count_metrics)
    sums = [(m.totals_slvra.unique().tolist(), m.counts_slvra.unique().tolist()) for m in all_metrics]
    return sums, recorder.mean_loss()


def test_loss_recorder_sums_over_processes():
    # so that every process makes the same learning rate and checkpoint decisions
    for sums, mean_loss in run_in_process_group(2, loss_recorder_sums_after_all_reduce):
        assert sums == [([3.0], [30.0])] * 2
        assert mean_loss == pytest.approx(0.1)


def evaluated_and_expected_weights(tensorboard_dir) -> tuple[list[list[float]], list[float]]:
    """
    for each plot that evaluate_model made, the total weight of the training and validation data, and the total weight
    of all the processes' shards
    """
    device, rank, world_size = torch.device("cpu"), dist.get_rank(), dist.get_world_size()
    memory_mapped_data = MemoryMappedData.load_from_tarfile(PREPROCESSED_DATA)
    train_dataset = ReadsDataset(memory_mapped_data, 10, [0], rank=rank, world_size=world_size)
    valid_dataset = ReadsDataset(memory_mapped_data, 10, [1], rank=rank, world_size=world_size)

    evaluated = []
    EvaluationMetrics.make_plots = lambda self, *args, **kwargs: evaluated.append(self)
    model_training.evaluate_model(
        load_model(PERMUTECT_V0_9_2_ARTIFACT_MODEL, device=device),
        epoch=1,
        num_sources=1,
        balancer=Balancer(num_sources=1, device=device),  # whose weights are all 1
        downsampler=Downsampler(num_sources=1),
        train_loader=train_dataset.make_data_loader(batch_size=64),
        valid_loader=valid_dataset.make_data_loader(batch_size=64),
        summary_writer=SummaryWriter(f"{tensorboard_dir}/{rank}"),
    )
    epoch_types = (Epoch.TRAIN, Epoch.VALID)
    weights = [[metrics.accuracy_metrics_by_epoch_type[e].sum().item() for e in epoch_types] for metrics in evaluated]
    # each datum is downsampled three times, and with one worker per process the shards are the size of the datasets
    expected = [3.0 * world_size * len(dataset) for dataset in (train_dataset, valid_dataset)]
    return weights, expected


def test_evaluation_sums_over_processes(tmp_path):
    # each process evaluates its own shard, and process 0 plots all the shards
    (weights_0, expected), (weights_1, _) = run_in_process_group(2, evaluated_and_expected_weights, str(tmp_path))
    assert weights_0 == [pytest.approx(expected)]
    assert weights_1 == []


def trained_model_state(tensorboard_dir) -> dict:
    device, rank, world_size = torch.device("cpu"), dist.get_rank(), dist.get_world_size()
    memory_mapped_data = MemoryMappedData.load_from_tarfile(PREPROCESSED_DATA)
    train_dataset = ReadsDataset(memory_mapped_data, 10, [0], rank=rank, world_size=world_size)
    valid_dataset = ReadsDataset(memory_mapped_data, 10, [1], rank=rank, world_size=world_size)
    model = load_model(PERMUTECT_V0_9_2_ARTIFACT_MODEL, device=device)
    training_params = TrainingParameters(
        batch_size=64,
        num_epochs=2,
        learning_rate=0.001,
        weight_decay=0.01,
        num_workers=0,
        trainable_parameter_sets=[ParameterSet.WHOLE_MODEL],
    )
    with SummaryWriter(f"{tensorboard_dir}/{rank}") as summary_writer:
        model_training.train_artifact_model(model, train_dataset, valid_dataset, training_params, summary_writer)
    return model.state_dict()


def test_distributed_training_keeps_the_models_of_the_processes_the_same(tmp_path):
    # the processes train on different shards, and synchronizing their gradients keeps their models the same
    states = run_in_process_group(2, trained_model_state, str(tmp_path))
    pretrained = load_model(PERMUTECT_V0_9_2_ARTIFACT_MODEL, device=torch.device("cpu")).state_dict()
    assert any(not torch.equal(states[0][name], tensor) for name, tensor in pretrained.items())
    for name, tensor in states[0].items():
        assert torch.equal(states[1][name], tensor), name
