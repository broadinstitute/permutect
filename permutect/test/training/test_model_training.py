import pytest
import torch
from torch.utils.tensorboard import SummaryWriter

from permutect.architecture.artifact_model import load_model
from permutect.data.memory_mapped_data import MemoryMappedData
from permutect.data.reads_dataset import ReadsDataset
from permutect.test.test_file_names import PERMUTECT_V0_9_2_ARTIFACT_MODEL
from permutect.test.test_file_names import PREPROCESSED_DATA
from permutect.training import model_training
from permutect.training.balancer import Balancer
from permutect.training.downsampler import Downsampler
from permutect.utils.enums import Epoch


@pytest.fixture(scope="module")
def memory_mapped_data():
    return MemoryMappedData.load_from_tarfile(PREPROCESSED_DATA)


def run_validation_epoch(memory_mapped_data, tensorboard_dir, monkeypatch, epoch, epochs_per_evaluation, last_epoch):
    """
    run one validation epoch on small subsets of the data
    return whether the training Balancer changed and how many times the model was evaluated
    """
    device = torch.device("cpu")
    model = load_model(PERMUTECT_V0_9_2_ARTIFACT_MODEL, device=device)
    train_dataset = ReadsDataset(memory_mapped_data, num_folds=10, folds_to_use=[0])
    valid_dataset = ReadsDataset(memory_mapped_data, num_folds=10, folds_to_use=[1])
    num_sources = train_dataset.num_sources()

    balancer = Balancer(num_sources=num_sources, device=device).to(device=device, dtype=model._dtype)
    downsampler = Downsampler(num_sources=num_sources).to(device=device, dtype=model._dtype)
    downsampler.optimize_downsampling_balance(train_dataset.totals_slvra.to(device=device))
    balancer_state_before = {name: tensor.clone() for name, tensor in balancer.state_dict().items()}

    evaluations = []

    def evaluate_model_spy(*args, **kwargs):
        evaluations.append(args)
        return evaluate_model(*args, **kwargs)

    evaluate_model = model_training.evaluate_model
    monkeypatch.setattr(model_training, "evaluate_model", evaluate_model_spy)

    # a validation epoch doesn't use the checkpoint, optimizer, or scheduler
    model_training.train_one_epoch(
        balancer,
        checkpoint=None,
        device=device,
        downsampler=downsampler,
        epoch=epoch,
        epoch_type=Epoch.VALID,
        epochs_per_evaluation=epochs_per_evaluation,
        last_epoch=last_epoch,
        model=model,
        num_sources=num_sources,
        summary_writer=SummaryWriter(tensorboard_dir),
        train_loader=train_dataset.make_data_loader(batch_size=64),
        train_optimizer=None,
        train_scheduler=None,
        valid_loader=valid_dataset.make_data_loader(batch_size=64),
    )

    balancer_changed = balancer.count_since_last_recomputation != 0 or any(
        not torch.equal(tensor, balancer_state_before[name]) for name, tensor in balancer.state_dict().items()
    )
    return balancer_changed, len(evaluations)


def test_validation_and_evaluation_do_not_update_the_training_balancer(memory_mapped_data, tmp_path, monkeypatch):
    # the last epoch, when plots and the evaluation are due
    balancer_changed, num_evaluations = run_validation_epoch(
        memory_mapped_data, tmp_path, monkeypatch, epoch=1, epochs_per_evaluation=5, last_epoch=1
    )
    assert num_evaluations == 1
    assert not balancer_changed


def test_evaluation_only_runs_when_plots_are_due(memory_mapped_data, tmp_path, monkeypatch):
    _, num_evaluations = run_validation_epoch(
        memory_mapped_data, tmp_path, monkeypatch, epoch=1, epochs_per_evaluation=5, last_epoch=2
    )
    assert num_evaluations == 0
