import math

import torch

from permutect.training.checkpoint import Checkpoint


def make_model_and_checkpoint():
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    return model, Checkpoint(torch.device("cpu"), model, optimizer)


def perturb(model):
    with torch.no_grad():
        model.weight.add_(1.0)


def test_nan_loss_is_never_saved_as_best():
    model, checkpoint = make_model_and_checkpoint()
    checkpoint.save_checkpoint_if_needed(1, math.nan)
    assert checkpoint.best_checkpoint is None

    checkpoint.save_checkpoint_if_needed(2, 1.0)
    assert checkpoint.best_checkpoint == {"epoch": 2, "loss": 1.0}

    # neither a NaN nor a worse loss replaces the best checkpoint, but a better one does
    checkpoint.save_checkpoint_if_needed(3, math.nan)
    checkpoint.save_checkpoint_if_needed(4, 1.5)
    assert checkpoint.best_checkpoint == {"epoch": 2, "loss": 1.0}
    checkpoint.save_checkpoint_if_needed(5, 0.5)
    assert checkpoint.best_checkpoint == {"epoch": 5, "loss": 0.5}


def test_load_checkpoint_if_needed():
    model, checkpoint = make_model_and_checkpoint()

    # there is nothing to load before the first save, even for a NaN loss
    checkpoint.load_checkpoint_if_needed(math.nan)

    checkpoint.save_checkpoint_if_needed(1, 1.0)
    saved_weight = model.weight.detach().clone()

    # a loss up to twice the best keeps the current weights
    perturb(model)
    checkpoint.load_checkpoint_if_needed(1.5)
    assert not torch.equal(model.weight, saved_weight)

    # a larger loss, or a NaN loss, reloads the best checkpoint
    checkpoint.load_checkpoint_if_needed(2.5)
    assert torch.equal(model.weight, saved_weight)

    perturb(model)
    checkpoint.load_checkpoint_if_needed(math.nan)
    assert torch.equal(model.weight, saved_weight)
