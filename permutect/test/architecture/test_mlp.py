import torch
from torch import nn

from permutect.architecture.mlp import MLP
from permutect.misc_utils import backpropagate


def test_layers():
    # a negative layer size is a residual block of that many layers, each as wide as its input, and each linear layer has
    # batch normalization before it and dropout after it
    mlp = MLP([5, 8, -2, 3], batch_normalize=True, dropout_p=0.1)
    assert (mlp.input_dimension(), mlp.output_dimension()) == (5, 3)
    linear_layers = [module for module in mlp.modules() if isinstance(module, nn.Linear)]
    assert [(layer.in_features, layer.out_features) for layer in linear_layers] == [(5, 8), (8, 8), (8, 8), (8, 3)]
    assert sum(isinstance(module, nn.BatchNorm1d) for module in mlp.modules()) == 4
    assert sum(isinstance(module, nn.Dropout) for module in mlp.modules()) == 4
    # the output is logits, without a nonlinearity after the last layer
    assert not isinstance(mlp._model[-1], nn.SELU) and not isinstance(mlp._model[-2], nn.SELU)
    assert mlp.forward(torch.randn(10, 5)).shape == (10, 3)


def test_xor_data():
    # A single hidden layer should suffice to classify a 2D XOR pattern where the 1st and 3rd quadrants are labeled 0 and
    # the 2nd and 4th are 1.  Really only 2 neurons are needed, but more speed convergence, and with 4 training got stuck
    # far from a solution for 3 seeds in 20.
    torch.manual_seed(0)
    x = torch.randn(1000, 2)
    y = (torch.sign(x[:, 0] * x[:, 1]) + 1) / 2  # labels are 0 / 1
    model = MLP([2, 8, 1])

    # Adam on all of the data at every step
    optimizer, loss_func, losses = torch.optim.Adam(model.parameters()), nn.BCEWithLogitsLoss(reduction="mean"), []
    for _ in range(10000):
        loss = loss_func(model.forward(x).reshape(y.shape), y)
        backpropagate(optimizer, loss)
        losses.append(loss.item())
    assert losses[-1] < 0.15
    assert losses[1000] < losses[0] and losses[2000] < losses[1000] and losses[3000] < losses[2000]

    pred = torch.sign(torch.sigmoid(model.forward(x).detach().squeeze()) - 0.5)
    errors = torch.sum(torch.abs((pred - torch.sign(y - 0.5)) / 2)).item()
    assert errors < len(x) / 50
