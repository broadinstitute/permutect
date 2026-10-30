import torch

from permutect.architecture.mlp import MLP
from permutect.misc_utils import backpropagate


# test with artificial data where a*x = 0 is a perfect linear separator, with a margin around it
def test_linearly_separable_data():
    torch.manual_seed(0)
    input_dim = 3
    num_samples = 1000
    # centered data, so that samples fall on both sides of the separator.  With nonnegative a and x every label was 1.
    a = torch.randn(input_dim, 1)
    x = torch.randn(num_samples, input_dim)
    x = x[torch.abs(torch.matmul(x, a).squeeze()) > 0.1 * torch.norm(a)]
    y = (torch.sign(torch.matmul(x, a)) + 1) / 2  # labels are 0 / 1
    assert 0.2 < torch.mean(y).item() < 0.8

    layer_sizes = [input_dim, 1]
    model = MLP(layer_sizes)

    loss_func = torch.nn.BCEWithLogitsLoss(reduction="mean")
    optimizer = torch.optim.Adam(model.parameters())
    loss_list = []

    num_epochs = 10000
    for epoch in range(num_epochs):
        prediction = model.forward(x)
        loss = loss_func(prediction, y)
        backpropagate(optimizer, loss)
        loss_list.append(loss.item())

    assert loss_list[-1] < 0.1
    assert loss_list[1000] < loss_list[0]
    assert loss_list[2000] < loss_list[1000]
    assert loss_list[3000] < loss_list[2000]

    pred = torch.sign(torch.sigmoid(model.forward(x)) - 0.5)
    lab = torch.sign(y - 0.5)

    errors = torch.sum(torch.abs((pred - lab) / 2)).item()

    # we should have perfect accuracy
    assert errors == 0


# test with annular data where y = 1 when  1/3 < norm(x) < 2/3
def test_annular_data():
    torch.manual_seed(42)
    input_dim = 3
    num_samples = 1000
    x = torch.randn(num_samples, input_dim) / torch.sqrt(torch.tensor([input_dim]))

    norms = torch.norm(x, dim=1)
    y = (torch.sign(norms - 0.33) * torch.sign(0.66 - norms) + 1) / 2

    layer_sizes = [input_dim, 5, 5, 5, 5, 1]
    model = MLP(layer_sizes)

    loss_func = torch.nn.BCEWithLogitsLoss(reduction="mean")
    optimizer = torch.optim.Adam(model.parameters())
    loss_list = []

    num_epochs = 10000
    for epoch in range(num_epochs):
        prediction = model.forward(x)
        loss = loss_func(torch.squeeze(prediction), y)
        backpropagate(optimizer, loss)
        loss_list.append(loss.item())

    assert loss_list[-1] < 0.2

    pred = torch.squeeze(torch.sign(torch.sigmoid(model.forward(x)) - 0.5))
    lab = torch.sign(y - 0.5)

    errors = torch.sum(torch.abs((pred - lab) / 2)).item()

    # we should have perfect accuracy
    assert errors < 100


# A single hidden layer should suffice to perfectly classify a 2D XOR pattern where
# the 1st and 3rd quadrants are labeled 0 and the 2nd and 4th are 1
def test_xor_data():
    torch.manual_seed(0)
    num_samples = 1000
    x = torch.randn(num_samples, 2)
    y = (torch.sign(x[:, 0] * x[:, 1]) + 1) / 2  # labels are 0 / 1

    # Use a single hidden layer.  Really only 2 neurons are needed, but more speeds convergence, and with 4 training
    # got stuck far from a solution for 3 seeds in 20
    layer_sizes = [2, 8, 1]
    model = MLP(layer_sizes)

    loss_func = torch.nn.BCEWithLogitsLoss(reduction="mean")
    optimizer = torch.optim.Adam(model.parameters())
    loss_list = []

    num_epochs = 10000
    for epoch in range(num_epochs):
        prediction = model.forward(x)
        loss = loss_func(prediction.squeeze(), y)
        backpropagate(optimizer, loss)
        loss_list.append(loss.item())

    assert loss_list[-1] < 0.15
    assert loss_list[1000] < loss_list[0]
    assert loss_list[2000] < loss_list[1000]
    assert loss_list[3000] < loss_list[2000]

    pred = torch.sign(torch.sigmoid(model.forward(x).detach().squeeze()) - 0.5)
    lab = torch.sign(y - 0.5)

    errors = torch.sum(torch.abs((pred - lab) / 2)).item()

    # we should have near-perfect accuracy
    assert errors < num_samples / 50
