import torch

from permutect.architecture.monotonic import MonoDense
from permutect.test.architecture.test_mlp import train


def test_is_monotonic():
    torch.manual_seed(0)
    input_dim = 3
    num_samples = 100
    x = torch.randn(num_samples, input_dim)
    x[:, 0] = torch.arange(num_samples) / num_samples  # 0th column is sorted least to greatest
    x[:, 1] = 0.5  # other columns are constant
    x[:, 2] = 0.7

    # an initialized random model.  We're not learning anything here.
    model = MonoDense(
        input_dimension=input_dim,
        output_dimensions=[12, -2, -2, 1],
        num_increasing=1,
        num_decreasing=0,
    )

    prediction = model.forward(x).flatten()

    sorted_prediction = prediction.sort().values

    assert torch.sum(torch.abs(prediction - sorted_prediction)) < 0.00001


# test with artificial data where y = a dot x where a is a positive vector
def test_monotonic_linear_data():
    torch.manual_seed(0)
    input_dim = 3
    num_samples = 100
    a = torch.ones(input_dim)
    x = torch.rand(num_samples, input_dim)
    y = torch.sum(x * a, dim=1)  # row-by-row dot product

    model = MonoDense(input_dimension=input_dim, output_dimensions=[1], num_increasing=input_dim, num_decreasing=0)
    loss_list = train(model, x, y, torch.nn.MSELoss(reduction="mean"))

    assert loss_list[-1] < 0.01
    assert loss_list[1000] < loss_list[0]
    assert loss_list[2000] < loss_list[1000]
    assert loss_list[3000] < loss_list[2000]


# test with artificial data where y = x1 - x2 + x3^2; monotonic increasing in x1, decreasing in x2, and neither in x3
# we need more than one layer to get the quadratic
def test_mix():
    torch.manual_seed(0)
    input_dim = 3
    num_samples = 100
    x = torch.rand(num_samples, input_dim)
    y = x[:, 0] - x[:, 1] + torch.square(x[:, 2])

    model = MonoDense(
        input_dimension=input_dim,
        output_dimensions=[6, 6, 6, 1],
        num_increasing=1,
        num_decreasing=1,
    )
    loss_list = train(model, x, y, torch.nn.MSELoss(reduction="mean"))

    assert loss_list[-1] < 0.01
    assert loss_list[1000] < loss_list[0]
    assert loss_list[2000] < loss_list[1000]
    assert loss_list[3000] < loss_list[2000]


def test_cant_learn_non_monotonic():
    torch.manual_seed(0)
    input_dim = 1
    num_samples = 100
    x = torch.randn(num_samples, input_dim)
    y = torch.square(x)

    model = MonoDense(
        input_dimension=input_dim,
        output_dimensions=[6, 6, 6, 1],
        num_increasing=1,
        num_decreasing=0,
    )
    loss_func = torch.nn.MSELoss(reduction="mean")
    train(model, x, y, loss_func)
    assert loss_func(model.forward(x).reshape(y.shape), y).item() > 0.1


def test_cubic():
    torch.manual_seed(0)
    input_dim = 3
    num_samples = 100
    x = torch.rand(num_samples, input_dim)
    y = torch.sum(x**3, dim=1)

    model = MonoDense(
        input_dimension=input_dim,
        output_dimensions=[12, 12, 12, 1],
        num_increasing=input_dim,
        num_decreasing=0,
    )
    loss_list = train(model, x, y, torch.nn.MSELoss(reduction="mean"))

    # y has variance 0.24.  The final loss was 0.011 to 0.013 for 3 seeds in 20.
    assert loss_list[-1] < 0.02
    assert loss_list[1000] < loss_list[0]
    assert loss_list[2000] < loss_list[1000]
    assert loss_list[3000] < loss_list[2000]
