from torch import nn, Tensor
import torch

"""
Constrain a vector to have unit norm.  Usage in a Module:

# initialize arbitrarily -- self.unit_vec does not obey the constraint
self.unit_vec = nn.Parameter(torch.rand(dim))

# this moves self.unit_vec to self.original_unit_vec or something similar, and creates a
# replacement self.unit_vec that always equals the original unit_vec with UnitVector::forward applied,
# thereby ensuring it respects the constraint.
parametrize.register_parametrization(self, "unit_vec", UnitVector())

Note that the forward function uses norm(. . ., dim=-1) so this works equally well for a single vector,
or 2D, 3D etc arrays of vectors.
"""
class UnitVector(nn.Module):
    def __init__(self):
        super().__init__()

    # project the unconstrained underlying tensor
    def forward(self, X: Tensor):
        return X / torch.norm(X, dim=-1, keepdim=True)

    # we don't define a right inverse because random initialization works well for the underlying
    # unnormalized vector
    #   def right_inverse(self, A):

"""
Constrain a parameter to have all positive values. Usage in a Module:

# initialize with positive values -- these are the actual positive values, not the underlying
# pre-exponentiated unconstrained values
self.positive_values = nn.Parameter(torch.ones(dim1, dim2))

parametrize.register_parametrization(self, "positive_values", PositiveNumber())
"""
class PositiveNumber(nn.Module):
    def __init__(self, n):
        super().__init__()

    # Maps the unconstrained tensor X to a positive-valued tensor
    def forward(self, X: Tensor):
        return torch.exp(X)

    # Maps a positive-valued tensor P to the unconstrained log tensor
    def right_inverse(self, P: Tensor):
        return torch.log(P)

"""
Constrain a parameter to have values in range (min_val, max_val)

# initialize with bounded values -- these are the actual bounded values, not the underlying unconstrained values
self.positive_values = nn.Parameter(torch.ones(dim1, dim2))

parametrize.register_parametrization(self, "positive_values", PositiveNumber())
"""
class BoundedNumber(nn.Module):
    def __init__(self, min_val: float, max_val: float):
        super().__init__()
        assert min_val <= max_val
        self.min_val = min_val
        self.max_val = max_val
        self.size = max_val - min_val

    # Maps the unconstrained tensor X to a positive-valued tensor
    def forward(self, X: Tensor):
        return self.size * torch.sigmoid(X) + self.min_val

    # Maps a positive-valued tensor P to the unconstrained log tensor
    def right_inverse(self, P: Tensor):
        # torch.logit is the inverse of torch.sigmoid
        return torch.logit((P - self.min_val) / self.size)
