from typing import List

import torch
from matplotlib import pyplot as plt
from torch import nn, Tensor, IntTensor, repeat_interleave
from torch.nn import Parameter

from permutect.architecture.monotonic import MonoDense
from permutect.data.count_binning import MAX_REF_COUNT, MAX_ALT_COUNT
from permutect.metrics import plotting
from permutect.utils.enums import Variation


class FeatureClustering(nn.Module):
    VAR_TYPE_EMBEDDING_DIM = 10
    def __init__(self, feature_dimension: int, calibration_hidden_layer_sizes: List[int]):
        super(FeatureClustering, self).__init__()

        self.feature_dim = feature_dimension
        self.centroid_distance_normalization = Parameter(1 / torch.sqrt(torch.tensor(self.feature_dim)), requires_grad=False)

        # distance in feature space that marks the artifact logit = 0 decision boundary
        self.decision_boundary = Parameter(torch.tensor(1.0))

        # calibration takes [distance from origin, ref count, alt count] as input and maps it to [calibrated logit]
        # It is monotonically increasing in the input distance.  We don't currently constrain the count dependences, but
        # it's worth considering.

        # To ensure that the decision boundary maps to zero we take f(distance, counts) = g(distance, counts) - g(distance=decision boundary, counts),
        # where g has the desired monotonicity.

        # the input features are distance, ref count, alt count, var_type embedding
        # calibration is increasing in distance.
        self.distance_calibration = MonoDense(3 + FeatureClustering.VAR_TYPE_EMBEDDING_DIM, calibration_hidden_layer_sizes + [1], 1, 0)
        self.var_type_embeddings_ve = Parameter(torch.rand(len(Variation), FeatureClustering.VAR_TYPE_EMBEDDING_DIM))

    def calculate_logits(self, features_be: Tensor, ref_counts_b: IntTensor, alt_counts_b: IntTensor, var_types_b: IntTensor):
        dist_b = torch.norm(features_be, dim=-1) * self.centroid_distance_normalization
        calibrated_logits_b = self.calibrated_distances(dist_b, ref_counts_b, alt_counts_b, var_types_b)
        return calibrated_logits_b

    # avoid implicit forward calls because PyCharm doesn't recognize them
    def forward(self, features: Tensor):
        pass

    def calibrated_distances(self, distances_b: Tensor, ref_counts_b: Tensor, alt_counts_b: Tensor, var_types_b: IntTensor):
        # indices: 'b' for batch, 3 for logit, ref, alt
        ref_b1 = ref_counts_b.view(-1, 1) / MAX_REF_COUNT
        alt_b1 = alt_counts_b.view(-1, 1) / MAX_ALT_COUNT
        var_type_embeddings_ve = self.var_type_embeddings_ve[var_types_b]

        monotonic_inputs_be = torch.hstack((distances_b.view(-1, 1), ref_b1, alt_b1, var_type_embeddings_ve))

        # these are inputs with distance = decision boundary, the output of which must be zero
        zero_inputs_be = torch.hstack((self.decision_boundary * torch.ones_like(distances_b).view(-1, 1), ref_b1, alt_b1, var_type_embeddings_ve))
        result_b1 = self.distance_calibration.forward(monotonic_inputs_be) - self.distance_calibration.forward(zero_inputs_be)
        return result_b1.view(-1)

    def plot_distance_calibration(self, var_type: Variation, device, dtype):
        alt_counts = [1, 3, 5, 10, 15]
        ref_counts = [1, 3, 5, 10]
        distances = torch.arange(start=0, end=10, step=0.1, device=device, dtype=dtype)
        cal_fig, cal_axes = plt.subplots(len(alt_counts), len(ref_counts), sharex='all', sharey='all',
                                        squeeze=False, figsize=(10, 6), dpi=100)

        var_types_b = var_type * torch.ones(len(distances), device=device, dtype=torch.long)
        for row_idx, alt_count in enumerate(alt_counts):
            alt_counts_b = alt_count * torch.ones_like(distances, device=device, dtype=dtype)
            for col_idx, ref_count in enumerate(ref_counts):
                ref_counts_b = ref_count * torch.ones_like(distances, device=device, dtype=dtype)

                # TODO: different function call here
                calibrated = self.calibrated_distances(distances, ref_counts_b, alt_counts_b, var_types_b)
                plotting.simple_plot_on_axis(cal_axes[row_idx, col_idx], [(distances.detach().cpu(), calibrated.detach().cpu(), "")], None, None)

        plotting.tidy_subplots(cal_fig, cal_axes, x_label="ref count", y_label="alt count",
                               row_labels=[str(n) for n in alt_counts], column_labels=[str(n) for n in ref_counts])

        return cal_fig, cal_axes