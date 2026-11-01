import itertools

import torch
from torch import IntTensor
from torch import Tensor
from torch.distributions import Beta
from torch.nn import Module
from torch.nn import Parameter
from torch.nn.parameter import Buffer
from torch.nn.utils import parametrize

from permutect.architecture.parameterizations import LogWeights
from permutect.data.batch import Batch
from permutect.data.batch import BatchIndexedTensor
from permutect.data.count_binning import COUNT_BIN_SKIP
from permutect.data.count_binning import MAX_ALT_COUNT
from permutect.data.count_binning import MAX_REF_COUNT
from permutect.data.count_binning import NUM_ALT_COUNT_BINS
from permutect.data.count_binning import NUM_REF_COUNT_BINS
from permutect.data.count_binning import alt_count_bin_index
from permutect.data.count_binning import ref_count_bin_index
from permutect.misc_utils import backpropagate
from permutect.utils.stats_utils import beta_binomial_log_lk


def binned_transitions(max_count: int, num_bins: int, bin_index, alpha_k11: Tensor, beta_k11: Tensor) -> Tensor:
    """
    The probabilities of downsampling each count bin to each count bin under each beta binomial basis component 'k',
    indexed by k, the original count bin and the downsampled count bin
    """
    counts = IntTensor(range(max_count + 1))
    counts_k1d, downsampled_counts_k1d = counts.view(1, -1, 1), counts.view(1, 1, -1)  # dummy indices for broadcasting
    trans_kcd = torch.where(
        counts_k1d >= downsampled_counts_k1d,
        torch.exp(beta_binomial_log_lk(n=counts_k1d, k=downsampled_counts_k1d, alpha=alpha_k11, beta=beta_k11)),
        0,
    )

    # now we need to bin this.  If you think about it carefully, the appropriate thing to do is *sum* over downsampled
    # counts that correspond to the same bin and to *average* over original counts.  We could do this in some fancy vectorized
    # way, but because this is a one-time cost during construction we employ a slower but clearer approach.
    # we implement the average by summing over everything and dividing by the count bin skip, which is the number
    # of counts per bin
    binned_trans_kbb = torch.zeros(len(alpha_k11), num_bins, num_bins)
    for count, downsampled_count in itertools.product(range(max_count + 1), repeat=2):
        binned_trans_kbb[:, bin_index(count), bin_index(downsampled_count)] += trans_kcd[:, count, downsampled_count]
    return binned_trans_kbb / COUNT_BIN_SKIP


class Downsampler(Module):
    # downsampling is done as a mixture of beta binomials with a *fixed* set of basis beta distributions
    BETA_BASIS_SHAPES = torch.tensor([[1.0, 1.0], [1.0, 5.0], [5.0, 1.0], [5.0, 5.0]])

    def __init__(self, num_sources: int):
        super(Downsampler, self).__init__()
        self.num_sources = num_sources

        # copy this as a member so that it gets moved to the correct device with the downsampler
        self.beta_basis = Buffer(Downsampler.BETA_BASIS_SHAPES)

        # these are 3D -- with two dummy read count indices -- for broadcasting
        alpha_k11 = self.beta_basis[:, 0].view(-1, 1, 1)
        beta_k11 = self.beta_basis[:, 1].view(-1, 1, 1)

        # transitions to downsampled count bins, indexed by basis 'k', then ref (r) and downsampled ref (y) count bins, or
        # alt (a) and downsampled alt (z) count bins
        self.binned_ref_trans_kry = Buffer(
            binned_transitions(MAX_REF_COUNT, NUM_REF_COUNT_BINS, ref_count_bin_index, alpha_k11, beta_k11)
        )
        # in the downsampling we guarantee at least one alt read count, so a beta binomial that samples no alt reads
        # ends up with one read, hence is in bin 0, which is where alt_count_bin_index puts 0.
        self.binned_alt_trans_haz = Buffer(
            binned_transitions(MAX_ALT_COUNT, NUM_ALT_COUNT_BINS, alt_count_bin_index, alpha_k11, beta_k11)
        )

        # for each source/label/var type/ ref/alt bin we have mixture component weights for both ref and alt downsampling
        # 'k' and 'h' are both indices to denote the mixture components
        slvra_shape = BatchIndexedTensor.shape_without_logits(num_sources)
        slvrak_shape = slvra_shape + (len(self.beta_basis),)

        # the log weights tensors should never be changed when training a model and as such should usually never
        # have gradients.  However, 1) defining them as parameters lets us parametrize them with LogWeights and
        # 2) we do train them when fitting the downsampler.  Thus we define them as Parameters rather than Buffers
        # BUT we always detach() them.  Furthermore, when training a model the optimizer should only involve the
        # model's parameters, not the downsampler's
        self.log_ref_weights_slvrak = Parameter(torch.zeros(slvrak_shape))
        parametrize.register_parametrization(self, "log_ref_weights_slvrak", LogWeights())
        self.log_alt_weights_slvrah = Parameter(torch.zeros(slvrak_shape))
        parametrize.register_parametrization(self, "log_alt_weights_slvrah", LogWeights())

    def weights_parameters(self):
        return [
            self.parametrizations.log_ref_weights_slvrak.original,
            self.parametrizations.log_alt_weights_slvrah.original,
        ]

    # this is used in model training, NOT for fitting the downsampler, so for security we detach the weights
    def calculate_downsampling_fractions(self, batch: Batch) -> tuple[Tensor, Tensor]:
        # we will flatten all the batch indices -- slvra, but not k -- to get a 2D tensor indexed by the batch's
        # flattened indices ('f') and the downsampler's mixture index k
        ref_weights_fk = torch.exp(self.log_ref_weights_slvrak.detach()).view(-1, len(self.beta_basis))
        alt_weights_fk = torch.exp(self.log_alt_weights_slvrah.detach()).view(-1, len(self.beta_basis))

        # use the weights for choosing from random samples
        ref_weights_bk = ref_weights_fk[batch.batch_indices().flattened_idx]
        alt_weights_bk = alt_weights_fk[batch.batch_indices().flattened_idx]

        refk_b = torch.multinomial(ref_weights_bk, num_samples=1)  # one sample for each batch element
        altk_b = torch.multinomial(alt_weights_bk, num_samples=1)

        # The '2' indicates that each beta shape has two elements, alpha and beta
        ref_shapes_b2 = self.beta_basis[refk_b.flatten()]
        alt_shapes_b2 = self.beta_basis[altk_b.flatten()]
        ref_fracs_b = Beta(ref_shapes_b2[:, 0], ref_shapes_b2[:, 1]).sample().view(-1)
        alt_fracs_b = Beta(alt_shapes_b2[:, 0], alt_shapes_b2[:, 1]).sample().view(-1)
        return ref_fracs_b, alt_fracs_b

    # used to fit the downsampler, log weights are NOT detach()ed
    def calculate_expected_downsampled_counts(self, counts_slvra: BatchIndexedTensor):
        ref_weights_slvrak = torch.exp(self.log_ref_weights_slvrak)
        alt_weights_slvrah = torch.exp(self.log_alt_weights_slvrah)

        # in words, we're calculating (recall that y, z are downsampled ref, alt count bins)
        # result_slvyz = sum_{rakh} counts_slvra * ref_weights_slvrak * alt_weights_slvrah * ref_trans_kry * alt_trans_haz
        result_slvyz = torch.einsum(
            "slvra, slvrak, slvrah, kry, haz->slvyz",
            counts_slvra,
            ref_weights_slvrak,
            alt_weights_slvrah,
            self.binned_ref_trans_kry,
            self.binned_alt_trans_haz,
        )
        return result_slvyz

    def optimize_downsampling_balance(self, counts_slvra: BatchIndexedTensor):
        optimizer = torch.optim.AdamW(self.parameters())
        # TODO: magic constant -- choose when to end optimization more intelligently
        for step in range(10000):
            expected_slvyz = self.calculate_expected_downsampled_counts(counts_slvra)

            # divide by the total over all counts for each slv bin to get a probability distribution over output r/a count bins
            normalized_slvyz = expected_slvyz / torch.sum(expected_slvyz, dim=(-2, -1), keepdim=True)

            # we want downsampled counts to be as even as possible among all ref and alt count bins.  This is equivalent
            # to minimizing the sum of squared downsampled counts.  Since the downsampling
            # preserves total probability, it can't minimize this loss by scaling down the result.  It can only minimize
            # it by redistributing.
            sums_of_squares_slv = torch.sum(torch.square(normalized_slvyz), dim=(-2, -1))
            loss = torch.sum(sums_of_squares_slv)
            backpropagate(optimizer, loss)
