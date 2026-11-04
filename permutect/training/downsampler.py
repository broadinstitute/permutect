import torch
from torch import Tensor
from torch.distributions import Beta
from torch.nn import Module
from torch.nn import Parameter
from torch.nn.parameter import Buffer
from torch.nn.utils import parametrize

from permutect.architecture.parameterizations import LogWeights
from permutect.data.batch import Batch
from permutect.data.batch import BatchIndexedTensor
from permutect.data.count_binning import MAX_ALT_COUNT
from permutect.data.count_binning import MAX_REF_COUNT
from permutect.data.count_binning import MIN_ALT_COUNT
from permutect.data.count_binning import NUM_ALT_COUNT_BINS
from permutect.data.count_binning import NUM_REF_COUNT_BINS
from permutect.data.count_binning import alt_count_bin_index
from permutect.data.count_binning import ref_count_bin_index
from permutect.misc_utils import backpropagate
from permutect.utils.stats_utils import beta_binomial_log_lk

# fitting stops when the loss decreases by less than FITTING_TOLERANCE times itself over FITTING_STEPS_PER_CHECK steps
FITTING_LEARNING_RATE = 0.01
FITTING_TOLERANCE = 1e-5
FITTING_STEPS_PER_CHECK = 100
MAX_FITTING_STEPS = 20000


def count_transitions(
    counts: Tensor, num_bins: int, bin_index, alpha_k11: Tensor, beta_k11: Tensor, num_always_kept: int = 0
) -> Tensor:
    """
    The probabilities of downsampling each of the counts to each count bin under each beta binomial basis component
    'k', indexed by k, the count and the downsampled count bin.  Downsampling always keeps num_always_kept reads and
    keeps each other read with a beta-distributed probability, so it takes a count n to a count m with probability
    BetaBinomial(m - num_always_kept | n - num_always_kept).
    """
    downsampled_counts = torch.arange(num_always_kept, int(counts.max()) + 1)
    # the numbers of reads beyond those always kept, with dummy indices for broadcasting
    extra_k1m = (downsampled_counts - num_always_kept).view(1, 1, -1)
    extra_kn1 = (counts - num_always_kept).view(1, -1, 1)
    trans_knm = torch.where(
        extra_kn1 >= extra_k1m,
        torch.exp(beta_binomial_log_lk(n=extra_kn1, k=extra_k1m, alpha=alpha_k11, beta=beta_k11)),
        0,
    )

    # sum over the downsampled counts in each bin
    trans_knb = torch.zeros(len(alpha_k11), len(counts), num_bins)
    return trans_knb.index_add_(2, bin_index(downsampled_counts), trans_knm)


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

        # the ref counts (c) and alt counts (d) of the count totals that the downsampler balances, and their bins
        ref_counts_c = torch.arange(MAX_REF_COUNT + 1)
        alt_counts_d = torch.arange(MIN_ALT_COUNT, MAX_ALT_COUNT + 1)
        self.ref_bins_c = Buffer(ref_count_bin_index(ref_counts_c))
        self.alt_bins_d = Buffer(alt_count_bin_index(alt_counts_d))

        # transitions to downsampled count bins, indexed by basis 'k', then ref count (c) and downsampled ref count bin
        # (y), or alt count (d) and downsampled alt count bin (z)
        self.ref_trans_kcy = Buffer(
            count_transitions(ref_counts_c, NUM_REF_COUNT_BINS, ref_count_bin_index, alpha_k11, beta_k11)
        )
        # DownsampledBatch always keeps one alt read and keeps each of the others with the beta-distributed probability
        self.alt_trans_hdz = Buffer(
            count_transitions(
                alt_counts_d, NUM_ALT_COUNT_BINS, alt_count_bin_index, alpha_k11, beta_k11, num_always_kept=1
            )
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

    def _count_weights(self, log_weights_slvrak: Tensor) -> Tensor:
        # the weights of each ref and alt count (c, d) are those of their bins (r, a)
        return torch.exp(log_weights_slvrak)[:, :, :, self.ref_bins_c][:, :, :, :, self.alt_bins_d]

    # used to fit the downsampler, log weights are NOT detach()ed
    def calculate_expected_downsampled_counts(self, count_totals_slvcd: Tensor):
        # in words, we're calculating (recall that y, z are downsampled ref, alt count bins)
        # result_slvyz = sum_{cdkh} counts_slvcd * ref_weights_slvcdk * alt_weights_slvcdh * ref_trans_kcy * alt_trans_hdz
        result_slvyz = torch.einsum(
            "slvcd, slvcdk, slvcdh, kcy, hdz->slvyz",
            count_totals_slvcd,
            self._count_weights(self.log_ref_weights_slvrak),
            self._count_weights(self.log_alt_weights_slvrah),
            self.ref_trans_kcy,
            self.alt_trans_hdz,
        )
        return result_slvyz

    def balance_loss(self, count_totals_slvcd: Tensor) -> Tensor:
        expected_slvyz = self.calculate_expected_downsampled_counts(count_totals_slvcd)

        # divide by the total over all counts for each slv bin to get a probability distribution over output r/a count bins
        # an slv bin with no data sums to 0, so clamp its total to normalize it to zeros instead of 0/0
        totals_slv11 = torch.sum(expected_slvyz, dim=(-2, -1), keepdim=True)
        normalized_slvyz = expected_slvyz / totals_slv11.clamp_min(torch.finfo(expected_slvyz.dtype).tiny)

        # we want downsampled counts to be as even as possible among all ref and alt count bins.  This is equivalent
        # to minimizing the sum of squared downsampled counts.  Since the downsampling
        # preserves total probability, it can't minimize this loss by scaling down the result.  It can only minimize
        # it by redistributing.
        sums_of_squares_slv = torch.sum(torch.square(normalized_slvyz), dim=(-2, -1))
        return torch.sum(sums_of_squares_slv)

    def optimize_downsampling_balance(self, count_totals_slvcd: Tensor) -> int:
        """
        :param count_totals_slvcd: numbers of data by source, label, variant type, ref count and alt count, as
            ReadsDataset counts them
        :return: the number of steps until the fit converged
        """
        # no weight decay: the fit is to known totals, so there is nothing to overfit
        optimizer = torch.optim.Adam(self.parameters(), lr=FITTING_LEARNING_RATE)
        previous_loss = None
        for step in range(MAX_FITTING_STEPS):
            loss = self.balance_loss(count_totals_slvcd)
            # checking only every so many steps keeps the CPU from waiting for the GPU at every step
            if step % FITTING_STEPS_PER_CHECK == 0:
                current_loss = loss.item()
                if previous_loss is not None and previous_loss - current_loss <= FITTING_TOLERANCE * previous_loss:
                    return step
                previous_loss = current_loss
            backpropagate(optimizer, loss)
        return MAX_FITTING_STEPS
