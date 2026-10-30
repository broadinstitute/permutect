import torch
from torch import nn

from permutect import constants
from permutect.architecture.spectra.artifact_spectra import ArtifactSpectra
from permutect.architecture.spectra.normal_artifact_spectrum import NormalArtifactSpectrum
from permutect.architecture.spectra.somatic_spectrum import SomaticSpectrum
from permutect.data.batch import Batch
from permutect.data.datum import Data
from permutect.misc_utils import float_dtype
from permutect.misc_utils import resolve_device
from permutect.utils.enums import Call
from permutect.utils.math_utils import log1mexp
from permutect.utils.stats_utils import beta_binomial_log_lk
from permutect.utils.stats_utils import binomial_log_lk


# given germline, the probability of these particular reads being alt
# log_afs are natural logs of population allele frequencies
def germline_log_likelihood(log_afs, mafs, alt_counts, depths, het_beta=None):
    # het and hom alt probabilities are 2 AF (1 - AF) and AF^2, which sum to AF (2 - AF).  Thus, conditioned on
    # germline, the het proportion divided by 2 is (1 - AF) / (2 - AF) and the hom proportion is AF / (2 - AF).
    log_2m_afs = torch.log(2 - torch.exp(log_afs))
    log_half_het_prop = log1mexp(log_afs) - log_2m_afs
    log_hom_prop = log_afs - log_2m_afs

    # the following should all be 1D tensors of length batch size
    if het_beta is None:
        alt_minor_ll = log_half_het_prop + binomial_log_lk(depths, alt_counts, mafs)
        alt_major_ll = log_half_het_prop + binomial_log_lk(depths, alt_counts, 1 - mafs)
    else:
        het_shape = torch.tensor([het_beta], device=depths.device)
        alt_minor_ll = alt_major_ll = log_half_het_prop + beta_binomial_log_lk(depths, alt_counts, het_shape, het_shape)
    hom_alpha, hom_beta = torch.tensor([98.0], device=depths.device), torch.tensor([2.0], device=depths.device)
    hom_ll = log_hom_prop + beta_binomial_log_lk(depths, alt_counts, hom_alpha, hom_beta)

    return torch.logsumexp(torch.vstack((alt_minor_ll, alt_major_ll, hom_ll)), dim=0)


class PosteriorModelSpectra(nn.Module):
    """
    Encapsulates all spectra of the posterior model:
        1) somatic variant spectrum
        2) artifact spectra
        3) normal artifact spectrum
        4) germline likelihoods
    Basically, anything that computes the likelihood of alt and ref read counts goes here.
    """

    def __init__(self, device=None, het_beta: float = None):
        super(PosteriorModelSpectra, self).__init__()

        self._device = resolve_device(device)
        self._dtype = float_dtype(self._device)
        self.het_beta = het_beta

        # TODO introduce parameters class so that num_components is not hard-coded
        self.somatic_spectrum = SomaticSpectrum(num_components=5)
        self.artifact_spectra = ArtifactSpectra()
        self.normal_artifact_spectra = NormalArtifactSpectrum()

    def spectra_log_likelihoods_bc(self, batch: Batch) -> tuple[torch.Tensor, torch.Tensor]:
        """
        'bc' indexing denotes by datum within batch, then by Call type
        """
        var_types_b, log_afs_b, mafs_b = (
            batch.get(Data.VARIANT_TYPE),
            batch.get(Data.LOG_ALLELE_FREQUENCY),
            batch.get(Data.MAF),
        )
        depths_b, alt_counts_b = batch.get(Data.ORIGINAL_DEPTH), batch.get(Data.ORIGINAL_ALT_COUNT)
        normal_depths_b, normal_alt_counts_b = (
            batch.get(Data.ORIGINAL_NORMAL_DEPTH),
            batch.get(Data.ORIGINAL_NORMAL_ALT_COUNT),
        )

        na_tumor_log_lks_b, na_normal_log_lks_b = self.normal_artifact_spectra.forward(
            var_types_b=var_types_b,
            tumor_alt_counts_b=alt_counts_b,
            tumor_depths_b=depths_b,
            normal_alt_counts_b=normal_alt_counts_b,
            normal_depths_b=normal_depths_b,
        )

        spectra_log_lks_bc = torch.zeros((batch.size(), len(Call)), device=self._device, dtype=self._dtype)
        spectra_log_lks_bc[:, Call.SOMATIC] = self.somatic_spectrum.forward(depths_b, alt_counts_b, mafs_b)
        spectra_log_lks_bc[:, Call.ARTIFACT] = self.artifact_spectra.forward(var_types_b, depths_b, alt_counts_b)
        spectra_log_lks_bc[:, Call.NORMAL_ARTIFACT] = na_tumor_log_lks_b
        spectra_log_lks_bc[:, Call.SEQ_ERROR] = batch.get(Data.SEQ_ERROR_LOG_LK)
        spectra_log_lks_bc[:, Call.GERMLINE] = germline_log_likelihood(
            log_afs_b, mafs_b, alt_counts_b, depths_b, self.het_beta
        )

        normal_seq_error_log_lks = batch.get(Data.NORMAL_SEQ_ERROR_LOG_LK)
        normal_log_lks_bc = torch.zeros_like(spectra_log_lks_bc)
        normal_log_lks_bc[:, Call.SOMATIC] = normal_seq_error_log_lks
        normal_log_lks_bc[:, Call.ARTIFACT] = normal_seq_error_log_lks
        normal_log_lks_bc[:, Call.SEQ_ERROR] = normal_seq_error_log_lks
        normal_log_lks_bc[:, Call.NORMAL_ARTIFACT] = torch.where(
            normal_alt_counts_b < 1, constants.IMPOSSIBLE_LOG_PROB, na_normal_log_lks_b
        )
        normal_log_lks_bc[:, Call.GERMLINE] = germline_log_likelihood(
            log_afs_b, batch.get(Data.NORMAL_MAF), normal_alt_counts_b, normal_depths_b, self.het_beta
        )

        return spectra_log_lks_bc, normal_log_lks_bc
