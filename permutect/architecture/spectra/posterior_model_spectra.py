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
from permutect.segmentation.segmentation_model import SegmentationModel
from permutect.utils.enums import Call


class PosteriorModelSpectra(nn.Module):
    """
    Encapsulates all spectra of the posterior model:
        1) somatic variant spectrum
        2) artifact spectra
        3) normal artifact spectrum
        4) germline and contamination likelihoods, of the segmentation model, which are fixed
    Basically, anything that computes the likelihood of alt and ref read counts goes here.
    """

    def __init__(self, segmentation_model: SegmentationModel = None, device=None):
        super(PosteriorModelSpectra, self).__init__()

        self._device = resolve_device(device)
        self._dtype = float_dtype(self._device)
        # without segments, and with default parameters, if not given.  Its parameters are frozen.
        self.segmentation_model = SegmentationModel() if segmentation_model is None else segmentation_model

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
        # the tumor and the normal share a germline genotype, so their germline likelihood is a joint one, which goes
        # here, and the normal's germline column below stays 0
        spectra_log_lks_bc[:, Call.GERMLINE] = self.segmentation_model.germline_log_likelihoods(
            alt_counts_b, depths_b, normal_alt_counts_b, normal_depths_b, log_afs_b, mafs_b, batch.get(Data.NORMAL_MAF)
        )
        spectra_log_lks_bc[:, Call.CONTAMINATION] = self.segmentation_model.tumor.contamination_log_likelihoods(
            alt_counts_b, depths_b, log_afs_b
        )

        normal_seq_error_log_lks = batch.get(Data.NORMAL_SEQ_ERROR_LOG_LK)
        normal_log_lks_bc = torch.zeros_like(spectra_log_lks_bc)
        normal_log_lks_bc[:, Call.SOMATIC] = normal_seq_error_log_lks
        normal_log_lks_bc[:, Call.ARTIFACT] = normal_seq_error_log_lks
        normal_log_lks_bc[:, Call.SEQ_ERROR] = normal_seq_error_log_lks
        # the contaminant's reads are in the tumor alone
        normal_log_lks_bc[:, Call.CONTAMINATION] = normal_seq_error_log_lks
        normal_log_lks_bc[:, Call.NORMAL_ARTIFACT] = torch.where(
            normal_alt_counts_b < 1, constants.IMPOSSIBLE_LOG_PROB, na_normal_log_lks_b
        )

        return spectra_log_lks_bc, normal_log_lks_bc
