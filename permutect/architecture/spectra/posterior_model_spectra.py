import torch
from torch import nn

from permutect import constants
from permutect.architecture.posterior_model_priors import get_ref_contexts_and_alt_bases
from permutect.architecture.posterior_model_priors import has_alt_orientation_counts_b
from permutect.architecture.posterior_model_priors import possible_slippage_b
from permutect.architecture.spectra.artifact_spectra import ArtifactSpectra
from permutect.architecture.spectra.normal_artifact_spectrum import NormalArtifactSpectrum
from permutect.architecture.spectra.somatic_spectrum import SomaticSpectrum
from permutect.data.batch import Batch
from permutect.data.datum import Data
from permutect.misc_utils import float_dtype
from permutect.misc_utils import resolve_device
from permutect.read_orientation.read_orientation_model import ReadOrientationModel
from permutect.segmentation.germline_model import GermlineModel
from permutect.slippage.slippage_model import SlippageModel
from permutect.utils.enums import INDEL_VARIATIONS
from permutect.utils.enums import Call
from permutect.utils.enums import Variation


class PosteriorModelSpectra(nn.Module):
    """
    Encapsulates all spectra of the posterior model:
        1) somatic variant spectrum
        2) artifact spectra
        3) normal artifact spectrum
        4) germline and contamination likelihoods, of the germline model, which are fixed
        5) read orientation artifact likelihoods, of the read orientation model, which are fixed
        6) slippage likelihoods, of the slippage model, which are fixed
    Basically, anything that computes the likelihood of alt and ref read counts goes here.
    """

    def __init__(
        self,
        germline_model: GermlineModel = None,
        device=None,
        read_orientation_model: ReadOrientationModel = None,
        slippage_model: SlippageModel = None,
    ):
        super(PosteriorModelSpectra, self).__init__()

        self._device = resolve_device(device)
        self._dtype = float_dtype(self._device)
        # without segments, and with default parameters, if not given.  Its parameters are frozen.
        self.germline_model = GermlineModel() if germline_model is None else germline_model
        # without one, read orientation artifacts are impossible, as their priors make them
        self.read_orientation_model = read_orientation_model
        self.slippage_model = SlippageModel() if slippage_model is None else slippage_model

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
        # here, and the normal's germline column below stays 0.  Indels have their own parameters, given their STRs'
        # repeat unit lengths and lengths, which are 0 outside STRs and NaN for data that filter_variants hasn't annotated.
        indels_b = torch.isin(var_types_b, torch.tensor(INDEL_VARIATIONS, device=var_types_b.device))
        repeat_unit_lengths_b, repeat_lengths_b = (
            torch.nan_to_num(batch.get(field), nan=0.0) for field in (Data.REPEAT_UNIT_LENGTH, Data.REPEAT_LENGTH)
        )
        spectra_log_lks_bc[:, Call.GERMLINE] = self.germline_model.germline_log_likelihoods(
            alt_counts_b,
            depths_b,
            normal_alt_counts_b,
            normal_depths_b,
            log_afs_b,
            mafs_b,
            batch.get(Data.NORMAL_MAF),
            indels=indels_b,
            repeat_unit_lengths=repeat_unit_lengths_b,
            repeat_lengths=repeat_lengths_b,
        )
        spectra_log_lks_bc[:, Call.CONTAMINATION] = self.germline_model.tumor.contamination_log_likelihoods(
            alt_counts_b, depths_b, log_afs_b
        )
        spectra_log_lks_bc[:, Call.READ_ORIENTATION] = self.read_orientation_log_likelihoods_b(batch)
        # 0 where the prior makes slippage impossible
        spectra_log_lks_bc[:, Call.SLIPPAGE] = torch.where(
            possible_slippage_b(self.slippage_model, batch),
            self.slippage_model.tumor_log_likelihoods_b(depths_b, alt_counts_b),
            0,
        )

        normal_seq_error_log_lks = batch.get(Data.NORMAL_SEQ_ERROR_LOG_LK)
        normal_log_lks_bc = torch.zeros_like(spectra_log_lks_bc)
        normal_log_lks_bc[:, Call.SOMATIC] = normal_seq_error_log_lks
        normal_log_lks_bc[:, Call.ARTIFACT] = normal_seq_error_log_lks
        normal_log_lks_bc[:, Call.SEQ_ERROR] = normal_seq_error_log_lks
        # the contaminant's reads are in the tumor alone
        normal_log_lks_bc[:, Call.CONTAMINATION] = normal_seq_error_log_lks
        # read orientation artifacts come from damage to the tumor's DNA
        normal_log_lks_bc[:, Call.READ_ORIENTATION] = normal_seq_error_log_lks
        # as for sequencing error, the normal's alt reads are sequencing errors.  Slippage can happen in the normal too,
        # which a joint model of the tumor and normal could use.
        normal_log_lks_bc[:, Call.SLIPPAGE] = normal_seq_error_log_lks
        normal_log_lks_bc[:, Call.NORMAL_ARTIFACT] = torch.where(
            normal_alt_counts_b < 1, constants.IMPOSSIBLE_LOG_PROB, na_normal_log_lks_b
        )

        return spectra_log_lks_bc, normal_log_lks_bc

    def read_orientation_log_likelihoods_b(self, batch: Batch) -> torch.Tensor:
        """
        The log likelihood of each variant's alt count and alt F1R2 count given a read orientation artifact, relative to
        the likelihood of its alt F1R2 count given a non-artifact, a factor that every other call shares and leaves out.
        A read orientation artifact is a mixture of an F1R2 and an F2R1 artifact of the alt base, weighted by their
        priors.  It's 0 where the priors make the artifact impossible: without a read orientation model, and for a
        variant that isn't an SNV or lacks the alt orientation counts.
        """
        depths_b, alt_counts_b = batch.get(Data.ORIGINAL_DEPTH), batch.get(Data.ORIGINAL_ALT_COUNT)
        model = self.read_orientation_model
        if model is None:
            return torch.zeros_like(depths_b, dtype=self._dtype)
        left_b, ref_b, right_b, alt_b = get_ref_contexts_and_alt_bases(batch)
        # the F1R2 likelihoods' counts are those of reads with an orientation, as in GATK's ReadOrientationFilter
        alt_f1r2_counts_b, alt_f2r1_counts_b = (
            torch.nan_to_num(batch.get(field)) for field in (Data.ALT_F1R2_COUNT, Data.ALT_F2R1_COUNT)
        )
        oriented_alt_counts_b = alt_f1r2_counts_b + alt_f2r1_counts_b

        log_priors_bo = model.log_priors_bo(left_b, ref_b, right_b, alt_b)
        artifact_log_lks_bo = model.artifact_spectrum_log_likelihoods_bo(alt_b, depths_b, alt_counts_b)
        artifact_log_lks_bo += model.artifact_f1r2_log_likelihoods_bo(alt_b, oriented_alt_counts_b, alt_f1r2_counts_b)
        log_prior_b = torch.logsumexp(log_priors_bo, dim=-1)
        log_lks_b = torch.logsumexp(log_priors_bo + artifact_log_lks_bo, dim=-1) - log_prior_b
        log_lks_b -= model.non_artifact_f1r2_log_likelihoods_b(oriented_alt_counts_b, alt_f1r2_counts_b)

        is_snv_b = batch.get(Data.VARIANT_TYPE) == Variation.SNV
        possible_b = is_snv_b & has_alt_orientation_counts_b(batch) & torch.isfinite(log_prior_b)
        return torch.where(possible_b, log_lks_b, 0)
