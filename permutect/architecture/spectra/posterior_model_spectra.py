import math

import torch
from torch import Tensor
from torch import nn

from permutect import constants
from permutect.architecture.posterior_model_priors import get_ref_contexts_and_alt_bases
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

# The weight of damage, against Mutect2's sequencing error, in the SEQ_ERROR likelihood of an SNV whose alt reads'
# orientations are known, given a read orientation model: a site without a variant has its alt reads from one mechanism
# or the other, a priori equally likely.
SEQ_ERROR_DAMAGE_WEIGHT = 0.5


def has_orientation_counts_b(batch: Batch) -> Tensor:
    """
    whether each variant has the tumor's counts of alt and of ref reads in F1R2 and F2R1 orientation, which
    filter_variants sets
    """
    fields = (Data.ALT_F1R2_COUNT, Data.ALT_F2R1_COUNT, Data.REF_F1R2_COUNT, Data.REF_F2R1_COUNT)
    return ~torch.stack([torch.isnan(batch.get(field)) for field in fields]).any(dim=0)


class PosteriorModelSpectra(nn.Module):
    """
    Encapsulates all spectra of the posterior model:
        1) somatic variant spectrum
        2) artifact spectra
        3) normal artifact spectrum
        4) germline and contamination likelihoods, of the germline model, which are fixed
        5) damage's likelihoods, of the read orientation model, which are fixed and part of SEQ_ERROR's
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
        # without one, SEQ_ERROR's likelihood is Mutect2's alone
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
        spectra_log_lks_bc[:, Call.SEQ_ERROR] = self.seq_error_log_likelihoods_b(batch)
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
        # as for sequencing error, the normal's alt reads are sequencing errors.  Slippage can happen in the normal too,
        # which a joint model of the tumor and normal could use.
        normal_log_lks_bc[:, Call.SLIPPAGE] = normal_seq_error_log_lks
        normal_log_lks_bc[:, Call.NORMAL_ARTIFACT] = torch.where(
            normal_alt_counts_b < 1, constants.IMPOSSIBLE_LOG_PROB, na_normal_log_lks_b
        )

        return spectra_log_lks_bc, normal_log_lks_bc

    def seq_error_log_likelihoods_b(self, batch: Batch, with_damage: bool = True) -> Tensor:
        """
        The log likelihood of each variant's alt reads given that the site has no variant, SEQ_ERROR.  Without a read
        orientation model, or for a variant that isn't an SNV or lacks the orientation counts of its ref or alt reads,
        it's Mutect2's sequencing error likelihood.  Otherwise it mixes that with the likelihood of the alt reads from
        damage, which puts them in each orientation at the rates of damage to that orientation's strand (see
        permutect.read_orientation.read_orientation_model), weighted by SEQ_ERROR_DAMAGE_WEIGHT.  Damage's likelihood is
        relative to the hypergeometric split of a real variant's alt reads between the orientations, a factor that
        every other call shares and leaves out.  With with_damage False, it's Mutect2's likelihood regardless.

        Mutect2 gives the orientations of only some reads, so damage's likelihood includes the other reads too, as the
        other calls' likelihoods do, each with damage at the rate expected of a read of either orientation.
        """
        seq_error_b = batch.get(Data.SEQ_ERROR_LOG_LK)
        model = self.read_orientation_model
        if model is None or not with_damage:
            return seq_error_b
        alt_f1r2_b, alt_f2r1_b, ref_f1r2_b, ref_f2r1_b = (
            torch.nan_to_num(batch.get(field))
            for field in (Data.ALT_F1R2_COUNT, Data.ALT_F2R1_COUNT, Data.REF_F1R2_COUNT, Data.REF_F2R1_COUNT)
        )
        damage_b, snvs_b = model.damage_log_likelihoods_b(
            *get_ref_contexts_and_alt_bases(batch),
            alt_f1r2_b,
            alt_f2r1_b,
            ref_f1r2_b + alt_f1r2_b,
            ref_f2r1_b + alt_f2r1_b,
            batch.get(Data.ORIGINAL_DEPTH),
            batch.get(Data.ORIGINAL_ALT_COUNT),
        )
        mixed_b = torch.logaddexp(
            seq_error_b + math.log(1 - SEQ_ERROR_DAMAGE_WEIGHT),
            damage_b.to(seq_error_b.dtype) + math.log(SEQ_ERROR_DAMAGE_WEIGHT),
        )
        is_snv_b = batch.get(Data.VARIANT_TYPE) == Variation.SNV
        return torch.where(is_snv_b & snvs_b & has_orientation_counts_b(batch), mixed_b, seq_error_b)
