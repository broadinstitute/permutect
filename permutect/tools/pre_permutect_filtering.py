import argparse

import cyvcf2
import pysam

from permutect import constants
from permutect.misc_utils import VcfEditor
from permutect.misc_utils import tumor_depths
from permutect.utils.sequence_utils import StemLoop
from permutect.utils.sequence_utils import find_near_stem_loop

HAIRPIN_FILTER = "hairpin"
STEM_LOOP_FREE_ENERGY_INFO_KEY = "SLFE"
STEM_LOOP_DISTANCE_INFO_KEY = "SLDIST"


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(
        description="Apply filters to a Mutect2 VCF before filter_variants, which trusts these filters and excludes "
        "the variants they filter from its posterior model."
    )
    parser.add_argument("--" + constants.INPUT_NAME, required=True, help="unfiltered input Mutect2 VCF")
    parser.add_argument(
        "--" + constants.REFERENCE_NAME,
        required=True,
        help="reference genome .fasta file, accompanied by a .fai index",
    )
    parser.add_argument("--" + constants.OUTPUT_NAME, required=True, help="path to output VCF")
    parser.add_argument(
        "--stem-loop-max-energy",
        dest=constants.STEM_LOOP_MAX_ENERGY_NAME,
        type=float,
        default=1.5,
        help="maximum total stem-loop energy for applying the hairpin filter (strictly less than this value)",
    )
    return parser.parse_args(argv)


def main_without_parsing(args):
    apply_pre_permutect_filters(
        input_vcf=getattr(args, constants.INPUT_NAME),
        ref_fasta=getattr(args, constants.REFERENCE_NAME),
        output_vcf=getattr(args, constants.OUTPUT_NAME),
        stem_loop_max_energy=getattr(args, constants.STEM_LOOP_MAX_ENERGY_NAME),
    )


def find_hairpin(
    v: cyvcf2.Variant, tumor_sample_index: int, reference: pysam.FastaFile, stem_loop_max_energy: float
) -> tuple[StemLoop, int] | None:
    """
    (stem loop, distance to stem) if the variant has a low allele fraction and is near a stem loop, otherwise None
    """
    ref_depth, total_alt_depth = tumor_depths(v, tumor_sample_index)
    total_depth = ref_depth + total_alt_depth
    # check the allele fraction first because folding the reference sequence is much more expensive
    if total_depth == 0 or total_alt_depth / total_depth >= 0.1:
        return None
    return find_near_stem_loop(reference, v.CHROM, v.start, max_energy=stem_loop_max_energy)


def apply_pre_permutect_filters(input_vcf, ref_fasta, output_vcf, stem_loop_max_energy: float = 1.5):
    reference = pysam.FastaFile(ref_fasta)
    info_fields = {
        STEM_LOOP_FREE_ENERGY_INFO_KEY: ("Float", 1, "Total free energy of the selected nearby stem loop"),
        STEM_LOOP_DISTANCE_INFO_KEY: ("Integer", 1, "Distance in bases from variant start to the selected stem"),
    }
    filter_descriptions = {HAIRPIN_FILTER: "near low-alt-fraction stem-loop"}
    editor = VcfEditor(input_vcf, output_vcf, info_fields, filter_descriptions)

    for v in editor:
        hairpin = find_hairpin(v, editor.tumor_sample_index, reference, stem_loop_max_energy)
        if hairpin is not None:
            stem_loop, stem_distance = hairpin
            v.FILTER = HAIRPIN_FILTER if v.FILTER is None else v.FILTER + ";" + HAIRPIN_FILTER
            v.INFO[STEM_LOOP_FREE_ENERGY_INFO_KEY] = stem_loop.total_energy
            v.INFO[STEM_LOOP_DISTANCE_INFO_KEY] = stem_distance
    reference.close()


def main():
    args = parse_arguments()
    main_without_parsing(args)


if __name__ == "__main__":
    main()
