import argparse
import math
from collections import Counter

import cyvcf2
import pysam

from permutect import constants
from permutect.misc_utils import VcfEditor
from permutect.misc_utils import declare_format_dp
from permutect.misc_utils import tumor_depths
from permutect.utils.sequence_utils import StemLoop
from permutect.utils.sequence_utils import find_near_stem_loop

HAIRPIN_FILTER = "hairpin"
STEM_LOOP_FREE_ENERGY_INFO_KEY = "SLFE"
STEM_LOOP_DISTANCE_INFO_KEY = "SLDIST"

# FilterMutectCalls' multiallelic filter, but only for indels.  It isn't called "multiallelic" so that filter_variants
# can trust it without trusting FilterMutectCalls' multiallelic filter, which also filters SNVs.
MULTIALLELIC_INDEL_FILTER = "multiallelic_indel"
TUMOR_LOG10_ODDS_INFO_KEY = "TLOD"
# As in FilterMutectCalls, a site is multiallelic if more than one alt allele has a tumor log odds above 5.
# FilterMutectCalls converts Mutect2's TLOD, a log10 odds, to a natural log before comparing it with 5.
MULTIALLELIC_MIN_TUMOR_LOG10_ODDS = 5 / math.log(10)
MULTIALLELIC_MAX_ALT_ALLELES = 1


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
    parser.add_argument(
        "--" + constants.NO_HAIRPIN_FILTER_NAME,
        action="store_true",
        help="don't apply the hairpin filter.  The multiallelic indel filter applies regardless.",
    )
    return parser.parse_args(argv)


def main_without_parsing(args):
    apply_pre_permutect_filters(
        input_vcf=getattr(args, constants.INPUT_NAME),
        ref_fasta=getattr(args, constants.REFERENCE_NAME),
        output_vcf=getattr(args, constants.OUTPUT_NAME),
        stem_loop_max_energy=getattr(args, constants.STEM_LOOP_MAX_ENERGY_NAME),
        hairpin_filter=not getattr(args, constants.NO_HAIRPIN_FILTER_NAME),
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


def find_multiallelic_sites(input_vcf) -> set[tuple[str, int]]:
    """
    The (contig, position) of each site with more than MULTIALLELIC_MAX_ALT_ALLELES alt alleles whose tumor log odds
    exceed MULTIALLELIC_MIN_TUMOR_LOG10_ODDS.  The alt alleles of all records at a position count, so that a site is
    multiallelic whether or not its alleles have been split into biallelic records, as the SplitMultiallelics task of
    permutect.wdl does.
    """
    num_passing_alts = Counter()
    vcf = cyvcf2.VCF(input_vcf)
    declare_format_dp(vcf)
    for v in vcf:
        tumor_log10_odds = v.INFO.get(TUMOR_LOG10_ODDS_INFO_KEY)
        if tumor_log10_odds is None:
            continue
        # cyvcf2 gives a single value for a biallelic record and a tuple otherwise
        tumor_log10_odds = tumor_log10_odds if isinstance(tumor_log10_odds, tuple) else (tumor_log10_odds,)
        num_passing = sum(odds > MULTIALLELIC_MIN_TUMOR_LOG10_ODDS for odds in tumor_log10_odds)
        if num_passing > 0:
            num_passing_alts[(v.CHROM, v.POS)] += num_passing
    vcf.close()
    return {site for site, num_passing in num_passing_alts.items() if num_passing > MULTIALLELIC_MAX_ALT_ALLELES}


def has_indel_allele(v: cyvcf2.Variant) -> bool:
    return any(len(alt) != len(v.REF) for alt in v.ALT)


def add_filter(v: cyvcf2.Variant, filter_name: str):
    v.FILTER = filter_name if v.FILTER is None else v.FILTER + ";" + filter_name


def apply_pre_permutect_filters(
    input_vcf, ref_fasta, output_vcf, stem_loop_max_energy: float = 1.5, hairpin_filter: bool = True
):
    multiallelic_sites = find_multiallelic_sites(input_vcf)
    reference = pysam.FastaFile(ref_fasta)
    info_fields = {
        STEM_LOOP_FREE_ENERGY_INFO_KEY: ("Float", 1, "Total free energy of the selected nearby stem loop"),
        STEM_LOOP_DISTANCE_INFO_KEY: ("Integer", 1, "Distance in bases from variant start to the selected stem"),
    }
    filter_descriptions = {
        HAIRPIN_FILTER: "near low-alt-fraction stem-loop",
        MULTIALLELIC_INDEL_FILTER: "indel at a site with more than one alt allele, as for Mutect2's multiallelic filter",
    }
    editor = VcfEditor(input_vcf, output_vcf, info_fields, filter_descriptions)

    for v in editor:
        if hairpin_filter:
            hairpin = find_hairpin(v, editor.tumor_sample_index, reference, stem_loop_max_energy)
            if hairpin is not None:
                stem_loop, stem_distance = hairpin
                add_filter(v, HAIRPIN_FILTER)
                v.INFO[STEM_LOOP_FREE_ENERGY_INFO_KEY] = stem_loop.total_energy
                v.INFO[STEM_LOOP_DISTANCE_INFO_KEY] = stem_distance
        if (v.CHROM, v.POS) in multiallelic_sites and has_indel_allele(v):
            add_filter(v, MULTIALLELIC_INDEL_FILTER)
    reference.close()


def main():
    args = parse_arguments()
    main_without_parsing(args)


if __name__ == "__main__":
    main()
