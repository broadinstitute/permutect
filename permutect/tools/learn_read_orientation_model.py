import argparse

from permutect import constants
from permutect.read_orientation.capture_bias import fit_capture_bias
from permutect.read_orientation.f1r2_counts_reader import read_f1r2_counts
from permutect.read_orientation.read_orientation_model import DEFAULT_CAPTURE_PSEUDOCOUNTS
from permutect.read_orientation.read_orientation_model import fit_read_orientation_model
from permutect.segmentation.germline_model import GermlineModel


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(
        description="Learn a sample's read orientation artifact model, which plays the role of GATK "
        "LearnReadOrientationModel's, from its F1R2 counts from Mutect2 --f1r2-tar-gz or CollectF1R2Counts, eg those of "
        "each scatter of a run.  The model is generative: hybrid capture splits a site's reads between F1R2 and F2R1, "
        "from the + and - strands of the original DNA, by the sample's capture bias, a beta-binomial fit to the reads "
        "of the alt tables' sites; a real variant's alt reads are in both orientations in proportion; and every strand "
        "of every site is damaged at some rate, which puts alt reads in its own orientation.  Each of the 192 physical "
        "damage processes, the damaged strand's 3-base context and the base it comes to read, gets a distribution of "
        "its rate across sites, with a shared hierarchical prior, and each pair of a context and its reverse complement "
        "a prior of the genotypes, learned by expectation maximization.  The output is a ReadOrientationModel for "
        "filter_variants --read_orientation_model, whose damage is part of the likelihood of a sequencing error."
    )
    parser.add_argument(
        "--" + constants.INPUT_NAME,
        nargs="+",
        required=True,
        help="one or more tar.gz files of the F1R2 counts of one sample, eg from the scatters of a Mutect2 run, which "
        "are summed",
    )
    parser.add_argument("--" + constants.OUTPUT_NAME, required=True, help="path to the output ReadOrientationModel")
    parser.add_argument(
        "--" + constants.GERMLINE_MODEL_NAME,
        help="optional GermlineModel from segment_pileups of the same sample, whose segmentation of the sample by "
        "minor allele fraction and emission model give the likelihoods of germline hets, accounting for hets in regions "
        "of loss of heterozygosity.  Without it, germline hets have GATK's likelihoods.",
    )
    parser.add_argument(
        "--" + constants.CONVERGENCE_THRESHOLD_NAME,
        type=float,
        default=1e-4,
        help="EM stops when the prior changes by at most this much (Euclidean distance) in an iteration",
    )
    parser.add_argument(
        "--" + constants.MAX_EM_ITERATIONS_NAME,
        type=int,
        default=20,
        help="maximum number of iterations of expectation maximization",
    )
    return parser.parse_args(argv)


def main_without_parsing(args):
    germline_model_path = getattr(args, constants.GERMLINE_MODEL_NAME)
    learn_read_orientation_model(
        input_tar_gzs=getattr(args, constants.INPUT_NAME),
        output_path=getattr(args, constants.OUTPUT_NAME),
        convergence_threshold=getattr(args, constants.CONVERGENCE_THRESHOLD_NAME),
        max_em_iterations=getattr(args, constants.MAX_EM_ITERATIONS_NAME),
        germline_model=None if germline_model_path is None else GermlineModel.load(germline_model_path),
    )


def learn_read_orientation_model(
    input_tar_gzs,
    output_path,
    convergence_threshold: float = 1e-4,
    max_em_iterations: int = 20,
    germline_model: GermlineModel = None,
):
    """
    The read orientation model of the sample of the F1R2 counts in input_tar_gzs, saved to output_path.  Given
    germline_model, segment_pileups' model of the counts' sample, germline hets have the likelihoods of its emission
    model at the minor allele fractions of its segmentation.
    """
    counts_by_sample = read_f1r2_counts(input_tar_gzs, None if germline_model is None else germline_model.tumor)
    # Mutect2 collects the F1R2 counts of tumor samples, and Permutect calls one tumor
    if len(counts_by_sample) != 1:
        raise ValueError(
            f"A read orientation model is of one sample, but the F1R2 counts are of {len(counts_by_sample)} samples: "
            + ", ".join(sorted(counts_by_sample))
        )
    ((sample, counts),) = counts_by_sample.items()
    if germline_model is not None and germline_model.tumor.sample != sample:
        raise ValueError(
            f"The germline model is of sample {germline_model.tumor.sample}, but the F1R2 counts are of sample {sample}"
        )
    emission_model = None if germline_model is None else germline_model.tumor.emission_model

    capture = fit_capture_bias(counts.capture_counts)
    if capture is None:
        print(f"No alt table sites to fit the capture bias to, so it is Beta{DEFAULT_CAPTURE_PSEUDOCOUNTS}")
        capture_pseudocounts = DEFAULT_CAPTURE_PSEUDOCOUNTS
    else:
        print(
            f"Capture bias, from the reads of {counts.capture_counts.num_sites()} sites: F1R2 fraction "
            f"{capture.f1r2_fraction.item():.3f}, concentration {capture.concentration.item():.1f}"
        )
        capture_pseudocounts = capture.pseudocounts()

    model = fit_read_orientation_model(
        counts, capture_pseudocounts, emission_model, convergence_threshold, max_em_iterations
    )
    model.save(output_path)


def main():
    args = parse_arguments()
    main_without_parsing(args)


if __name__ == "__main__":
    main()
