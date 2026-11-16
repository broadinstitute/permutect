import argparse
import io
import tarfile
import time
from urllib.parse import quote_plus

from permutect import constants
from permutect.read_orientation.f1r2_counts_reader import read_f1r2_counts
from permutect.read_orientation.read_orientation_model import STATES
from permutect.read_orientation.read_orientation_model import ContextPrior
from permutect.read_orientation.read_orientation_model import learn_priors
from permutect.read_orientation.read_orientation_model import reverse_complement

# GATK's artifact prior tables, which FilterMutectCalls reads with --orientation-bias-artifact-priors
GATK_PRIORS_EXTENSION = ".orientation_priors"
GATK_PRIORS_COLUMNS = ("context", "rev_comp", *map(str.lower, STATES), "num_examples", "num_alt_examples")
GATK_SAMPLE_METADATA = "#<METADATA>SAMPLE="


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(
        description="Learn the prior probabilities of GATK's read orientation artifact model, as GATK "
        "LearnReadOrientationModel does, from the F1R2 counts of Mutect2 --f1r2-tar-gz or CollectF1R2Counts, eg those "
        "of each scatter of a run.  For each sample, each 3-base reference context and its reverse complement get a "
        "prior over 8 artifact states, F1R2 or F2R1 for each alt base, and 4 non-artifact states, learned by "
        "expectation maximization.  The output is a tar.gz of GATK's prior tables, one per sample, for "
        "FilterMutectCalls --orientation-bias-artifact-priors."
    )
    parser.add_argument(
        "--" + constants.INPUT_NAME,
        nargs="+",
        required=True,
        help="one or more tar.gz files of F1R2 counts, eg from the scatters of a Mutect2 run, which are summed",
    )
    parser.add_argument("--" + constants.OUTPUT_NAME, required=True, help="path to the output tar.gz of prior tables")
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
    learn_read_orientation_model(
        input_tar_gzs=getattr(args, constants.INPUT_NAME),
        output_tar_gz=getattr(args, constants.OUTPUT_NAME),
        convergence_threshold=getattr(args, constants.CONVERGENCE_THRESHOLD_NAME),
        max_em_iterations=getattr(args, constants.MAX_EM_ITERATIONS_NAME),
    )


def learn_read_orientation_model(
    input_tar_gzs, output_tar_gz, convergence_threshold: float = 1e-4, max_em_iterations: int = 20
):
    if not output_tar_gz.endswith(".tar.gz"):
        raise ValueError(f"The output {output_tar_gz} must end in .tar.gz")
    counts_by_sample = read_f1r2_counts(input_tar_gzs)
    priors_by_sample = {
        sample: learn_priors(counts, convergence_threshold, max_em_iterations)
        for sample, counts in counts_by_sample.items()
    }
    write_gatk_priors(output_tar_gz, priors_by_sample)


def write_gatk_priors(output_tar_gz, priors_by_sample: dict[str, dict[str, ContextPrior]]):
    """
    Write a tar.gz of GATK's prior tables, one per sample, named by the sample URL-encoded as Java's URLEncoder does,
    which leaves letters, digits and .-*_ as they are and turns spaces into +
    """
    with tarfile.open(output_tar_gz, mode="w:gz") as tar:
        for sample, priors in priors_by_sample.items():
            table = gatk_prior_table(sample, priors).encode()
            member = tarfile.TarInfo(quote_plus(sample, safe="*").replace("~", "%7E") + GATK_PRIORS_EXTENSION)
            member.size, member.mtime = len(table), int(time.time())
            tar.addfile(member, io.BytesIO(table))


def gatk_prior_table(sample: str, priors: dict[str, ContextPrior]) -> str:
    """GATK's table of a sample's priors, with a row for each context in lexicographic order"""
    rows = (
        (context, reverse_complement(context), *prior.prior_k.tolist(), prior.num_sites, prior.num_alt_sites)
        for context, prior in sorted(priors.items())
    )
    lines = [GATK_SAMPLE_METADATA + sample, "\t".join(GATK_PRIORS_COLUMNS), *("\t".join(map(str, row)) for row in rows)]
    return "\n".join(lines) + "\n"


def main():
    args = parse_arguments()
    main_without_parsing(args)


if __name__ == "__main__":
    main()
