import json
import os
import subprocess
import tarfile

import cyvcf2
import pytest

from permutect.test.test_file_names import *
from permutect.test.wdl.test_wdl_scripts import task_output_files
from permutect.test.wdl.test_wdl_scripts import task_output_globs
from permutect.tools.filter_variants import POST_PROB_INFO_KEY
from permutect.tools.pre_permutect_filtering import HAIRPIN_FILTER
from permutect.tools.pre_permutect_filtering import MULTIALLELIC_INDEL_FILTER

PERMUTECT_TEST_DOCKER = os.environ.get("PERMUTECT_TEST_DOCKER", "permutect:wdl-test")
CROMWELL_JAR = os.environ.get("CROMWELL_JAR")
PERMUTECT_WDL = os.path.abspath(os.path.join(test_dir, "..", "..", "scripts", "permutect.wdl"))

pytestmark = pytest.mark.skipif(
    CROMWELL_JAR is None, reason="WDL tests need Cromwell and a Permutect docker image; see `pixi run -e wdl test-wdl`"
)


def make_cached_m2_outputs(
    working_dir, archive_name: str, with_pileups: bool, with_normal: bool, with_f1r2_counts: bool
) -> str:
    """
    An archive of Mutect2 outputs for SEQC2 WES chr20, as made by the ArchiveM2Outputs task, with or without the
    pileups at common variants, which include the normal's only for a matched-normal run, and with or without the F1R2
    counts of the scatters, for which another sample's counts stand in
    """
    m2_outputs = {
        "test_dataset": SEQC2_WES_PLAIN_TEXT_DATA,
        "mutect2_vcf": MUTECT2_SEQC2_WES_VCF,
        "mutect2_vcf_idx": MUTECT2_SEQC2_WES_VCF_INDEX,
        "contigs_table": HG38_CONTIGS_TABLE,
        "mutect_stats": MUTECT2_SEQC2_WES_STATS,
    }
    if with_pileups:
        m2_outputs["tumor_pileups"] = SEQC2_WES_TUMOR_PILEUPS
        if with_normal:
            m2_outputs["normal_pileups"] = SEQC2_WES_NORMAL_PILEUPS

    archive = os.path.join(working_dir, archive_name)
    file_names_in_archive = task_output_files(PERMUTECT_WDL, "ExtractM2Outputs")
    f1r2_directory = os.path.dirname(task_output_globs(PERMUTECT_WDL, "ExtractM2Outputs")["f1r2_counts"])
    with tarfile.open(archive, "w") as tar:
        for name, m2_output in m2_outputs.items():
            tar.add(m2_output, arcname=file_names_in_archive[name])
        for n, scatter in enumerate(DREAM_TUMOR_4_F1R2_SCATTERS if with_f1r2_counts else []):
            tar.add(scatter, arcname=os.path.join(f1r2_directory, f"scatter-{n}.tar.gz"))
    return archive


def run_permutect_wdl(
    working_dir,
    filter_pcr_errors: bool,
    with_pileups: bool,
    tumor_only: bool,
    with_normal: bool,
    learn_read_orientation: bool = False,
) -> dict:
    """
    Run permutect.wdl with Cromwell on SEQC2 WES chr20 data, giving cached Mutect2 outputs so that Mutect2 is skipped.
    With learn_read_orientation, the run has run_orientation_bias_mixture_model_filter and the caches have F1R2 counts.
    Returns the Cromwell workflow metadata.
    """
    tumor_only_mode = tumor_only or not with_normal

    # the workflow requires reads, but only Mutect2 uses them
    placeholder_bam, placeholder_bai = os.path.join(working_dir, "reads.bam"), os.path.join(working_dir, "reads.bai")
    for placeholder in (placeholder_bam, placeholder_bai):
        open(placeholder, "w").close()

    inputs = {
        "Permutect.artifact_model": PERMUTECT_V0_9_2_ARTIFACT_MODEL,
        "Permutect.ref_fasta": HG38_CHR20_REFERENCE,
        "Permutect.ref_fai": HG38_CHR20_REFERENCE_INDEX,
        "Permutect.ref_dict": HG38_CHR20_REFERENCE_DICT,
        "Permutect.primary_bam": placeholder_bam,
        "Permutect.primary_bai": placeholder_bai,
        "Permutect.control_bam": placeholder_bam if with_normal else "",
        "Permutect.control_bai": placeholder_bai if with_normal else "",
        "Permutect.scatter_count": 1,
        "Permutect.batch_size": 64,
        # data loader workers share tensors through a Unix socket in TMPDIR, which Cromwell's Local backend puts too
        # deep in the execution directory for the socket path length limit
        "Permutect.num_workers": 0,
        "Permutect.filter_pcr_errors": filter_pcr_errors,
        "Permutect.run_orientation_bias_mixture_model_filter": learn_read_orientation,
        # a tumor-only run is also given matched-normal cached outputs, which it must ignore
        "Permutect.cached_m2_outputs": make_cached_m2_outputs(
            working_dir,
            "cached_m2_outputs.tar",
            with_pileups,
            with_normal=True,
            with_f1r2_counts=learn_read_orientation,
        ),
        "Permutect.tumor_only": tumor_only,
        "Permutect.permutect_docker": PERMUTECT_TEST_DOCKER,
        # Terra turns unset optional inputs into empty strings, which the workflow must treat as unset
        "Permutect.masks": "",
        "Permutect.test_dataset_truth_vcf": "",
        "Permutect.test_dataset_truth_vcf_idx": "",
        "Permutect.tumor_only_cached_m2_outputs": (
            make_cached_m2_outputs(
                working_dir, "tumor_only_cached_m2_outputs.tar", with_pileups, False, learn_read_orientation
            )
            if tumor_only_mode
            else ""
        ),
    }
    inputs_json, metadata_json = os.path.join(working_dir, "inputs.json"), os.path.join(working_dir, "metadata.json")
    with open(inputs_json, "w") as file:
        json.dump(inputs, file)

    # Cromwell writes its executions and logs into the working directory.  Looking up the docker image's hash in a
    # registry is pointless for a locally built image.
    cromwell_command = ["java", "-Ddocker.hash-lookup.enabled=false", "-jar", os.path.abspath(CROMWELL_JAR), "run"]
    subprocess.run(
        cromwell_command + [PERMUTECT_WDL, "--inputs", inputs_json, "--metadata-output", metadata_json],
        cwd=working_dir,
        timeout=3600,
    )
    assert os.path.exists(metadata_json), "Cromwell did not run the workflow; see its log above"
    with open(metadata_json) as file:
        metadata = json.load(file)
    assert metadata["status"] == "Succeeded", json.dumps(metadata.get("failures"), indent=2)
    return metadata


def check_cached_m2_outputs(
    metadata: dict, with_pileups: bool, tumor_only_mode: bool, learn_read_orientation: bool = False
):
    # the cache for this run's mode is unpacked rather than running Mutect2, and both caches are passed through to the
    # outputs as is
    calls = metadata["calls"]
    assert "Permutect.ExtractM2Outputs" in calls
    assert "Permutect.Mutect2" not in calls and "Permutect.ArchiveM2Outputs" not in calls
    inputs, outputs = metadata["inputs"], metadata["outputs"]
    cache_used = inputs["tumor_only_cached_m2_outputs" if tumor_only_mode else "cached_m2_outputs"]
    assert calls["Permutect.ExtractM2Outputs"][0]["inputs"]["cached_m2_outputs"] == cache_used
    assert outputs["Permutect.m2_outputs"] == inputs["cached_m2_outputs"]
    tumor_only_cache = inputs["tumor_only_cached_m2_outputs"] if tumor_only_mode else None
    assert outputs["Permutect.tumor_only_m2_outputs"] == tumor_only_cache

    # with pileups, segment_pileups segments the samples, the normal only in a matched-normal run, and filter_variants
    # uses the germline model
    assert ("Permutect.SegmentPileups" in calls) == with_pileups
    filtering_inputs = calls["Permutect.PermutectFiltering"][0]["inputs"]
    assert (filtering_inputs["germline_model"] is not None) == with_pileups
    for output in ("germline_model", "tumor_segments", "tumor_segments_plot", "contamination_table"):
        assert (outputs["Permutect." + output] is not None) == with_pileups
    for output in ("normal_segments", "normal_segments_plot"):
        assert (outputs["Permutect." + output] is not None) == (with_pileups and not tumor_only_mode)

    # with run_orientation_bias_mixture_model_filter, learn_read_orientation_model learns the priors of the cached F1R2
    # counts' sample
    assert ("Permutect.LearnReadOrientationModel" in calls) == learn_read_orientation
    priors = outputs["Permutect.read_orientation_priors"]
    assert (priors is not None) == learn_read_orientation
    if learn_read_orientation:
        with tarfile.open(priors, "r:gz") as tar:
            assert tar.getnames() == ["synthetic.challenge.set4.tumour.orientation_priors"]


def output_vcf_records(metadata: dict, tumor_only_mode: bool) -> list[cyvcf2.Variant]:
    # the calls go to the output VCF for the run's mode, and the other mode's output VCF is unset
    outputs = metadata["outputs"]
    vcf, other_vcf = (
        ("tumor_only_output_vcf", "output_vcf") if tumor_only_mode else ("output_vcf", "tumor_only_output_vcf")
    )
    for suffix in ("", "_idx"):
        assert os.path.exists(outputs["Permutect." + vcf + suffix])
        assert outputs["Permutect." + other_vcf + suffix] is None
    assert os.path.exists(outputs["Permutect.tensorboard_report"])
    return list(cyvcf2.VCF(outputs["Permutect." + vcf]))


def has_filter(v: cyvcf2.Variant, filter_name: str) -> bool:
    return v.FILTER is not None and filter_name in v.FILTER.split(";")


@pytest.fixture(scope="module")
def metadata_with_pcr_error_filtering(tmp_path_factory):
    # also learning the read orientation model
    working_dir = tmp_path_factory.mktemp("with_pcr_error_filtering")
    return run_permutect_wdl(
        working_dir,
        filter_pcr_errors=True,
        with_pileups=True,
        tumor_only=False,
        with_normal=True,
        learn_read_orientation=True,
    )


@pytest.fixture(scope="module")
def metadata_without_pcr_error_filtering(tmp_path_factory):
    # with tumor_only overriding the normal, so that only the tumor is segmented
    working_dir = tmp_path_factory.mktemp("without_pcr_error_filtering")
    return run_permutect_wdl(working_dir, filter_pcr_errors=False, with_pileups=True, tumor_only=True, with_normal=True)


@pytest.fixture(scope="module")
def metadata_without_normal(tmp_path_factory):
    # tumor-only because the normal is the empty string that Terra gives for unset inputs, although tumor_only is false.
    # Also without pileups, which Mutect2 makes only with common variants.
    working_dir = tmp_path_factory.mktemp("without_normal")
    return run_permutect_wdl(
        working_dir, filter_pcr_errors=False, with_pileups=False, tumor_only=False, with_normal=False
    )


def test_permutect_wdl_with_pcr_error_filtering(metadata_with_pcr_error_filtering):
    check_cached_m2_outputs(
        metadata_with_pcr_error_filtering, with_pileups=True, tumor_only_mode=False, learn_read_orientation=True
    )
    assert "Permutect.PrePermutectFiltering" in metadata_with_pcr_error_filtering["calls"]
    # the empty truth VCF counts as no truth VCF
    assert "Permutect.PermutectConcordance" not in metadata_with_pcr_error_filtering["calls"]
    records = output_vcf_records(metadata_with_pcr_error_filtering, tumor_only_mode=False)
    assert any(v.INFO.get(POST_PROB_INFO_KEY) is not None for v in records)

    target_records = [v for v in records if v.CHROM == "chr20" and v.POS == 4932703]
    assert len(target_records) == 1
    assert has_filter(target_records[0], HAIRPIN_FILTER)


def test_permutect_wdl_without_pcr_error_filtering(metadata_without_pcr_error_filtering):
    check_cached_m2_outputs(metadata_without_pcr_error_filtering, with_pileups=True, tumor_only_mode=True)
    # pre_permutect_filtering runs regardless, for its multiallelic indel filter, but without its hairpin filter
    assert "Permutect.PrePermutectFiltering" in metadata_without_pcr_error_filtering["calls"]
    records = output_vcf_records(metadata_without_pcr_error_filtering, tumor_only_mode=True)
    assert any(v.INFO.get(POST_PROB_INFO_KEY) is not None for v in records)
    assert not any(has_filter(v, HAIRPIN_FILTER) for v in records)
    assert any(has_filter(v, MULTIALLELIC_INDEL_FILTER) for v in records)


def test_permutect_wdl_without_normal(metadata_without_normal):
    check_cached_m2_outputs(metadata_without_normal, with_pileups=False, tumor_only_mode=True)
    records = output_vcf_records(metadata_without_normal, tumor_only_mode=True)
    assert any(v.INFO.get(POST_PROB_INFO_KEY) is not None for v in records)
