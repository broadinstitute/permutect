"""
Fast checks of the WDL scripts that need neither Cromwell nor Docker.
"""

import glob
import os
import re
import shlex
import subprocess
import sys
import tarfile

import pytest

from permutect import constants
from permutect.test.test_file_names import test_dir
from permutect.tools import edit_dataset

SCRIPTS_DIR = os.path.abspath(os.path.join(test_dir, "..", "..", "scripts"))
COMPARE_WDL = os.path.join(SCRIPTS_DIR, "compare_permutect_results.wdl")
CONCATENATE_WDL = os.path.join(SCRIPTS_DIR, "concatenate.wdl")
EDIT_DATASET_WDL = os.path.join(SCRIPTS_DIR, "permutect_edit_dataset.wdl")
PERMUTECT_WDL = os.path.join(SCRIPTS_DIR, "permutect.wdl")

# a stand-in for gatk SelectVariants: copy the -V input to the -O output and write an index
FAKE_GATK = """#!/bin/bash
while [ $# -gt 0 ]; do
    case "$1" in
        -V) input="$2"; shift 2;;
        -O) output="$2"; shift 2;;
        *) shift;;
    esac
done
cp "$input" "$output" && touch "$output.idx"
"""


def test_index_inputs_are_given_index_files():
    # e.g. permutect_ffn_idx = PermutectConcordance.ffn passed the VCF itself as its own index
    bindings = []
    for wdl in glob.glob(os.path.join(SCRIPTS_DIR, "*.wdl")):
        with open(wdl) as file:
            for line_number, line in enumerate(file, start=1):
                # call inputs are bound as 'name = expression'; declarations start with a type instead
                match = re.match(r"\s*(\w+_idx)\s*=\s*(.+?),?\s*$", line)
                if match:
                    bindings.append((os.path.basename(wdl), line_number, match.group(1), match.group(2)))

    assert bindings, "no index-file inputs found; has the WDL syntax changed?"
    wrong = [binding for binding in bindings if "idx" not in binding[3]]
    assert not wrong, f"index inputs bound to something other than an index: {wrong}"


def test_command_placeholders_start_with_a_tilde():
    # a placeholder written {" --source " + new_source} instead of ~{...} reaches bash as literal text
    missing_tilde = []
    for wdl in glob.glob(os.path.join(SCRIPTS_DIR, "*.wdl")):
        with open(wdl) as file:
            text = file.read()
        for command in re.finditer(r"command\s*<<<(.*?)>>>", text, re.S):
            for placeholder in re.finditer(r'(?<![~$])\{\s*(?:"|true=|false=|sep=|default=)', command.group(1)):
                line_number = text.count("\n", 0, command.start(1) + placeholder.start()) + 1
                missing_tilde.append(f"{os.path.basename(wdl)}:{line_number}")

    assert not missing_tilde, f"command placeholders missing their ~: {missing_tilde}"


def test_runtime_attributes_guard_optional_inputs():
    # e.g. memory: mem + " GB" with Int? mem failed runtime evaluation in Cromwell whenever mem was not given
    unguarded = []
    for wdl in glob.glob(os.path.join(SCRIPTS_DIR, "*.wdl")):
        with open(wdl) as file:
            text = file.read()
        for task in re.split(r"^(?=task |workflow )", text, flags=re.M):
            runtime = re.search(r"runtime\s*\{(.*?)\}", task, re.S)
            if not task.startswith("task ") or runtime is None:
                continue
            optional_names = re.findall(r"^\s*[\w\[\], ]+\?\s+(\w+)", task[: task.index("command")], re.M)
            # the expressions after each 'key:', without string literals and with select_first([...]) and defined(...)
            expressions = "\n".join(re.findall(r"^\s*\w+\s*:(.*)$", runtime.group(1), re.M))
            expressions = re.sub(r'"[^"]*"|select_first\(\[[^\]]*\]\)|defined\(\w+\)', "", expressions)
            for name in optional_names:
                if re.search(r"\b" + name + r"\b", expressions):
                    unguarded.append(f"{os.path.basename(wdl)} task {task.split()[1]}: {name}")

    assert not unguarded, f"optional inputs used in runtime attributes without select_first or defined: {unguarded}"


def test_workflow_inputs_are_used():
    # e.g. Permutect.split_intervals_extra_args was declared but never passed to Mutect2, so setting it did nothing
    unused = []
    for wdl in glob.glob(os.path.join(SCRIPTS_DIR, "*.wdl")):
        with open(wdl) as file:
            text = re.sub(r"#[^\n]*", "", file.read())
        workflow = re.search(r"^workflow \w+ \{.*?(?=^task |\Z)", text, re.M | re.S).group(0)
        inputs = re.search(r"\binput\s*\{(.*?)\n\s*\}", workflow, re.S).group(1)
        rest_of_workflow = workflow.replace(inputs, "")
        for name in re.findall(r"^\s*[\w\[\], ]+\??\s+(\w+)\s*(?:=|$)", inputs, re.M):
            if not re.search(r"\b" + name + r"\b", rest_of_workflow):
                unused.append(f"{os.path.basename(wdl)}: {name}")

    assert not unused, f"workflow inputs that are never used: {unused}"


def task_command(wdl_path: str, task_name: str) -> str:
    with open(wdl_path) as file:
        wdl = file.read()
    task_start = wdl.index("task " + task_name + " {")
    command_start = wdl.index("command <<<", task_start) + len("command <<<")
    return wdl[command_start : wdl.index(">>>", command_start)]


def task_output_files(wdl_path: str, task_name: str) -> dict:
    """
    The file names of a task's file outputs, eg File? maf_segments = "maf_segments.seg", by output name
    """
    with open(wdl_path) as file:
        wdl = file.read()
    task_start = wdl.index("task " + task_name + " {")
    output_start = wdl.index("output {", task_start)
    outputs = wdl[output_start : wdl.index("}", output_start)]
    return dict(re.findall(r'File\??\s+(\w+)\s*=\s*"([^"]+)"', outputs))


def fill_placeholders(command: str, inputs: dict) -> str:
    """
    Replace the WDL placeholders ~{name}, ~{default="..." name} and ~{"prefix" + name}, as Cromwell would.
    An undefined (None) input gives its default, or nothing.
    """

    def fill(match):
        default, prefix, name = match.groups()
        value = inputs[name]
        if value is None:
            return default or ""
        return (prefix or "") + str(value)

    return re.sub(r'~\{(?:default="([^"]*)"\s+)?(?:"([^"]*)"\s*\+\s*)?(\w+)\}', fill, command)


def edit_dataset_arguments(monkeypatch, new_source):
    command = fill_placeholders(
        task_command(EDIT_DATASET_WDL, "EditDataset"),
        {"train_tar": "data.tar", "new_source": new_source, "edit_type": "keep_everything", "extra_args": None},
    )
    words = shlex.split(command.replace("\\\n", " "))
    monkeypatch.setattr(sys, "argv", words[words.index("edit_dataset") :])
    return edit_dataset.parse_arguments()


def test_edit_dataset_command(monkeypatch):
    arguments = edit_dataset_arguments(monkeypatch, new_source=3)
    assert getattr(arguments, constants.TRAIN_TAR_NAME) == ["data.tar"]
    assert getattr(arguments, constants.SOURCE_NAME) == 3

    arguments = edit_dataset_arguments(monkeypatch, new_source=None)
    assert getattr(arguments, constants.TRAIN_TAR_NAME) == ["data.tar"]
    assert getattr(arguments, constants.SOURCE_NAME) is None


def compare_command(working_dir) -> str:
    """
    The shell command of the Compare task, with its inputs replaced by small VCFs written to working_dir
    """
    # permutect_fp has no records, which must not stop the task under set -e
    record_counts = {"permutect_ffn": 2, "other_ffn": 3, "permutect_fp": 0, "other_fp": 4}
    inputs = {"gatk_override": "unused.jar"}
    for name, count in record_counts.items():
        inputs[name] = str(working_dir / (name + ".vcf"))
        lines = ["##fileformat=VCFv4.2", "#CHROM\tPOS\tID\tREF\tALT"] + [f"20\t{n + 1}\t.\tA\tT" for n in range(count)]
        (working_dir / (name + ".vcf")).write_text("\n".join(lines) + "\n")
    for name in ("permutect_summary", "other_summary"):
        inputs[name] = str(working_dir / (name + ".txt"))
        (working_dir / (name + ".txt")).write_text(name + "\n")

    return fill_placeholders(task_command(COMPARE_WDL, "Compare"), inputs)


def run_compare(working_dir, gatk_script: str) -> subprocess.CompletedProcess:
    bin_dir = working_dir / "bin"
    bin_dir.mkdir()
    (bin_dir / "gatk").write_text(gatk_script)
    (bin_dir / "gatk").chmod(0o755)
    env = dict(os.environ, PATH=str(bin_dir) + os.pathsep + os.environ["PATH"])
    return subprocess.run(["bash", "-c", compare_command(working_dir)], cwd=working_dir, env=env, capture_output=True)


def test_compare_summary_has_labeled_counts(tmp_path):
    result = run_compare(tmp_path, FAKE_GATK)
    assert result.returncode == 0, result.stderr.decode()

    summary = (tmp_path / "summary.txt").read_text()
    # the fake gatk copies its -V input, so the created and fixed counts equal the counts of those inputs
    expected = {
        "Permutect FFN count": 2,
        "Other tool FFN count": 3,
        "Permutect FP count": 0,
        "Other tool FP count": 4,
        "Permutect created FFN count": 2,
        "Permutect fixed FFN count": 3,
        "Permutect created FP count": 0,
        "Permutect fixed FP count": 4,
    }
    for label, count in expected.items():
        match = re.search(re.escape(label) + r":\s*(\d+)", summary)
        assert match is not None, f"'{label}' missing from summary:\n{summary}"
        assert int(match.group(1)) == count


def test_compare_fails_when_gatk_fails(tmp_path):
    result = run_compare(tmp_path, "#!/bin/bash\nexit 1\n")
    assert result.returncode != 0
    assert not (tmp_path / "summary.txt").exists()


def test_concatenate_sorts_by_heading_and_creates_no_stray_files(tmp_path):
    # a comment written with $ instead of # made touch create files named $, first, column, etc.
    inputs = []
    for heading in ("zebra", "Apple"):
        path = tmp_path / (heading + ".txt")
        path.write_text(heading + " heading\nbody of " + heading + "\n")
        inputs.append(str(path))
    command = task_command(CONCATENATE_WDL, "concatenate").replace("~{sep=' ' inputs}", " ".join(inputs))

    result = subprocess.run(["bash", "-c", command], cwd=tmp_path, capture_output=True)
    assert result.returncode == 0, result.stderr.decode()
    assert {path.name for path in tmp_path.iterdir()} == {"zebra.txt", "Apple.txt", "TMP", "result.txt"}
    assert (tmp_path / "result.txt").read_text() == "Apple heading\nbody of Apple\n\nzebra heading\nbody of zebra\n\n"


def fake_tool(exit_code: int = 0) -> str:
    # a stand-in that records its name in calls.txt, writes a partial VCF to stdout, and exits with exit_code
    return f'#!/bin/bash\necho "$(basename "$0")" >> calls.txt\necho "##fileformat=VCFv4.2"\nexit {exit_code}\n'


def run_with_fake_tools(working_dir, command: str, fake_tools: dict) -> tuple[bool, list[str]]:
    """
    Run a task's command with the given fake tools on the PATH.
    Returns whether the command succeeded and the names of the fake tools it called, in order.
    """
    bin_dir = working_dir / "bin"
    bin_dir.mkdir()
    for name, script in fake_tools.items():
        (bin_dir / name).write_text(script)
        (bin_dir / name).chmod(0o755)
    env = dict(os.environ, PATH=str(bin_dir) + os.pathsep + os.environ["PATH"])
    result = subprocess.run(["bash", "-c", command], cwd=working_dir, env=env, capture_output=True)
    calls_file = working_dir / "calls.txt"
    calls = calls_file.read_text().split() if calls_file.exists() else []
    return result.returncode == 0, calls


@pytest.mark.parametrize("bcftools_exit_code, expected_calls", [(0, ["bcftools", "gatk", "gatk"]), (255, ["bcftools"])])
def test_split_multiallelics_stops_when_bcftools_fails(tmp_path, bcftools_exit_code, expected_calls):
    # eg bcftools norm exits with 255 at a REF mismatch after writing part of its output
    inputs = {"ref_fasta": "ref.fasta", "input_vcf": "input.vcf", "command_mem": 3500}
    command = fill_placeholders(task_command(PERMUTECT_WDL, "SplitMultiallelics"), inputs)
    fake_tools = {"bcftools": fake_tool(bcftools_exit_code), "gatk": fake_tool()}
    succeeded, calls = run_with_fake_tools(tmp_path, command, fake_tools)
    assert succeeded == (bcftools_exit_code == 0)
    assert calls == expected_calls


@pytest.mark.parametrize(
    "filter_variants_exit_code, expected_calls",
    [(0, ["filter_variants", "tar", "bcftools", "bcftools"]), (1, ["filter_variants"])],
)
def test_permutect_filtering_stops_when_filter_variants_fails(tmp_path, filter_variants_exit_code, expected_calls):
    (tmp_path / "mutect.stats").write_text("statistic\tvalue\ncallable\t60000000\n")
    inputs = {
        "mutect_stats": "mutect.stats",
        "mutect2_vcf": "mutect2.vcf",
        "test_dataset": "test.dataset",
        "artifact_model": "model.pt",
        "contigs_table": "contigs.table",
        "batch_size": 64,
        "num_workers": 2,
        "num_spectrum_iterations": None,
        "spectrum_learning_rate": None,
        "maf_segments": None,
        "normal_maf_segments": None,
        "permutect_filtering_extra_args": None,
    }
    command = fill_placeholders(task_command(PERMUTECT_WDL, "PermutectFiltering"), inputs)
    fake_tools = {"filter_variants": fake_tool(filter_variants_exit_code), "tar": fake_tool(), "bcftools": fake_tool()}
    succeeded, calls = run_with_fake_tools(tmp_path, command, fake_tools)
    assert succeeded == (filter_variants_exit_code == 0)
    assert calls == expected_calls


@pytest.mark.parametrize("with_segments", [True, False])
def test_archive_and_extract_m2_outputs(tmp_path, with_segments):
    # Mutect2 outputs with distinct contents, named as Mutect2 names them
    m2_dir, archive_dir, extract_dir = tmp_path / "mutect2", tmp_path / "archive", tmp_path / "extract"
    for directory in (m2_dir, archive_dir, extract_dir):
        directory.mkdir()
    m2_file_names = {
        "test_dataset": "test-dataset.txt",
        "mutect2_vcf": "merged.vcf",
        "mutect2_vcf_idx": "merged.vcf.idx",
        "contigs_table": "contigs.table",
        "mutect_stats": "merged.stats",
        "maf_segments": "tumor.modelFinal.seg",
        "normal_maf_segments": "normal.modelFinal.seg",
    }
    m2_outputs = {}
    for name, file_name in m2_file_names.items():
        if with_segments or not name.endswith("maf_segments"):
            (m2_dir / file_name).write_text(f"contents of {name}\n")
            m2_outputs[name] = str(m2_dir / file_name)
        else:
            m2_outputs[name] = None

    archive_command = fill_placeholders(task_command(PERMUTECT_WDL, "ArchiveM2Outputs"), m2_outputs)
    result = subprocess.run(["bash", "-c", archive_command], cwd=archive_dir, capture_output=True)
    assert result.returncode == 0, result.stderr.decode()
    archive = archive_dir / task_output_files(PERMUTECT_WDL, "ArchiveM2Outputs")["m2_outputs"]

    # an uncompressed archive of the files themselves, not of links to them
    with tarfile.open(archive, mode="r:") as tar:
        assert all(member.isfile() or member.isdir() for member in tar.getmembers())

    extract_command = fill_placeholders(task_command(PERMUTECT_WDL, "ExtractM2Outputs"), {"cached_m2_outputs": archive})
    result = subprocess.run(["bash", "-c", extract_command], cwd=extract_dir, capture_output=True)
    assert result.returncode == 0, result.stderr.decode()

    # each archived Mutect2 output is extracted as the ExtractM2Outputs output of the same name
    extracted_file_names = task_output_files(PERMUTECT_WDL, "ExtractM2Outputs")
    assert extracted_file_names.keys() == m2_outputs.keys()
    for name, file_name in extracted_file_names.items():
        extracted = extract_dir / file_name
        if m2_outputs[name] is None:
            assert not extracted.exists()
        else:
            assert extracted.read_text() == f"contents of {name}\n"

    # GATK finds the VCF index by its name
    assert extracted_file_names["mutect2_vcf_idx"] == extracted_file_names["mutect2_vcf"] + ".idx"
