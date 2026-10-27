"""
Checks that the test-unit task, which CI and the pre-commit hook run, covers every test file.
"""

import glob
import os
import shlex

import pytest

from permutect.test.test_file_names import test_dir

tomllib = pytest.importorskip("tomllib")  # from Python 3.11

REPO_DIR = os.path.abspath(os.path.join(test_dir, "..", ".."))

NOT_RUN_BY_TEST_UNIT = {
    # test file names, not tests
    os.path.join("permutect", "test", "test_file_names.py"),
    # needs Cromwell and a Docker image, and has its own task, test-wdl
    os.path.join("permutect", "test", "wdl", "test_permutect_wdl.py"),
}


def test_unit_test_task_covers_every_test_file():
    with open(os.path.join(REPO_DIR, "pixi.toml"), "rb") as file:
        test_unit_command = tomllib.load(file)["tasks"]["test-unit"]
    words = shlex.split(test_unit_command)
    assert words[0] == "pytest"
    test_paths = [os.path.normpath(word) for word in words[1:] if not word.startswith("-")]

    test_files = glob.glob(os.path.join("permutect", "test", "**", "test_*.py"), root_dir=REPO_DIR, recursive=True)
    assert test_files, "no test files found"
    not_covered = [
        test_file
        for test_file in sorted(test_files)
        if test_file not in NOT_RUN_BY_TEST_UNIT
        and not any(test_file == path or test_file.startswith(path + os.sep) for path in test_paths)
    ]
    assert not not_covered, f"test files that 'pixi run test-unit' doesn't run: {not_covered}"
