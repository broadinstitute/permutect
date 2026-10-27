"""
Checks that the console scripts in pyproject.toml and the README agree with the code and the files in the repo.
"""

import importlib
import os
import re

import pytest

from permutect.test.test_file_names import test_dir

tomllib = pytest.importorskip("tomllib")  # from Python 3.11

REPO_DIR = os.path.abspath(os.path.join(test_dir, "..", ".."))


def console_scripts() -> dict:
    with open(os.path.join(REPO_DIR, "pyproject.toml"), "rb") as file:
        return tomllib.load(file)["project"]["scripts"]


def readme() -> str:
    with open(os.path.join(REPO_DIR, "README.md")) as file:
        return file.read()


def test_console_scripts_point_to_functions():
    for name, entry_point in console_scripts().items():
        module_name, function_name = entry_point.split(":")
        module = importlib.import_module(module_name)  # ModuleNotFoundError for a deleted tool
        assert callable(getattr(module, function_name, None)), f"{name}: {entry_point} is not a function"


def test_readme_lists_the_console_scripts():
    match = re.search(r"This installs the command line tools \(([^)]*)\)", readme())
    assert match is not None, "the README no longer lists the command line tools"
    assert set(re.findall(r"`(\w+)`", match.group(1))) == set(console_scripts())


def test_readme_sections_about_files_name_files_that_exist():
    # a heading that is a single file name or path, eg "## pyproject.toml" or "## scripts/permutect.wdl"
    file_headings = [heading for heading in re.findall(r"^#+ (\S+)$", readme(), re.M) if "." in heading]
    assert file_headings, "no README sections about files found"
    missing = [heading for heading in file_headings if not os.path.exists(os.path.join(REPO_DIR, heading))]
    assert not missing, f"README sections about files that don't exist: {missing}"
