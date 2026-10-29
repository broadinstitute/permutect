"""
Checks that the GitHub Actions workflows build each commit's Docker image once and run the WDL integration tests at
most once, whether the commit arrives by a push, a pull request, or both.
"""

import glob
import os
import re

import pytest

from permutect.test.test_file_names import test_dir

WORKFLOWS_DIR = os.path.abspath(os.path.join(test_dir, "..", "..", ".github", "workflows"))


def parse_workflow(path: str):
    """
    The events that trigger a workflow, and for each job its condition (None if it always runs), whether it builds the
    Docker image, and whether it runs the WDL tests.  Also the condition of the step that looks for an open pull
    request, or None if there is no such step.
    """
    with open(path) as file:
        text = file.read()
    on_block = re.search(r"^on:\n((?:[ #].*\n|\n)*)", text, re.M).group(1)
    events = set(re.findall(r"^  (\w+):", on_block, re.M))

    jobs = {}
    jobs_block = text[text.index("\njobs:\n") :]
    for name, body in re.findall(r"^  ([\w-]+):\n((?:    .*\n|\n)*)", jobs_block, re.M):
        condition = re.search(r"^    if: (.*)$", body, re.M)
        jobs[name] = (condition.group(1) if condition else None, "docker/build-push-action" in body, "test-wdl" in body)
    pr_step = re.search(r"^      - name: .*\n        id: pr\n        if: (.*)$", text, re.M)
    return events, jobs, (pr_step.group(1) if pr_step else None)


def evaluate(condition: str, context: dict) -> bool:
    """
    Evaluate the subset of GitHub Actions expression syntax that the workflows use: contexts such as github.ref,
    string literals, ==, !=, !, && , || and parentheses.  Missing contexts, such as unset outputs, are empty strings.
    """
    python = re.sub(r"\b[a-z_]+(?:\.[\w-]+)+", lambda match: repr(context.get(match.group(0), "")), condition)
    python = re.sub(r"!(?!=)", " not ", python.replace("&&", " and ").replace("||", " or "))
    return bool(eval(python))


# (pushed to main, changes the Dockerfile or scripts/, the branch has an open pull request)
SCENARIOS = [(True, False, False), (True, True, False)] + [
    (False, changes_scripts, open_pr) for changes_scripts in (False, True) for open_pr in (False, True)
]


@pytest.mark.parametrize("to_main, changes_scripts, open_pr", SCENARIOS)
def test_each_commit_builds_the_image_once(to_main, changes_scripts, open_pr):
    workflows = [parse_workflow(path) for path in glob.glob(os.path.join(WORKFLOWS_DIR, "*.yml"))]
    # a push to a branch with an open pull request also triggers a pull_request event for the same commit
    events = ["push"] + (["pull_request"] if open_pr else [])

    builds, wdl_runs = 0, 0
    for event in events:
        context = {
            "github.event_name": event,
            "github.ref": "refs/heads/main" if to_main else ("refs/pull/1/merge" if event == "pull_request" else "x"),
            "needs.changes.outputs.wdl": str(changes_scripts).lower(),
        }
        for workflow_events, jobs, pr_step_condition in workflows:
            if event not in workflow_events:
                continue
            if pr_step_condition is not None and evaluate(pr_step_condition, context):
                context["needs.changes.outputs.open_pr"] = str(open_pr).lower()
            for condition, builds_image, runs_wdl in jobs.values():
                if condition is None or evaluate(condition, context):
                    builds += builds_image
                    wdl_runs += runs_wdl

    assert builds == 1
    assert wdl_runs == (1 if (to_main or open_pr or changes_scripts) else 0)
