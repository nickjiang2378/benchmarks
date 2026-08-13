"""Pytest plugin injected via `-p report_plugin`: dumps {nodeid: outcome} JSON.

Outcomes: passed / failed / error / skipped. Collection failures are recorded
as __collect_error__:<path> entries so the validator can treat a file that
fails to import pre-fix (e.g. tests referencing not-yet-existing symbols) as
failing wholesale.
"""

import json
import os

_results: dict[str, str] = {}


def pytest_runtest_logreport(report):
    nid = report.nodeid
    if report.when == "call":
        _results[nid] = report.outcome
    elif report.outcome == "failed":  # setup/teardown crash
        _results[nid] = "error"
    elif report.when == "setup" and report.outcome == "skipped":
        _results.setdefault(nid, "skipped")


def pytest_collectreport(report):
    if report.failed:
        _results[f"__collect_error__:{report.nodeid}"] = "error"


def pytest_sessionfinish(session, exitstatus):
    out = os.environ.get("REPORT_JSON")
    if out:
        with open(out, "w") as f:
            json.dump(_results, f)
