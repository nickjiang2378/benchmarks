#!/usr/bin/env python3
"""Grade `perf` tasks: the score is 0 or 1.

    score = correctness x fast_enough x sanity      (all three are 0 or 1)

    correctness   1 only if every test passing at the base commit still passes.
                  Any confirmed regression, or a crash, scores 0 -- a perf refactor
                  has no licence to change behaviour. Lost tests are re-run once to
                  confirm, so a single flake cannot decide the score.
    fast_enough   1 only if agent_speedup >= threshold, where
                  threshold = 1 + 0.8 x (gold speedup *measured in this run*).
                  The stored calibration is only a reference: holding an agent to
                  a number measured on another machine under another load fails
                  correct patches. Self-testing the gold patch caught exactly that.
                  The 0.8 is slack for measurement noise, not partial credit -- an
                  optimization that gets part of the way there scores 0.
    sanity        0 if a held-out benchmark the agent never saw regresses by more
                  than max(--regress-tol, the noise measured between two base
                  readings bracketing the agent reading).

Base, gold and agent are all measured **in this same run**, back to back in one
worktree, because absolute timings are not comparable across machines or across
load. Only ratios measured together mean anything.

Correctness is scored against a base-commit baseline rather than against "all
tests pass", so environment drift (a stale pydantic-core wheel, an era-mismatched
pytest) cannot be charged to the agent -- the same approach stage 3 and
grade_testwrite already take.

A crash is a correctness failure, not an infrastructure error: the cheap wrong
answer to several of these tasks segfaults pydantic-core, and that must score 0
rather than erroring out the grader.

Usage:
    python3 grade_perf.py --run-name <run>                  # grade agent outputs
    python3 grade_perf.py --self-test perf-001              # gold patch, expect ~1.0
    python3 grade_perf.py --self-test perf-001 --patch p.diff  # grade an arbitrary patch
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
from mine import CLONE_DIR

ANSWERS = HERE / "answers_perf"
TASKS = HERE / "tasks_perf"
DRIVERS = HERE / "drivers"
GWT = HERE / "work" / "gwt_perf"
CACHE = HERE / "work" / "baselines"

BASE_DEPS = [
    "pytest-mock", "dirty-equals", "cloudpickle", "email-validator", "faker",
    "pytest-benchmark", "eval-type-backport", "jsonschema", "packaging", "rich",
    "pytest-run-parallel", "pytz", "pytest-timeout",
]
SUITE_IGNORES = ["tests/mypy", "tests/typechecking", "tests/benchmarks",
                 "tests/pydantic_core", "tests/plugin"]
HELD_OUT_BENCH = "tests/benchmarks/test_model_schema_generation.py"


def sh(cmd, cwd=None, env=None, timeout=3600):
    return subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True,
                          timeout=timeout)


class Worktree:
    def __init__(self, name: str, base: str):
        self.path = GWT / name
        self.base = base
        GWT.mkdir(parents=True, exist_ok=True)
        sh(["git", "-C", str(CLONE_DIR), "worktree", "remove", "--force", str(self.path)])
        r = sh(["git", "-C", str(CLONE_DIR), "worktree", "add", "-q", "--detach",
                str(self.path), base])
        assert r.returncode == 0, r.stderr
        self.python = self.path / ".venv" / "bin" / "python"

    def install(self, pytest_pin: str) -> tuple[bool, str]:
        r = sh(["uv", "venv", "-q", "--python", "3.12", ".venv"], cwd=self.path)
        if r.returncode != 0:
            return False, r.stderr[-300:]
        deps = [pytest_pin, *BASE_DEPS]
        # Prefer building pydantic-core from the workspace: post-unification the
        # pyproject pin does not track the in-repo Rust source, so a PyPI wheel can
        # be silently stale. Fall back to wheels when no Rust toolchain exists.
        r = sh(["uv", "pip", "install", "-q", "--python", ".venv/bin/python", "-e", ".", *deps],
               cwd=self.path)
        self.built_core = r.returncode == 0
        if not self.built_core:
            r = sh(["uv", "pip", "install", "-q", "--python", ".venv/bin/python",
                    "--no-sources", "-e", ".", *deps], cwd=self.path)
        return r.returncode == 0, r.stderr[-300:]

    def apply(self, patch: str) -> tuple[bool, str]:
        if not patch.strip():
            return True, "empty"
        p = self.path / "_grade.patch"
        p.write_text(patch if patch.endswith("\n") else patch + "\n")
        r = sh(["git", "apply", "--whitespace=nowarn", str(p)], cwd=self.path)
        p.unlink()
        return r.returncode == 0, r.stderr[-300:]

    def reset_source(self):
        """Back to the pristine base tree (also drops any agent edit to tests/)."""
        sh(["git", "checkout", "-q", self.base, "--", "."], cwd=self.path)
        sh(["git", "clean", "-fdq", "-e", ".venv", "-e", "perf_bench.py"], cwd=self.path)

    def remove(self):
        sh(["git", "-C", str(CLONE_DIR), "worktree", "remove", "--force", str(self.path)])


def env_for(wt: Worktree) -> dict:
    env = os.environ.copy()
    env.update({"PYTHONHASHSEED": "0", "COLUMNS": "120", "PAGER": "cat",
                "PYTHONDONTWRITEBYTECODE": "1"})
    return env


def measure(wt: Worktree, reps: int) -> float:
    """Min elapsed ms over `reps` fresh processes. Raises if the driver fails."""
    vals = []
    for _ in range(reps):
        r = sh([str(wt.python), "perf_bench.py"], cwd=wt.path, env=env_for(wt), timeout=1800)
        if r.returncode != 0:
            raise RuntimeError(f"driver failed: {(r.stderr or r.stdout).strip()[-300:]}")
        vals.append(float(r.stdout.strip().splitlines()[-1]))
    return min(vals)


def run_suite(wt: Worktree, targets: list[str] | None = None) -> tuple[dict, bool]:
    """Returns ({nodeid: outcome}, crashed). `targets` limits it to given node ids."""
    out_json = wt.path / "_report.json"
    out_json.unlink(missing_ok=True)
    env = env_for(wt)
    env.update({"PYTHONPATH": str(HERE), "REPORT_JSON": str(out_json)})
    scope = targets if targets else ["tests/"]
    r = sh([str(wt.python), "-m", "pytest", *scope, "-q", "-W", "ignore", "--tb=no",
            "--no-header", "-p", "no:cacheprovider", "-p", "report_plugin",
            "--continue-on-collection-errors", *[f"--ignore={i}" for i in SUITE_IGNORES]],
           cwd=wt.path, env=env, timeout=5400)
    # Signal death (segfault / abort) or a vanished report means the interpreter
    # died mid-run: the results we do have are not trustworthy.
    crashed = r.returncode < 0 or r.returncode in (134, 139) or not out_json.exists()
    if not out_json.exists():
        return {}, True
    data = json.loads(out_json.read_text())
    out_json.unlink()
    if "Fatal Python error" in (r.stdout + r.stderr):
        crashed = True
    return data, crashed


def held_out(wt: Worktree) -> dict:
    """Median time per benchmark from a repo benchmark file the agent never sees."""
    if not (wt.path / HELD_OUT_BENCH).exists():
        return {}
    out = wt.path / "_bench.json"
    sh([str(wt.python), "-m", "pytest", HELD_OUT_BENCH, "--benchmark-enable", "-q",
        "-W", "ignore", f"--benchmark-json={out}", "-p", "no:cacheprovider"],
       cwd=wt.path, env=env_for(wt), timeout=3600)
    if not out.exists():
        return {}
    data = json.loads(out.read_text())
    out.unlink()
    return {b["fullname"].split("::")[-1]: b["stats"]["median"] for b in data["benchmarks"]}


def passing(outcomes: dict) -> set:
    return {n for n, o in outcomes.items() if o == "passed"}


def grade_one(task_id: str, patch: str, reps: int, regress_tol: float,
              do_sanity: bool, skip_suite: bool = False) -> dict:
    ans = json.loads((ANSWERS / task_id / "answer.json").read_text())
    # Blind variants ship no benchmark, but they are still graded on one -- the same
    # driver, from the same calibration. Fall back to drivers/ so the sighted and
    # blind variants of a task are scored identically and their scores compare.
    shipped = TASKS / task_id / "perf_bench.py"
    driver_src = (shipped if shipped.exists()
                  else DRIVERS / ans["driver"]).read_text()
    out: dict = {"task_id": task_id, "pr": ans["pr_number"]}

    wt = Worktree(f"p-{task_id}", ans["base_commit"])
    try:
        (wt.path / "perf_bench.py").write_text(driver_src)
        ok, err = wt.install(ans["pytest_pin"])
        if not ok:
            return {**out, "score": 0.0, "reason": "grader_install_failed", "detail": err}
        out["core_from_workspace"] = getattr(wt, "built_core", False)

        # --- base: the state the agent started from
        t_base = measure(wt, reps)
        base_bench_a = held_out(wt) if do_sanity else {}
        base_outcomes, base_crashed = (({}, False) if skip_suite else run_suite(wt))
        if not skip_suite and (base_crashed or not base_outcomes):
            return {**out, "score": 0.0, "reason": "baseline_suite_unusable"}
        base_pass = passing(base_outcomes)

        # --- gold: same worktree, same conditions, for a comparable ratio
        okg, errg = wt.apply(ans["code_patch"])
        t_gold = measure(wt, reps) if okg else None
        # Gold's held-out profile too: a correct optimisation can legitimately cost
        # a little on paths where it adds bookkeeping but saves nothing. Holding the
        # agent to the *unpatched* tree would fail gold itself for that trade-off.
        gold_bench = held_out(wt) if (do_sanity and okg) else {}
        wt.reset_source()
        (wt.path / "perf_bench.py").write_text(driver_src)

        # --- agent
        ok, err = wt.apply(patch)
        if not ok:
            return {**out, "score": 0.0, "reason": "patch_apply_failed", "detail": err,
                    "base_ms": t_base, "gold_ms": t_gold}
        # tests/ and the benchmark are restored: agent edits there are discarded.
        sh(["git", "checkout", "-q", wt.base, "--", "tests/"], cwd=wt.path)
        sh(["git", "clean", "-fdq", "tests/"], cwd=wt.path)
        (wt.path / "perf_bench.py").write_text(driver_src)

        try:
            t_agent = measure(wt, reps)
        except RuntimeError as exc:
            return {**out, "score": 0.0, "reason": "driver_failed_after_patch",
                    "detail": str(exc)[:300], "base_ms": t_base, "gold_ms": t_gold}

        # Held-out benchmarks for the patched tree, then base again. Comparing a
        # reading taken now against one taken before a long suite run is invalid on
        # a loaded box: drift over that window can exceed the tolerance and fail a
        # perfect patch. Bracketing base around the agent reading gives both a
        # fair reference and a measured noise estimate.
        agent_bench = held_out(wt) if do_sanity else {}
        agent_outcomes, crashed = (({}, False) if skip_suite else run_suite(wt))
        if crashed:
            return {**out, "score": 0.0, "reason": "suite_crashed",
                    "base_ms": t_base, "gold_ms": t_gold, "agent_ms": t_agent,
                    "agent_speedup": round(t_base / t_agent, 4)}
        # Zero tolerance: breaking any test that passed at base scores 0. A perf
        # refactor has no licence to change behaviour, so "most tests still pass" is
        # not a partial success -- and scoring it as a fraction let a patch that
        # broke 51 tests keep 98.7% of its score.
        lost = sorted(base_pass - passing(agent_outcomes))
        if lost:
            # Confirm before zeroing: with a binary rule a single flaky or
            # order-dependent test would decide the whole score, so re-run just the
            # lost node ids and keep only those that fail again.
            recheck, recrashed = run_suite(wt, targets=lost)
            if recrashed:
                return {**out, "score": 0.0, "reason": "suite_crashed_on_recheck",
                        "tests_lost": len(lost), "base_ms": t_base, "gold_ms": t_gold,
                        "agent_ms": t_agent,
                        "agent_speedup": round(t_base / t_agent, 4)}
            lost = [n for n in lost if recheck.get(n) != "passed"]
        still = len(base_pass) - len(lost)
        correctness = 0.0 if lost else 1.0
        if skip_suite:
            correctness = 1.0

        # The speed gate is pass/fail: either the change is as fast as the
        # maintainers' own, or it is not. No partial credit -- an optimization that
        # gets part of the way there has not done the job.
        #
        # Measured against gold *as measured in this run*, never against the stored
        # calibration: that number came from a different machine under a different
        # load, and holding an agent to it is not meaningful.
        speedup = t_base / t_agent
        gold_speedup = t_base / t_gold if t_gold else None
        if gold_speedup:
            tol = ans.get("speed_tolerance", ans.get("credit_fraction", 0.8))
            threshold = 1 + tol * (gold_speedup - 1)
        else:  # gold patch would not apply; fall back to the stored calibration
            threshold = ans["threshold_speedup"]
        fast_enough = 1.0 if (threshold > 1 and speedup >= threshold) else 0.0

        sanity, regressions = 1.0, []
        if do_sanity and agent_bench:
            wt.reset_source()
            (wt.path / "perf_bench.py").write_text(driver_src)
            base_bench_b = held_out(wt)
            for name, t_agent_b in agent_bench.items():
                base_readings = [b[name] for b in (base_bench_a, base_bench_b) if name in b]
                if not base_readings:
                    continue
                # Reference = the most permissive of the unpatched tree and gold, so
                # a trade-off gold also makes is not charged to the agent. Require the
                # slowdown to clear the noise observed between the two base readings.
                ref = max(base_readings + ([gold_bench[name]] if name in gold_bench else []))
                noise = max(base_readings) / min(base_readings)
                if t_agent_b / ref > max(regress_tol, noise):
                    regressions.append({"benchmark": name,
                                        "slowdown": round(t_agent_b / ref, 3),
                                        "base_noise": round(noise, 3),
                                        "vs_gold": round(t_agent_b / gold_bench[name], 3)
                                        if name in gold_bench else None})
            if regressions:
                sanity = 0.0

        return {**out,
                "score": round(correctness * fast_enough * sanity, 4),
                "correctness": round(correctness, 4),
                "tests_kept": f"{still}/{len(base_pass)}" if base_pass else "skipped",
                "tests_lost": len(lost),
                "tests_lost_sample": lost[:10],
                "fast_enough": fast_enough,
                "sanity": sanity, "regressions": regressions,
                "base_ms": round(t_base, 3),
                "gold_ms": round(t_gold, 3) if t_gold else None,
                "agent_ms": round(t_agent, 3),
                "agent_speedup": round(speedup, 4),
                "gold_speedup_this_run": round(gold_speedup, 4) if gold_speedup else None,
                "threshold_this_run": round(threshold, 4),
                "threshold_calibrated": ans["threshold_speedup"]}
    finally:
        wt.remove()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", help="grade eval/work/agent_out/<run>/ agent outputs")
    ap.add_argument("--self-test", metavar="TASK_ID",
                    help="grade the gold patch (sanity check: expect ~1.0)")
    ap.add_argument("--patch", type=Path, help="with --self-test, grade this patch instead")
    ap.add_argument("--reps", type=int, default=5, help="driver repetitions per side")
    ap.add_argument("--regress-tol", type=float, default=1.10,
                    help="held-out benchmark slowdown that zeroes the score")
    ap.add_argument("--no-sanity", action="store_true", help="skip held-out benchmarks")
    ap.add_argument("--skip-suite", action="store_true",
                    help="skip the test suite (validates the timing math only, ~4x faster)")
    args = ap.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)

    if args.self_test:
        task_id = args.self_test
        if args.patch:
            patch = args.patch.read_text()
            label = f"patch {args.patch.name}"
        else:
            patch = json.loads((ANSWERS / task_id / "answer.json").read_text())["code_patch"]
            label = "gold patch"
        print(f"self-test {task_id} with {label} (reps={args.reps})")
        res = grade_one(task_id, patch, args.reps, args.regress_tol,
                        not args.no_sanity, args.skip_suite)
        print(json.dumps(res, indent=1))
        return

    if not args.run_name:
        raise SystemExit("need --run-name or --self-test")
    # run_agent.py writes to work/agent_out/<run>/, and drops a <id>.traj.json
    # trajectory beside each result -- those are not results, skip them.
    run_dir = HERE / "work" / "agent_out" / args.run_name
    results = [json.loads(p.read_text()) for p in sorted(run_dir.glob("*.json"))
               if not p.name.endswith(".traj.json")]
    scores = {}
    for res in results:
        tid = res["task_id"]
        if not (ANSWERS / tid / "answer.json").exists():
            continue
        scores[tid] = grade_one(tid, res.get("patch") or "", args.reps,
                                args.regress_tol, not args.no_sanity, args.skip_suite)
        print(f"{tid}  score={scores[tid]['score']}  {scores[tid].get('reason','')}")
    out = HERE / "work" / "scores" / f"{args.run_name}-perf.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(scores, indent=1) + "\n")
    vals = [s["score"] for s in scores.values()]
    if vals:
        print(f"\n{len(vals)} tasks  mean={statistics.mean(vals):.4f}")
    print(f"-> {out}")


if __name__ == "__main__":
    main()
