#!/usr/bin/env python3
"""Grade agent outputs against gold answers. Score in [0,1] per task.

Runs host-side (never inside the agent's container) in fresh git worktrees
with the same pinned recipe as miner Stage 3, so grading cannot be affected
by anything the agent did to its own environment.

  fix       apply agent patch, RESET tests/ to base, apply gold test-patch,
            run hidden tests. score = (F2P pass fraction) x (P2P pass fraction)
  testwrite apply only the tests/ part of the agent patch, run agent's test
            files at base, apply gold code-patch, rerun.
            score = 0 if nothing flips fail->pass, else flips/(flips+still_failing)
  localize  parse localization.json. score = F1 vs gold file set.
  suite     apply only the tests/ part of the agent patch on top of the correct
            implementation, then mutate the source with each hidden variant.
            score = (variants killed / variants) x (agent tests passing / total)

Usage: python3 grade.py --run-name <run>
Writes: work/scores/<run>.json
"""

import argparse
import json
import os
import re
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
from mine import CLONE_DIR, forbidden_introspection
from mine_suite import apply_edits  # same mutation semantics as the validator
# that proved each variant plausible-and-wrong, so the two cannot drift apart

ANSWERS = HERE / "answers"
GWT = HERE / "work" / "gwt"

TEST_DEPS = [
    "pytest==8.3.5", "pytest-mock", "dirty-equals", "cloudpickle",
    "email-validator", "faker", "pytest-benchmark", "pytest-examples",
    "eval-type-backport", "jsonschema", "packaging", "rich",
    "pytest-run-parallel", "pytz",
]


def sh(cmd, cwd=None, timeout=900, env=None):
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True,
                          timeout=timeout, env=env)


def run_pytest(wt: Path, targets: list[str], timeout: int = 900) -> dict | None:
    """{nodeid: outcome}; None if the run timed out (agent-authored tests can hang)."""
    existing = [f for f in targets if (wt / f).exists()]
    if not existing:
        return {}
    out_json = wt / "_report.json"
    env = os.environ.copy()
    env.update({"PYTHONPATH": str(HERE), "REPORT_JSON": str(out_json),
                "PYTHONHASHSEED": "0", "COLUMNS": "120"})
    try:
        sh([str(wt / ".venv/bin/python"), "-m", "pytest", "-q", "--tb=no",
            "--no-header", "-p", "no:cacheprovider", "-p", "report_plugin",
            "--continue-on-collection-errors", *existing], cwd=wt, env=env,
           timeout=timeout)
    except subprocess.TimeoutExpired:
        return None
    if not out_json.exists():
        return {}
    data = json.loads(out_json.read_text())
    out_json.unlink()
    return data


def collect_err_files(outcomes: dict) -> set:
    return {k.split(":", 1)[1].split("::")[0]
            for k in outcomes if k.startswith("__collect_error__:")}


class Worktree:
    def __init__(self, name: str, base: str):
        self.path = GWT / name
        self.base = base
        sh(["git", "-C", str(CLONE_DIR), "worktree", "remove", "--force",
            str(self.path)])
        r = sh(["git", "-C", str(CLONE_DIR), "worktree", "add", "--detach",
                str(self.path), base])
        assert r.returncode == 0, r.stderr

    def install(self):
        r = sh(["uv", "venv", "-q", "--python", "3.12", ".venv"], cwd=self.path)
        assert r.returncode == 0, r.stderr
        r = sh(["uv", "pip", "install", "-q", "--python", ".venv/bin/python",
                "-e", ".", *TEST_DEPS], cwd=self.path, timeout=1800)
        return r.returncode == 0, r.stderr[-300:]

    def apply_reverse(self, patch_text: str, exclude=None):
        p = self.path / "_grade_rev.patch"
        p.write_text(patch_text if patch_text.endswith("\n") else patch_text + "\n")
        cmd = ["git", "apply", "--reverse", "--whitespace=nowarn"]
        if exclude:
            cmd.append(f"--exclude={exclude}")
        r = sh([*cmd, str(p)], cwd=self.path)
        p.unlink()
        return r.returncode == 0, r.stderr[-300:]

    def apply(self, patch_text: str, include=None, exclude=None):
        if not patch_text.strip():
            return True, "empty"
        p = self.path / "_grade.patch"
        p.write_text(patch_text if patch_text.endswith("\n") else patch_text + "\n")
        cmd = ["git", "apply", "--whitespace=nowarn"]
        if include:
            cmd.append(f"--include={include}")
        if exclude:
            cmd.append(f"--exclude={exclude}")
        r = sh([*cmd, str(p)], cwd=self.path)
        p.unlink()
        return r.returncode == 0, r.stderr[-300:]

    def reset_tests(self):
        sh(["git", "checkout", self.base, "--", "tests/"], cwd=self.path)
        sh(["git", "clean", "-fdq", "tests/"], cwd=self.path)

    def reinstall(self):
        sh(["uv", "pip", "install", "-q", "--python", ".venv/bin/python",
            "-e", "."], cwd=self.path, timeout=1800)

    def remove(self):
        sh(["git", "-C", str(CLONE_DIR), "worktree", "remove", "--force",
            str(self.path)])


def patch_paths(patch_text: str) -> set:
    return set(re.findall(r"^(?:\+\+\+|---) [ab]/(\S+)", patch_text or "",
                          re.MULTILINE))


def middle_ring(wt: Path, patch: str, gold_test_files: list) -> list:
    """Regression test files likely affected by the agent's source changes.

    Ring = gold test files + name-matched test files for each touched source
    module + test files that textually reference touched _internal modules.
    Capped to keep grading bounded; a full-suite pass would be exhaustive but
    ~10x slower, and public-API re-exports make import analysis useless here.
    """
    touched = [p for p in patch_paths(patch) if p.startswith("pydantic/")
               and p.endswith(".py")]
    ring = set(gold_test_files)
    for src in touched:
        stem = src.rsplit("/", 1)[-1].removesuffix(".py").lstrip("_")
        ring.update(str(p.relative_to(wt)) for p in (wt / "tests").glob(f"test_{stem}*.py"))
        mod = src.removesuffix(".py").replace("/", ".")
        if "_internal" in mod:
            short = mod.rsplit(".", 1)[-1]
            r = sh(["grep", "-rl", short, "tests/", "--include=test_*.py"], cwd=wt)
            ring.update(l.strip() for l in r.stdout.splitlines() if l.strip())
    ring = {f for f in ring if (wt / f).exists() and "/mypy/" not in f
            and "/benchmarks/" not in f}
    return sorted(ring)[:15]


def ring_regressions(wt, patch: str, ans: dict) -> tuple[int, int]:
    """(broken, checked): ring tests that pass at base but fail after the
    agent's patch. Baseline-aware, so pre-existing env-drift failures never
    count against the agent. Assumes wt currently has agent patch + gold
    test patch applied (call from grade_fix after the F2P run)."""
    ring = middle_ring(wt.path, patch, ans["test_files"])
    extra = [f for f in ring if f not in set(ans["test_files"])]
    if not extra:
        return 0, 0
    after = run_pytest(wt.path, extra, timeout=600)
    if after is None:
        return 0, 0  # ring timeout: don't penalize on grader slowness
    failed_after = {n for n, o in after.items() if o in ("failed", "error")}
    if not failed_after:
        return 0, len(after)
    # Only now pay for a baseline: revert the agent's SOURCE changes (its
    # tests/ hunks were already wiped by reset_tests), rerun the failing files.
    ok, _ = wt.apply_reverse(patch, exclude="tests/*")
    if not ok:
        return 0, len(after)  # can't establish baseline; don't penalize
    base = run_pytest(wt.path, sorted({n.split("::")[0] for n in failed_after}),
                      timeout=600)
    wt.apply(patch, exclude="tests/*")
    base_ok = {n for n, o in (base or {}).items() if o == "passed"}
    broken = failed_after & base_ok
    return len(broken), len(after)


def grade_fix(res: dict, ans: dict) -> dict:
    patch = res.get("patch") or ""
    wt = Worktree(f"g-{res['task_id']}", ans["base_commit"])
    try:
        ok, err = wt.install()
        if not ok:
            return {"score": 0.0, "reason": "grader_install_failed", "detail": err}
        ok, err = wt.apply(patch)
        if not ok:
            return {"score": 0.0, "reason": "patch_apply_failed", "detail": err}
        wt.reset_tests()  # agent changes to tests/ are ignored by design
        if "pyproject.toml" in patch_paths(patch):
            wt.reinstall()
        ok, err = wt.apply(ans["test_patch"])
        if not ok:
            return {"score": 0.0, "reason": "gold_test_patch_failed", "detail": err}
        outcomes = run_pytest(wt.path, ans["test_files"])
        f2p = ans["f2p"]
        p2p = ans["p2p"]
        f2p_pass = sum(outcomes.get(n) == "passed" for n in f2p)
        p2p_pass = sum(outcomes.get(n) == "passed" for n in p2p)
        f2p_frac = f2p_pass / len(f2p)
        p2p_frac = p2p_pass / len(p2p) if p2p else 1.0
        # Middle-ring regression check: only worth running for solutions that
        # fixed something (F2P>0); a zero stays zero regardless.
        ring_broken = ring_checked = 0
        if f2p_pass > 0:
            ring_broken, ring_checked = ring_regressions(wt, patch, ans)
        ring_frac = (1.0 - ring_broken / ring_checked) if ring_checked else 1.0
        return {"score": round(f2p_frac * p2p_frac * ring_frac, 4),
                "f2p": f"{f2p_pass}/{len(f2p)}", "p2p": f"{p2p_pass}/{len(p2p)}",
                "ring": f"{ring_broken} broken/{ring_checked} checked"}
    finally:
        wt.remove()


def grade_testwrite(res: dict, ans: dict) -> dict:
    patch = res.get("patch") or ""
    test_paths = [p for p in patch_paths(patch)
                  if p.startswith("tests/") and re.search(r"test_[^/]+\.py$", p)]
    if not test_paths:
        return {"score": 0.0, "reason": "no_test_files_in_patch"}
    wt = Worktree(f"g-{res['task_id']}", ans["base_commit"])
    try:
        ok, err = wt.install()
        if not ok:
            return {"score": 0.0, "reason": "grader_install_failed", "detail": err}
        # Baseline BEFORE the agent's patch: tests already failing at base
        # (env drift, e.g. email-validator) are not the agent's fault and are
        # excluded from scoring entirely.
        baseline = run_pytest(wt.path, test_paths)
        base_broken = {n for n, o in baseline.items()
                       if o in ("failed", "error")} | {
            n for n in baseline if n.startswith("__collect_error__:")}
        base_cerr = collect_err_files(baseline)
        ok, err = wt.apply(patch, include="tests/*")  # source changes ignored
        if not ok:
            return {"score": 0.0, "reason": "patch_apply_failed", "detail": err}
        pre = run_pytest(wt.path, test_paths)
        pre_cerr = collect_err_files(pre)
        ok, err = wt.apply(ans["code_patch"])
        if not ok:
            return {"score": 0.0, "reason": "gold_code_patch_conflict", "detail": err}
        if "pyproject.toml" in patch_paths(ans["code_patch"]):
            wt.reinstall()
        post = run_pytest(wt.path, test_paths)
        post_cerr = collect_err_files(post)
        nodes = {n for n in (set(pre) | set(post))
                 if not n.startswith("__collect_error__:")
                 and n not in base_broken
                 and n.split("::")[0] not in base_cerr}
        flips = still_fail = 0
        for n in sorted(nodes):
            o_pre, o_post = pre.get(n, "absent"), post.get(n, "absent")
            fname = n.split("::")[0]
            if o_post == "absent" and fname not in post_cerr:
                # Node id exists pre but not post without a collection error:
                # the gold code patch shifted generated test ids (e.g.
                # pytest-examples derives ids from docs line numbers). The
                # agent cannot cause this -- its own broken tests report
                # failed/error -- so drift is excluded, not penalized.
                continue
            failed_pre = o_pre in ("failed", "error") or fname in pre_cerr
            failed_post = o_post in ("failed", "error") or fname in post_cerr
            if failed_pre and o_post == "passed":
                flips += 1
            elif failed_post:
                still_fail += 1
        if post_cerr and not nodes:
            return {"score": 0.0, "reason": "tests_do_not_collect"}
        if flips == 0:
            return {"score": 0.0, "reason": "no_flipping_tests",
                    "detail": f"pre_fail_states={sum(1 for n in nodes if pre.get(n) != 'passed')}"}
        score = flips / (flips + still_fail)
        return {"score": round(score, 4), "flips": flips, "still_fail": still_fail}
    finally:
        wt.remove()


def grade_suite(res: dict, ans: dict) -> dict:
    """Mutation testing: does the agent's suite kill plausible wrong implementations?

    score = kill_rate x validity, where kill_rate is the fraction of hidden
    variants for which at least one *stably passing* agent test fails, and
    validity is the fraction of the agent's tests that pass on the correct code.
    Only stable passers can kill: a test that fails (or flakes) on correct code
    would otherwise "kill" every variant while asserting nothing.
    """
    patch = res.get("patch") or ""
    test_paths = sorted(p for p in patch_paths(patch)
                        if p.startswith("tests/") and re.search(r"test_[^/]+\.py$", p))
    if not test_paths:
        return {"score": 0.0, "reason": "no_test_files_in_patch"}
    wt = Worktree(f"g-{res['task_id']}", ans["base_commit"])
    try:
        ok, err = wt.install()
        if not ok:
            return {"score": 0.0, "reason": "grader_install_failed", "detail": err}
        ok, err = wt.apply(ans["code_patch"])  # correct implementation
        if not ok:
            return {"score": 0.0, "reason": "grader_code_patch_failed", "detail": err}
        src = wt.path / ans["path"]
        correct_src = src.read_text()

        # Tests already failing before the agent touched anything (env drift, or
        # an existing file the agent extended) are not the agent's doing.
        baseline = run_pytest(wt.path, test_paths) or {}
        base_broken = {n for n, o in baseline.items() if o in ("failed", "error")}
        base_cerr = collect_err_files(baseline)

        ok, err = wt.apply(patch, include="tests/*")  # source edits ignored
        if not ok:
            return {"score": 0.0, "reason": "patch_apply_failed", "detail": err}
        introspection = {p: bad for p in test_paths
                         if (wt.path / p).exists()
                         and (bad := forbidden_introspection((wt.path / p).read_text()))}
        if introspection:
            return {"score": 0.0, "reason": "forbidden_introspection",
                    "detail": introspection}

        run_a = run_pytest(wt.path, test_paths)
        run_b = run_pytest(wt.path, test_paths)
        if run_a is None or run_b is None:
            return {"score": 0.0, "reason": "agent_tests_timeout"}
        nodes = {n for n in (set(run_a) | set(run_b))
                 if not n.startswith("__collect_error__:")
                 and n not in base_broken and n.split("::")[0] not in base_cerr}
        if not nodes:
            return {"score": 0.0, "reason": "tests_do_not_collect"}
        killers = {n for n in nodes
                   if run_a.get(n) == "passed" and run_b.get(n) == "passed"}
        validity = len(killers) / len(nodes)
        if not killers:
            return {"score": 0.0, "reason": "no_tests_pass_on_correct_code",
                    "n_tests": len(nodes)}

        killed, survived, broken_variants = [], [], []
        for v in ans["variants"]:
            src.write_text(correct_src)
            if apply_edits(wt.path, ans["path"], v["edits"]) is not None:
                broken_variants.append(v["name"])  # excluded from the denominator
                continue
            out = run_pytest(wt.path, test_paths, timeout=600)
            if out is None:
                survived.append(v["name"])
                continue
            cerr = collect_err_files(out)
            hit = any(out.get(n, "absent") in ("failed", "error", "absent")
                      or n.split("::")[0] in cerr for n in killers)
            (killed if hit else survived).append(v["name"])
        src.write_text(correct_src)

        scored = len(killed) + len(survived)
        if not scored:
            return {"score": 0.0, "reason": "no_applicable_variants"}
        kill_rate = len(killed) / scored
        gold_missed = {v["name"] for v in ans["variants"] if not v["gold_caught"]}
        return {
            "score": round(kill_rate * validity, 4),
            "kill_rate": round(kill_rate, 4),
            "validity": round(validity, 4),
            "killed": f"{len(killed)}/{scored}",
            "killed_gold_missed": f"{len(gold_missed & set(killed))}/{len(gold_missed)}",
            "n_tests": len(nodes),
            "survivors": sorted(survived),
            "broken_variants": broken_variants or None,
        }
    finally:
        wt.remove()


def grade_localize(res: dict, ans: dict) -> dict:
    raw = res.get("localization") or ""
    m = re.search(r"\[.*?\]", raw, re.DOTALL)
    if not m:
        return {"score": 0.0, "reason": "no_json_list"}
    try:
        pred = json.loads(m.group(0))
    except json.JSONDecodeError:
        return {"score": 0.0, "reason": "invalid_json"}
    norm = set()
    for p in pred:
        if isinstance(p, str):
            p = p.strip().removeprefix("/repo/").removeprefix("./")
            norm.add(p)
    gold = set(ans["gold_loc_files"])
    tp = len(norm & gold)
    if not norm or tp == 0:
        return {"score": 0.0, "pred": sorted(norm), "gold": sorted(gold)}
    prec, rec = tp / len(norm), tp / len(gold)
    f1 = 2 * prec * rec / (prec + rec)
    return {"score": round(f1, 4), "pred": sorted(norm), "gold": sorted(gold)}


def main() -> None:
    global ANSWERS
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--answers-dir", default=str(ANSWERS))
    # Scoring one agent run against a second adversary (a different generator
    # model's variants for the same PRs) needs a separate output name, or the
    # first set's scores are overwritten.
    ap.add_argument("--score-name", default="",
                    help="output file name under work/scores (default: --run-name)")
    args = ap.parse_args()
    ANSWERS = Path(args.answers_dir)
    out_dir = HERE / "work" / "agent_out" / args.run_name
    score_name = args.score_name or args.run_name
    GWT.mkdir(parents=True, exist_ok=True)
    scores_path = HERE / "work" / "scores"
    scores_path.mkdir(exist_ok=True)

    rows = []
    for f in sorted(out_dir.glob("pydantic-*.json")):
        if f.name.endswith(".traj.json"):
            continue
        res = json.loads(f.read_text())
        ans = json.loads((ANSWERS / res["task_id"] / "answer.json").read_text())
        t0 = time.time()
        if res.get("status") not in ("ran",):
            g = {"score": 0.0, "reason": f"agent_{res.get('status')}"}
        else:
            grader = {"fix": grade_fix, "testwrite": grade_testwrite,
                      "localize": grade_localize, "suite": grade_suite}[res["type"]]
            try:
                g = grader(res, ans)
            except Exception as e:  # noqa: BLE001
                g = {"score": 0.0, "reason": "grader_exception", "detail": repr(e)[:300]}
        row = {"task_id": res["task_id"], "type": res["type"],
               "cost": res.get("cost"), "n_calls": res.get("n_calls"),
               "exit_status": res.get("exit_status"),
               "grade_seconds": round(time.time() - t0, 1), **g}
        rows.append(row)
        print(f"{row['task_id']:28s} {row['type']:9s} score={row['score']:.3f} "
              f"{row.get('reason','')} ({row['grade_seconds']}s)", flush=True)

    by_type = {}
    for t in ("fix", "testwrite", "localize", "suite"):
        sub = [r["score"] for r in rows if r["type"] == t]
        if sub:
            by_type[t] = round(sum(sub) / len(sub), 4)
    overall = round(sum(r["score"] for r in rows) / len(rows), 4) if rows else 0
    summary = {"run": score_name, "agent_run": args.run_name,
               "answers": str(Path(args.answers_dir).name),
               "n": len(rows), "overall": overall,
               "by_type": by_type,
               "total_cost": round(sum(r.get("cost") or 0 for r in rows), 2)}

    # suite tasks: the two factors of the score, plus the anti-saturation number
    # (kills among variants the PR's own tests would have missed).
    suite = [r for r in rows if r["type"] == "suite"]
    if suite:
        def mean(key):
            xs = [r[key] for r in suite if key in r]
            return round(sum(xs) / len(xs), 4) if xs else None

        gm_hit = gm_tot = 0
        for r in suite:
            if "killed_gold_missed" in r:
                a, b = r["killed_gold_missed"].split("/")
                gm_hit += int(a)
                gm_tot += int(b)
        summary["suite_detail"] = {
            "mean_kill_rate": mean("kill_rate"),
            "mean_validity": mean("validity"),
            "gold_missed_kill_rate": (round(gm_hit / gm_tot, 4) if gm_tot else None),
            "gold_missed_killed": f"{gm_hit}/{gm_tot}",
            "zero_score_tasks": sum(1 for r in suite if r["score"] == 0),
        }
    (scores_path / f"{score_name}.json").write_text(
        json.dumps({"summary": summary, "rows": rows}, indent=1))
    print("\n" + json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
