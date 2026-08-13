#!/usr/bin/env python3
"""Category 3, performance optimization: make a slow path fast without breaking it.

Given a base commit and a description of what is slow, the agent must optimize
it. A change that breaks any test, or that fails to reach the speedup the
maintainers achieved, scores zero.

Different in kind from the other two categories, which are graded on hidden
tests alone:

  - **The prompt cannot leak the answer.** "This is slow, make it fast" contains
    no API names or message wording, so it sidesteps the message-matching
    unfairness that affects some `fix` tasks. There is nothing to guess about
    intent.
  - **It is hard to saturate.** Passing needs a real understanding of the data
    structure being traversed, not a localized edit.
  - **The bar is the maintainers' own change**, not a threshold anyone had to
    invent: the reference patch defines what counts as fast enough.
  - **It has a built-in adversarial trap.** The obvious fast fix is often
    incorrect, and the existing suite catches it (see below).

    harvest    GitHub labels  ->  work/perf_candidates.jsonl
    measure    triage each candidate on the repo's own benchmarks: wall clock,
               callgrind instruction reads, and the noise floor
    calibrate  measure each shipped task's driver on base vs gold
    assemble   ->  tasks_perf/ + answers_perf/

    export GITHUB_TOKEN=...
    python3 mine_perf.py harvest
    python3 mine_perf.py measure --pr 13523 11244 7947 10868 13573
    python3 mine_perf.py calibrate
    python3 mine_perf.py assemble --blind

Two findings worth knowing before you use this.

**Wall clock is not good enough on its own.** Running the *same* commit twice
gives a noise floor of mean 1.5% / max 3.8% -- the same magnitude as a typical
constant-factor perf PR, which means several benchmarks report the optimized
commit as slower. So `measure` also supports callgrind instruction reads, with
the fixed `import pydantic` cost cancelled by differencing two runs. That is
deterministic to ~3 parts per million, ~3000x better resolution, at ~25s a run.
Grade on wall clock when the gold speedup is >= 1.20x and on instructions
otherwise. Calibrate and grade on an *idle* machine: one task's noise floor
measured 1.0067x quiet and 1.074x under load, enough to swallow its entire
1.275x signal.

**The naive fix is wrong, and the suite catches it.** The best candidate's real
change memoizes schema traversal *while* preserving per-encounter bookkeeping.
The obvious four-line version ("skip any schema object already visited") does fix
the performance -- and segfaults pydantic-core, because a definition-ref
reachable from its own definition gets inlined into a cyclic schema. So the task
separates "made the benchmark fast" from "made the benchmark fast and correct",
and the existing suite is a sufficient guard. The grading implication: a hard
crash kills the pytest process, so the grader must treat a signal exit as a
correctness failure rather than as an infrastructure error.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import statistics
import subprocess
import urllib.parse
import urllib.request
from pathlib import Path

from mine import CLONE_DIR, HERE, WORK, load_env, sh

DRIVERS = HERE / "drivers"
TASKS_DIR = HERE / "tasks_perf"
ANSWERS_DIR = HERE / "answers_perf"
WT = WORK / "wt_perf"
VENVS = WORK / "venvs_perf"
CANDIDATES = WORK / "perf_candidates.jsonl"
MEASURED = WORK / "perf_measured.jsonl"
CALIBRATION = WORK / "perf_driver_calibration.json"

LABELS = ("relnotes-performance", "topic-performance")

# Same pinned recipe as mine.py's validate stage, plus the benchmark plugin.
TEST_DEPS = [
    "pytest==8.3.5", "pytest-benchmark", "pytest-run-parallel", "pytest-mock",
    "dirty-equals", "jsonschema", "typing-extensions", "annotated-types",
    "typing-inspection", "email-validator", "cloudpickle", "faker",
    "eval-type-backport", "packaging", "rich",
]

# Most pydantic perf PRs move schema-*build* cost -- validation and serialization
# live in Rust -- so schema generation is the default benchmark set.
DEFAULT_BENCH = [
    "test_model_schema_generation.py",
    "test_model_schema_generation_recursive.py",
    "test_fastapi_startup_simple.py",
    "test_validators_build.py",
]

CANARY = (
    "from pydantic import BaseModel\n"
    "class M(BaseModel):\n    a: int\n    b: str = 'x'\n"
    "assert M(a=1).b == 'x'\n"
    "assert M.model_json_schema()['properties']['a']['type'] == 'integer'\n"
)


def git(*args: str) -> str:
    r = subprocess.run(["git", "-C", str(CLONE_DIR), *args],
                       capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else ""


def git_checked(*args: str) -> str:
    r = subprocess.run(["git", "-C", str(CLONE_DIR), *args],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"git {args[:3]} failed: {r.stderr[-300:]}")
    return r.stdout


# ---------------------------------------------------------------------------
# harvest -- merged PRs carrying a performance label
# ---------------------------------------------------------------------------
#
# Unlike stage 1 there is almost no attrition here: perf PRs are overwhelmingly
# Python-side. The attrition is all in `measure`, where about half the candidates
# turn out not to move a benchmark measurably.


def gh(url: str, tok: str, tries: int = 4):
    req = urllib.request.Request(
        url, headers={"Authorization": f"Bearer {tok}",
                      "Accept": "application/vnd.github+json"})
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(req) as r:
                return json.load(r)
        except Exception:  # rate limit / transient
            if attempt == tries - 1:
                raise
            import time
            time.sleep(2 ** attempt * 3)


def local_commits(pr_number: int) -> tuple[str, str] | None:
    """pydantic squash-merges, so the PR is one commit whose subject ends `(#N)`.

    The parens are escaped: under -E they would be a group, so `(#N)$` would
    require the subject to end in the digits rather than in `)`.
    """
    sha = git("log", "--format=%H", "-1", "-E", rf"--grep=\(#{pr_number}\)$", "main")
    if not sha:
        return None
    parent = git("rev-parse", f"{sha}^")
    return (parent, sha) if parent else None


def core_pin(commit: str) -> str | None:
    for line in git("show", f"{commit}:pyproject.toml").splitlines():
        line = line.strip().strip(",").strip("'\"")
        if line.startswith("pydantic-core=="):
            return line.split("==", 1)[1].strip("'\" ")
    return None


def cmd_harvest(args) -> None:
    tok = load_env("GITHUB_TOKEN")
    cache = WORK / "perf_prs_raw.json"
    if cache.exists() and not args.refresh:
        prs = {int(k): v for k, v in json.loads(cache.read_text()).items()}
        print(f"loaded {len(prs)} PRs from cache; --refresh to re-query")
    else:
        prs: dict[int, dict] = {}
        for label in LABELS:
            for page in range(1, 6):
                q = f'repo:pydantic/pydantic is:pr is:merged label:"{label}"'
                data = gh("https://api.github.com/search/issues?q="
                          + urllib.parse.quote(q)
                          + f"&per_page=100&page={page}&sort=created&order=desc", tok)
                for item in data["items"]:
                    rec = prs.setdefault(item["number"], {"labels": []})
                    rec["title"] = item["title"]
                    rec["closed_at"] = item["closed_at"]
                    rec["labels"].append(label)
                if len(data["items"]) < 100:
                    break
        cache.write_text(json.dumps({str(k): v for k, v in prs.items()}, indent=1))
        print(f"harvested {len(prs)} merged perf-labelled PRs")

    out, discards = [], []
    for number in sorted(prs, reverse=True):
        meta = prs[number]
        commits = local_commits(number)
        if commits is None:
            discards.append({"pr": number, "reason": "commit_not_in_local_clone"})
            continue
        base, gold = commits
        files = gh(f"https://api.github.com/repos/pydantic/pydantic/pulls/"
                   f"{number}/files?per_page=100", tok)
        paths = [f["filename"] for f in files]
        src = [p for p in paths if p.startswith("pydantic/") and p.endswith(".py")]
        # The task venv installs pydantic-core as a built wheel, so patching its
        # Rust source silently no-ops -- same reason mine.py's validate discards it.
        rust = [p for p in paths if p.startswith("pydantic-core/")]
        if rust:
            discards.append({"pr": number, "reason": "touches_pydantic_core_rust",
                             "files": rust})
            continue
        if not src:
            discards.append({"pr": number, "reason": "no_python_source_change",
                             "files": paths})
            continue
        pr = gh(f"https://api.github.com/repos/pydantic/pydantic/pulls/{number}", tok)
        out.append({
            "pr": number, "title": meta["title"], "merged_at": meta["closed_at"][:10],
            "labels": sorted(set(meta["labels"])),
            "base_commit": base, "gold_commit": gold, "core_pin": core_pin(base),
            "additions": pr["additions"], "deletions": pr["deletions"],
            "src_files": src,
            "test_files": [p for p in paths
                           if p.startswith("tests/") and "benchmark" not in p],
            "bench_files": [p for p in paths if "benchmarks/" in p],
            "n_files": len(paths),
        })
        print(f"  PR{number:>6} {meta['closed_at'][:10]} +{pr['additions']:<5} "
              f"-{pr['deletions']:<5} src={len(src)}  {meta['title'][:52]}")

    CANDIDATES.write_text("".join(json.dumps(r) + "\n" for r in out))
    (WORK / "perf_discards.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in discards))
    print(f"\n{len(out)} candidates -> {CANDIDATES.name}")
    for reason in sorted({d["reason"] for d in discards}):
        print(f"    {sum(d['reason'] == reason for d in discards):>3}  {reason}")


# ---------------------------------------------------------------------------
# measurement harness (shared by measure and calibrate)
# ---------------------------------------------------------------------------
#
# pydantic is pure Python, so instead of `pip install -e .` per commit we install
# only pydantic-core=={pin} from PyPI once per era and point PYTHONPATH at the
# worktree -- ~50x faster. That relies on the pyproject pin matching the in-repo
# Rust source, which is untrue for some post-unification (2025+) commits where the
# version string did not move while the Rust code did. The canary catches it and
# marks the candidate env_mismatch; rerun those with --install.


def worktree(name: str, commit: str) -> Path:
    path = WT / name
    if path.exists():
        return path
    WT.mkdir(parents=True, exist_ok=True)
    r = sh(["git", "-C", str(CLONE_DIR), "worktree", "add", "-q", "--detach",
            str(path), commit])
    if r.returncode != 0:
        raise RuntimeError(f"worktree add failed for {commit}: {r.stderr[-300:]}")
    return path


def era_venv(pin: str) -> Path:
    venv = VENVS / f"core-{pin}"
    if (venv / "bin" / "python").exists():
        return venv
    VENVS.mkdir(parents=True, exist_ok=True)
    assert sh(["uv", "venv", "-q", "--python", "3.12", str(venv)]).returncode == 0
    r = sh(["uv", "pip", "install", "-q", "--python", str(venv / "bin/python"),
            f"pydantic-core=={pin}", *TEST_DEPS], timeout=1800)
    if r.returncode != 0:
        raise RuntimeError(f"venv install failed for core {pin}: {r.stderr[-400:]}")
    return venv


def install_editable(wt: Path) -> Path:
    """Fallback: a real install, building pydantic-core from the worktree."""
    venv = wt / ".venv"
    if not (venv / "bin" / "python").exists():
        assert sh(["uv", "venv", "-q", "--python", "3.12", ".venv"], cwd=wt).returncode == 0
        r = sh(["uv", "pip", "install", "-q", "--python", ".venv/bin/python",
                "-e", ".", *TEST_DEPS], cwd=wt, timeout=1800)
        if r.returncode != 0:
            raise RuntimeError(f"editable install failed: {r.stderr[-400:]}")
    return venv


def env_for(wt: Path, use_pythonpath: bool) -> dict:
    env = {"PYTHONHASHSEED": "0", "PATH": "/usr/bin:/bin:/usr/local/bin",
           "HOME": str(Path.home()), "COLUMNS": "120"}
    if use_pythonpath:
        env["PYTHONPATH"] = str(wt)
    return env


def canary_ok(python: Path, wt: Path, use_pythonpath: bool) -> tuple[bool, str]:
    r = sh([str(python), "-c", CANARY], cwd=wt, env=env_for(wt, use_pythonpath))
    return r.returncode == 0, r.stderr.strip().splitlines()[-1] if r.returncode else ""


def run_driver(python: Path, wt: Path, driver: Path, reps: int) -> tuple[float, list]:
    """Copy the driver in, run it `reps` times in fresh processes, return (min, all)."""
    shipped = wt / "perf_bench.py"
    shipped.write_text(driver.read_text())
    try:
        vals = []
        for _ in range(reps):
            r = sh([str(python), "perf_bench.py"], cwd=wt, env=env_for(wt, True),
                   timeout=1200)
            if r.returncode != 0:
                raise RuntimeError(f"driver failed: {(r.stderr or r.stdout).strip()[-400:]}")
            vals.append(float(r.stdout.strip().splitlines()[-1]))
        return min(vals), vals
    finally:
        shipped.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# measure -- triage candidates on the repo's own benchmark suite
# ---------------------------------------------------------------------------


def freeze_harness(gold: Path, base: Path) -> None:
    """Both sides must run the *same* benchmark code -- PRs sometimes add it."""
    stage = WORK / "_bench_stage"
    shutil.rmtree(stage, ignore_errors=True)
    shutil.copytree(gold / "tests" / "benchmarks", stage)
    for wt in (base, gold):
        target = wt / "tests" / "benchmarks"
        shutil.rmtree(target, ignore_errors=True)
        shutil.copytree(stage, target)


def restore_tests(wt: Path) -> None:
    sh(["git", "checkout", "-q", "HEAD", "--", "tests/"], cwd=wt)
    sh(["git", "clean", "-fdq", "tests/"], cwd=wt)


def run_bench(python: Path, wt: Path, files: list[str], out: Path, use_pp: bool) -> dict:
    present = [f"tests/benchmarks/{f}" for f in files
               if (wt / "tests/benchmarks" / f).exists()]
    if not present:
        return {}
    sh([str(python), "-m", "pytest", *present, "--benchmark-enable", "-q", "-W", "ignore",
        f"--benchmark-json={out}", "-p", "no:cacheprovider"],
       cwd=wt, env=env_for(wt, use_pp), timeout=2400)
    if not out.exists():
        return {}
    return {b["fullname"].split("::")[-1]: b["stats"]
            for b in json.loads(out.read_text())["benchmarks"]}


def callgrind_ir(python: Path, wt: Path, driver: Path, reps: int, use_pp: bool) -> int:
    r = sh(["valgrind", "--tool=callgrind", "--callgrind-out-file=/dev/null",
            str(python), str(driver), str(reps)],
           cwd=wt, env=env_for(wt, use_pp), timeout=2400)
    m = re.search(r"I\s+refs:\s+([\d,]+)", r.stderr + r.stdout)
    if not m:
        raise RuntimeError(f"could not parse callgrind output: {(r.stderr or r.stdout)[-300:]}")
    return int(m.group(1).replace(",", ""))


def ratios(base: dict, gold: dict) -> dict:
    common = [k for k in base if k in gold]
    if not common:
        return {}
    su = sorted(base[k]["min"] / gold[k]["min"] for k in common)
    return {"n": len(su), "median": round(statistics.median(su), 4),
            "best": round(max(su), 4), "worst": round(min(su), 4),
            "n_above_1_20": sum(s >= 1.20 for s in su),
            "per_benchmark": dict(sorted(
                ((k, round(base[k]["min"] / gold[k]["min"], 4)) for k in common),
                key=lambda kv: -kv[1]))}


def measure_one(cand: dict, args) -> dict:
    pr = cand["pr"]
    print(f"\n=== PR{pr} {cand['title'][:60]}")
    base_wt = worktree(f"{pr}-base", cand["base_commit"])
    gold_wt = worktree(f"{pr}-gold", cand["gold_commit"])
    rec = {k: cand[k] for k in ("pr", "title", "merged_at", "base_commit", "gold_commit")}

    if args.install:
        pys = {"base": install_editable(base_wt) / "bin/python",
               "gold": install_editable(gold_wt) / "bin/python"}
        use_pp = False
    else:
        if not cand.get("core_pin"):
            return {**rec, "status": "no_core_pin"}
        venv = era_venv(cand["core_pin"])
        pys = {"base": venv / "bin/python", "gold": venv / "bin/python"}
        use_pp = True

    for side, wt in (("base", base_wt), ("gold", gold_wt)):
        ok, err = canary_ok(pys[side], wt, use_pp)
        if not ok:
            print(f"  canary FAILED on {side}: {err}")
            return {**rec, "status": "env_mismatch", "detail": err}

    try:
        freeze_harness(gold_wt, base_wt)
        results = {}
        for side, wt in (("base", base_wt), ("gold", gold_wt), ("base2", base_wt)):
            results[side] = run_bench(pys["base" if side.startswith("base") else side],
                                      wt, args.bench, WORK / f"bm_{pr}_{side}.json", use_pp)
            print(f"  {side:<5} {len(results[side])} benchmarks")
    finally:
        restore_tests(base_wt)
        restore_tests(gold_wt)

    if not results["base"] or not results["gold"]:
        return {**rec, "status": "no_benchmarks_ran"}

    rec["wall"] = ratios(results["base"], results["gold"])
    rec["noise"] = ratios(results["base"], results["base2"])
    w, n = rec["wall"], rec["noise"]
    print(f"  speedup  median={w['median']}x best={w['best']}x worst={w['worst']}x "
          f"({w['n_above_1_20']}/{w['n']} >= 1.20x)")
    if n:
        print(f"  noise    median={n['median']}x best={n['best']}x  (base vs base)")

    if args.ir:
        driver = Path(args.ir).resolve()
        ir = {}
        for side, wt in (("base", base_wt), ("gold", gold_wt)):
            hi = callgrind_ir(pys[side], wt, driver, args.ir_reps, use_pp)
            lo = callgrind_ir(pys[side], wt, driver, 0, use_pp)
            ir[side] = hi - lo
            print(f"  ir {side:<5} {ir[side]:,} instructions (import cancelled)")
        rec["ir"] = {"driver": driver.name, "reps": args.ir_reps, **ir,
                     "speedup": round(ir["base"] / ir["gold"], 4)}
        print(f"  ir speedup = {rec['ir']['speedup']}x")

    verdict = "reject_below_noise"
    if rec["wall"]["median"] >= 1.20 or rec["wall"]["best"] >= 1.50:
        verdict = "strong"
    elif rec["wall"]["n_above_1_20"] >= 1:
        verdict = "usable_subset"
    elif rec.get("ir", {}).get("speedup", 1) >= 1.01:
        verdict = "ir_only"
    rec["status"], rec["verdict"] = "measured", verdict
    print(f"  verdict: {verdict}")
    return rec


def cmd_measure(args) -> None:
    if not CANDIDATES.exists():
        raise SystemExit("run `mine_perf.py harvest` first")
    cands = [json.loads(l) for l in CANDIDATES.read_text().splitlines() if l.strip()]
    if args.pr:
        cands = [c for c in cands if c["pr"] in set(args.pr)]
    if not cands:
        raise SystemExit("no matching candidates")

    done = {}
    if MEASURED.exists():
        done = {json.loads(l)["pr"]: json.loads(l)
                for l in MEASURED.read_text().splitlines() if l.strip()}
    for cand in cands:
        if cand["pr"] in done and not args.pr:
            continue
        try:
            done[cand["pr"]] = measure_one(cand, args)
        except Exception as exc:  # noqa: BLE001
            print(f"  ERROR {type(exc).__name__}: {exc}")
            done[cand["pr"]] = {"pr": cand["pr"], "title": cand["title"],
                                "status": "error", "detail": f"{type(exc).__name__}: {exc}"}
        MEASURED.write_text("".join(json.dumps(done[k]) + "\n"
                                    for k in sorted(done, reverse=True)))

    print(f"\nwrote {len(done)} records -> {MEASURED.name}")
    for v in ("strong", "usable_subset", "ir_only", "reject_below_noise"):
        hits = [r["pr"] for r in done.values() if r.get("verdict") == v]
        if hits:
            print(f"  {v:<20} {len(hits):>3}  {sorted(hits, reverse=True)}")


# ---------------------------------------------------------------------------
# calibrate -- measure each shipped task's own driver on base vs gold
# ---------------------------------------------------------------------------
#
# Every perf task ships one self-contained driver that prints elapsed
# milliseconds, so the metric is era-independent and does not depend on the
# repo's own benchmark files existing at the base commit (they often do not).
#
# Writing the driver is the part that needs care. A first attempt at 004/005/006
# invented plausible-looking workloads and measured 1.19x / 1.11x / 1.05x --
# barely above noise -- because they built fresh classes each iteration, so
# class-creation overhead diluted the thing being optimised. Rebuilding them
# around model_rebuild(force=True) on a fixed set of models, the way the repo's
# own benchmarks do, recovers the real signal (1.27x / 1.48x / 1.21x). A perf
# task is only as good as its driver.

TASK_DRIVERS = {
    "perf-001": (13523, "perf_001.py"),  # exponential -> linear core-schema traversal
    "perf-002": (7947, "perf_002.py"),   # lazy package imports
    "perf-003": (10868, "perf_003.py"),  # cache __setattr__ setters
    "perf-004": (11244, "perf_004.py"),  # schema cleaning refactor
    "perf-005": (13573, "perf_005.py"),  # type lookup in schema generation
    "perf-006": (10863, "perf_006.py"),  # get_type_ref
}


def cmd_calibrate(args) -> None:
    cands = {c["pr"]: c for c in
             (json.loads(l) for l in CANDIDATES.read_text().splitlines() if l.strip())}
    results = json.loads(CALIBRATION.read_text()) if CALIBRATION.exists() else {}

    for task_id, (pr, driver_name) in TASK_DRIVERS.items():
        if args.only and task_id not in args.only:
            continue
        cand = cands.get(pr)
        if cand is None:
            print(f"{task_id}: PR{pr} not in candidates -- run harvest first")
            continue
        print(f"\n=== {task_id}  PR{pr}  {cand['title'][:52]}")
        try:
            python = era_venv(cand["core_pin"]) / "bin/python"
            sides = {}
            for side, commit in (("base", cand["base_commit"]),
                                 ("gold", cand["gold_commit"])):
                wt = worktree(f"{pr}-{side}", commit)
                ok, err = canary_ok(python, wt, True)
                if not ok:
                    raise RuntimeError(f"canary failed on {side}: {err}")
                best, vals = run_driver(python, wt, DRIVERS / driver_name, args.reps)
                sides[side] = {"min_ms": round(best, 4),
                               "median_ms": round(statistics.median(vals), 4)}
                print(f"  {side:<5} min={best:10.2f}ms")
            # Noise floor: base measured a second time, same driver, same venv.
            base_again, _ = run_driver(python, worktree(f"{pr}-base", cand["base_commit"]),
                                       DRIVERS / driver_name, args.reps)
            speedup = sides["base"]["min_ms"] / sides["gold"]["min_ms"]
            noise = (max(base_again, sides["base"]["min_ms"])
                     / min(base_again, sides["base"]["min_ms"]))
            results[task_id] = {
                "pr": pr, "title": cand["title"], "driver": driver_name,
                "base_commit": cand["base_commit"], "gold_commit": cand["gold_commit"],
                "core_pin": cand["core_pin"], "reps": args.reps,
                "base_ms": sides["base"]["min_ms"], "gold_ms": sides["gold"]["min_ms"],
                "gold_speedup": round(speedup, 4), "noise_ratio": round(noise, 4),
                "status": "ok"}
            print(f"  gold speedup = {speedup:.3f}x   (noise floor {noise:.4f}x)")
        except Exception as exc:  # noqa: BLE001
            print(f"  ERROR {type(exc).__name__}: {exc}")
            results[task_id] = {"pr": pr, "driver": driver_name, "status": "error",
                                "detail": f"{type(exc).__name__}: {exc}"}
        CALIBRATION.write_text(json.dumps(results, indent=1) + "\n")
    print(f"\nwrote {CALIBRATION}")


# ---------------------------------------------------------------------------
# assemble -- emit the task dirs
# ---------------------------------------------------------------------------
#
# Prompts state the **symptom only**: what a user could report without reading
# the source -- the observable symptom, the shape and scale of the workload, the
# desired outcome, and the behavior that must not change. They never name the
# library, never say the tree was rewound, and never imply a reference fix exists.
# The numeric target is stated as a bare requirement, which gives an unambiguous
# bar without revealing where the number came from.
#
# Getting this wrong is easy. Each prompt originally ended with a sentence
# explaining *why* the code was slow ("repeated once per path through the graph
# of references", "recomputing string representations that are stable for a given
# type"). Every one reads naturally -- it is what a maintainer writes *after*
# finding the fix -- and every one converts the task from "diagnose and optimise"
# into "apply the change I just described". They are cut.

# Fraction of the maintainers' improvement an agent must reach to pass the speed
# gate. Under 1.0 purely as slack for measurement noise, so a solution that is
# materially equivalent but a few percent slower is not failed by the noise floor.
# It is not partial credit: below the threshold the score is 0.
SPEED_TOLERANCE = 0.8

INSTRUCTIONS = """\
The repository at /repo is a Python library. Part of it is slower than it should
be; the report below describes which behavior is slow. Make it faster by
changing the library source.

A benchmark for the slow behavior is at /repo/perf_bench.py. Run
`python perf_bench.py`: it prints a single number, elapsed milliseconds, lower
is better. Use it to check your progress. Do NOT modify perf_bench.py or
anything under tests/ -- both are restored to their original state before
scoring, so edits there are discarded.

Your change is scored on three things, all of which must hold:

1. Correctness. Every test that passes now must still pass. Breaking even one
   scores zero, as does a change that crashes the test suite. There is no
   partial credit here: keep behavior identical.
   Do NOT run the test suite yourself -- it is slow, and running it will consume
   the budget you need for the actual work. Reason about correctness from the
   code you are changing, and if you want to check something, run individual
   test functions, not files or directories.
2. Speed. perf_bench.py must become at least {threshold:.2f}x faster than it is
   now. Anything short of that scores zero -- there is no partial credit for
   getting part of the way.
3. No displaced cost. Other benchmarks in the repository, which you are not
   shown, must not get materially slower. Making the measured path fast by
   moving work elsewhere scores zero.

Keep the public API and observable behavior unchanged."""

# Blind variant: no benchmark. The agent must build its own repro from the report,
# profile, and decide when it is done. This removes the object an agent could
# overfit to -- with a benchmark in hand, "make this number drop" is satisfiable by
# special-casing the measured shape, and nothing in the score distinguishes that
# from a general fix. It costs feedback, so run both variants and compare: same
# base commit, same driver at grade time, same threshold.
INSTRUCTIONS_BLIND = """\
The repository at /repo is a Python library. Part of it is slower than it should
be; the report below describes which behavior is slow. Make it faster by
changing the library source.

There is no benchmark in the repository for this. Reproduce the slow behavior
from the report yourself, and profile it, to decide what to change and when you
are done. Do NOT modify anything under tests/ -- it is restored to its original
state before scoring, so edits there are discarded.

Your change is scored on three things, all of which must hold:

1. Correctness. Every test that passes now must still pass. Breaking even one
   scores zero, as does a change that crashes the test suite. There is no
   partial credit here: keep behavior identical.
   Do NOT run the test suite yourself -- it is slow, and running it will consume
   the budget you need for the actual work. Reason about correctness from the
   code you are changing, and if you want to check something, run individual
   test functions, not files or directories.
2. Speed. The behavior described below is timed on a fixed workload you are not
   shown, and must become at least {threshold:.2f}x faster than it is now.
   Anything short of that scores zero -- there is no partial credit for getting
   part of the way.
3. No displaced cost. Other benchmarks in the repository, which you are also not
   shown, must not get materially slower. Making one path fast by moving work
   elsewhere scores zero.

Keep the public API and observable behavior unchanged."""

PROMPTS = {
    "perf-001": """\
Creating a group of models that reference one another is disproportionately slow.

Build a chain of models where each new model has a field pointing at each of the
previous few models -- around twenty models, each of them tiny. Creating that
chain takes *seconds*. Worse, adding models makes it dramatically rather than
gradually worse: twenty-five is far worse than twenty, and thirty does not finish
in any reasonable time.

No individual model here is large or unusual, and once they exist they validate
and serialize fine. It is creating them that is slow.

Creating a group of interconnected models like these should not blow up as the
group grows.""",
    "perf-002": """\
Importing the library is slow.

A program that imports it and then does almost nothing still spends tens of
milliseconds inside the import itself, before any useful work starts. For
short-lived processes -- CLI tools, test collection, serverless handlers, shell
completions -- that import dominates total runtime.

Reduce the cost of importing the package. Everything that can be imported from
the package today must still be importable from exactly the same place, with the
same behavior.""",
    "perf-003": """\
Setting an attribute on a model instance is slow.

A loop that repeatedly assigns to a few fields of an already-constructed model is
far slower than the work involved warrants, and most of the time is not spent
validating the values being assigned.

Make attribute assignment on model instances faster. Assignment must keep
behaving exactly as it does now: validation on assignment where configured,
private attributes, cached properties, frozen models, and errors for unknown
attributes.""",
    "perf-004": """\
Preparing a model's internal schema is slower than it should be for models whose
annotations involve nesting, recursion, or discriminated unions.

Defining many such model classes -- which is what a large codebase does at import
time -- spends a disproportionate amount of time in that preparation, out of
proportion to how large or complicated the models actually are.

Make preparing the internal schema faster for models like these.""",
    "perf-005": """\
Preparing the internal schema for a model with a large number of fields is slower
than it should be.

Take a model with a hundred fields that all share the same annotation. Preparing
its schema costs roughly a hundred times what a single field costs, and the cost
tracks the number of fields rather than how many distinct types are involved.

Make preparing the internal schema faster for models with many fields.""",
    "perf-006": """\
Preparing the internal schema for models with deeply parameterized annotations is
slower than it should be.

Annotations that nest containers, unions, and optionals inside one another --
`list[dict[str, Union[int, float]]]`, `Optional[list[Union[str, int]]]`, fields
referring to other models and to containers of other models -- cost noticeably
more to prepare than their size suggests.

Make preparing the internal schema faster for models with annotations like
these.""",
}


def pytest_pin(commit: str) -> str:
    """The test recipe has to be era-dependent.

    mine.py pins pytest==8.3.5 because >=8.4 turns 2024-era generator parametrize
    into collection errors -- but 2026-era tests call pytest.raises(..., check=...),
    which only exists in >=8.4. One pin cannot span both, so choose by the age of
    the base commit.
    """
    date = git_checked("show", "-s", "--format=%ad", "--date=format:%Y-%m", commit).strip()
    return "pytest>=8.4" if date >= "2025-09" else "pytest==8.3.5"


def cmd_assemble(args) -> None:
    if not CALIBRATION.exists():
        raise SystemExit("run `mine_perf.py calibrate` first")
    calib = json.loads(CALIBRATION.read_text())
    TASKS_DIR.mkdir(exist_ok=True)
    ANSWERS_DIR.mkdir(exist_ok=True)
    emitted = []

    for task_id in sorted(calib):
        rec = calib[task_id]
        if rec.get("status") != "ok" or task_id not in PROMPTS:
            print(f"{task_id}: skipped ({rec.get('status')})")
            continue

        gold_speedup = rec["gold_speedup"]
        threshold = 1 + SPEED_TOLERANCE * (gold_speedup - 1)
        if threshold - 1 <= 2 * (rec["noise_ratio"] - 1):
            print(f"{task_id}: WARNING threshold {threshold:.3f}x is within 2x the "
                  f"noise floor {rec['noise_ratio']:.4f}x -- grade with more reps")

        patch = git_checked("diff", rec["base_commit"], rec["gold_commit"], "--", "pydantic/")
        src_files = sorted({l.split(" b/")[-1].strip()
                            for l in patch.splitlines() if l.startswith("+++ b/")})

        variants = [(task_id, True)] + ([(f"{task_id}-blind", False)] if args.blind else [])
        for vid, ships_benchmark in variants:
            task_dir, ans_dir = TASKS_DIR / vid, ANSWERS_DIR / vid
            if task_dir.exists() and not args.force:
                print(f"{vid}: exists, skipping (--force to overwrite)")
                continue
            task_dir.mkdir(parents=True, exist_ok=True)
            ans_dir.mkdir(parents=True, exist_ok=True)
            (task_dir / "task.json").write_text(json.dumps({
                "id": vid,
                "category": "performance",
                "type": "perf",
                "base_commit": rec["base_commit"],
                "ships_benchmark": ships_benchmark,
                "instructions": (INSTRUCTIONS if ships_benchmark
                                 else INSTRUCTIONS_BLIND).format(threshold=threshold),
                "prompt": PROMPTS[task_id],
            }, indent=1) + "\n")
            if ships_benchmark:
                shutil.copyfile(DRIVERS / rec["driver"], task_dir / "perf_bench.py")
            (ans_dir / "answer.json").write_text(json.dumps({
                "id": vid,
                "type": "perf",
                "ships_benchmark": ships_benchmark,
                "pr_number": rec["pr"],
                "pr_url": f"https://github.com/pydantic/pydantic/pull/{rec['pr']}",
                "pr_title": rec["title"],
                "base_commit": rec["base_commit"],
                "gold_commit": rec["gold_commit"],
                "code_patch": patch,
                "gold_loc_files": src_files,
                "driver": rec["driver"],
                "core_pin": rec["core_pin"],
                "pytest_pin": pytest_pin(rec["base_commit"]),
                "calibration": {"base_ms": rec["base_ms"], "gold_ms": rec["gold_ms"],
                                "gold_speedup": gold_speedup,
                                "noise_ratio": rec["noise_ratio"], "reps": rec["reps"]},
                "threshold_speedup": round(threshold, 4),
                "speed_tolerance": SPEED_TOLERANCE,
            }, indent=1) + "\n")
            emitted.append(vid)
            print(f"{vid:<16} PR{rec['pr']:<6} gold={gold_speedup:>7.3f}x  "
                  f"threshold={threshold:>7.3f}x  "
                  f"bench={'shipped' if ships_benchmark else 'withheld'}")

    print(f"\n{len(emitted)} perf tasks -> {TASKS_DIR}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="stage", required=True)

    h = sub.add_parser("harvest", help="perf-labelled PRs -> work/perf_candidates.jsonl")
    h.add_argument("--refresh", action="store_true", help="re-query GitHub, ignoring cache")
    h.set_defaults(fn=cmd_harvest)

    m = sub.add_parser("measure", help="triage candidates on the repo's benchmarks")
    m.add_argument("--pr", type=int, nargs="*", help="only these PR numbers")
    m.add_argument("--bench", nargs="*", default=DEFAULT_BENCH)
    m.add_argument("--install", action="store_true",
                   help="real editable install per worktree (builds pydantic-core)")
    m.add_argument("--ir", metavar="DRIVER",
                   help="also measure callgrind instructions on DRIVER (needs valgrind)")
    m.add_argument("--ir-reps", type=int, default=20)
    m.set_defaults(fn=cmd_measure)

    c = sub.add_parser("calibrate", help="measure each task's driver on base vs gold")
    c.add_argument("--only", nargs="*", help="task ids to measure")
    c.add_argument("--reps", type=int, default=5, help="process repetitions per side")
    c.set_defaults(fn=cmd_calibrate)

    a = sub.add_parser("assemble", help="-> tasks_perf/ + answers_perf/")
    a.add_argument("--force", action="store_true", help="overwrite existing task dirs")
    a.add_argument("--blind", action="store_true",
                   help="also emit <id>-blind variants that ship no benchmark")
    a.set_defaults(fn=cmd_assemble)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
