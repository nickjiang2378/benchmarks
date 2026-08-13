#!/usr/bin/env python3
"""Category 1, general feature / spec implementation: implement the PR, pass its tests.

Given the issue or PR description, the agent must make the change; it is scored
on the percentage of the PR's own test cases that pass. Everything here builds
that category out of pydantic's merged PRs.

Each stage is idempotent and resumable -- rerunning skips records already
processed -- and each writes an inspectable JSONL artifact plus a reason-coded
discard log, so a low yield tells you *which* filter to loosen instead of
leaving you to guess.

    harvest    GitHub GraphQL  ->  work/candidates.jsonl    (1550 -> 491)
    extract    local git       ->  work/extracted.jsonl     (491 -> 470)
    validate   pytest          ->  work/validated.jsonl     (470 -> 342)
    assemble   pure python     ->  tasks/ + answers/
    clean      small LLM       ->  tasks/ (in place)        (boilerplate deleted)
    judge      LLM judge       ->  work/fairness_tasks.json (fair / unfair)
    filter     pure python     ->  work/ids_<name>.json     (the shipped set)

    python3 mine.py harvest              # needs GITHUB_TOKEN
    python3 mine.py extract
    python3 mine.py validate --workers 4 # the expensive one, ~40 min at 4 cores
    python3 mine.py assemble
    python3 mine.py clean                # needs ANTHROPIC_API_KEY
    python3 mine.py judge
    # then probe with a cheap model (run_agent.py + grade.py), and:
    python3 mine.py filter --probe-run haiku-probe

The last two stages are what make the category unsaturated and answerable:
`judge` rates whether the hidden tests are inferable from the prompt at all, and
`filter` keeps only the tasks a cheap model could not already solve.

This module also holds the helpers the other scripts import (paths, JSONL I/O,
the introspection blocklist), so there is one definition of each.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# ---------------------------------------------------------------------------
# shared paths and helpers
# ---------------------------------------------------------------------------

HERE = Path(__file__).resolve().parent
WORK = HERE / "work"
CLONE_DIR = HERE / "pydantic-clone"
TASKS_DIR = HERE / "tasks"
ANSWERS_DIR = HERE / "answers"
REPO = "pydantic/pydantic"

WORK.mkdir(exist_ok=True)

# A mutation-killing test suite is only interesting if it detects behavior. A
# test that reads pydantic's own source (or its bytecode, or a hash of it) kills
# every variant vacuously without asserting anything about behavior, so both the
# variant validator and the grader reject suites that do it.
INTROSPECTION_PATTERNS = [
    (r"\binspect\.(getsource|getsourcelines|getsourcefile|getfile)\b", "reads source text"),
    (r"\bdis\.(dis|get_instructions|Bytecode)\b", "reads bytecode"),
    (r"\bast\.parse\b", "parses source"),
    (r"\b(subprocess|os\.system|os\.popen|pty\.spawn)\b", "shells out"),
    (r"\bopen\s*\(\s*[^)]*(pydantic[/\\]|__file__)", "opens a source file"),
    (r"\b(Path|pathlib\.Path)\s*\([^)]*__file__[^)]*\)\s*\.\s*read_text", "reads a source file"),
    (r"\.__code__\b", "inspects code objects"),
    (r"\b(hashlib|importlib\.metadata\.version)\b.*\bpydantic\b", "fingerprints the install"),
]


def forbidden_introspection(text: str) -> list[str]:
    """Reasons `text` (a test module) is disqualified as a behavioral test."""
    return [why for pat, why in INTROSPECTION_PATTERNS
            if re.search(pat, text, re.MULTILINE)]


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def append_jsonl(path: Path, rec: dict) -> None:
    with path.open("a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def log_discard(path: Path, key, reason: str, **extra) -> None:
    append_jsonl(path, {"key": key, "reason": reason, **extra})


def load_env(name: str) -> str:
    """Read a credential from the environment, falling back to ./.env."""
    val = os.environ.get(name)
    if not val and (HERE / ".env").exists():
        for line in (HERE / ".env").read_text().splitlines():
            if line.startswith(f"{name}="):
                val = line.split("=", 1)[1].strip()
    if not val:
        sys.exit(f"{name} not found in the environment or ./.env")
    return val


def sh(cmd, cwd=None, timeout=600, env=None) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True,
                          timeout=timeout, env=env)


def git(*args: str) -> str:
    """Run git in the clone and return stdout, raising on failure."""
    return subprocess.run(["git", "-C", str(CLONE_DIR), *args],
                          capture_output=True, text=True, check=True).stdout


# ---------------------------------------------------------------------------
# stage 1 -- harvest candidate PRs from the GitHub API (metadata only)
# ---------------------------------------------------------------------------
#
# One GraphQL search per half-year window since 2024-01-01; the windows exist to
# dodge the 1000-result cap on a single search. Each PR node carries its files,
# line counts, author, merge commit and linked issues (title + body), so no later
# stage ever touches the network.

RAW = WORK / "harvest_raw.jsonl"
CANDIDATES = WORK / "candidates.jsonl"
EXTRACTED = WORK / "extracted.jsonl"
VALIDATED = WORK / "validated.jsonl"
WINDOWS_DONE = WORK / "harvest_windows_done.txt"

WINDOWS = [
    ("2024-01-01", "2024-06-30"),
    ("2024-07-01", "2024-12-31"),
    ("2025-01-01", "2025-06-30"),
    ("2025-07-01", "2025-12-31"),
    ("2026-01-01", "2026-08-12"),
]

QUERY = """
query($q: String!, $cursor: String) {
  search(query: $q, type: ISSUE, first: 50, after: $cursor) {
    issueCount
    pageInfo { hasNextPage endCursor }
    nodes {
      ... on PullRequest {
        number title body mergedAt url
        author { login __typename }
        additions deletions changedFiles
        mergeCommit { oid parents(first: 3) { totalCount } }
        files(first: 30) { nodes { path additions deletions } }
        closingIssuesReferences(first: 3) { nodes { number title body } }
      }
    }
  }
}
"""

BOT_LOGINS = {"dependabot", "pre-commit-ci", "github-actions", "renovate"}
MIN_FILES, MAX_FILES = 1, 15
MIN_LINES, MAX_LINES = 5, 600


def graphql(token: str, query: str, variables: dict, retries: int = 5) -> dict:
    body = json.dumps({"query": query, "variables": variables}).encode()
    for attempt in range(retries):
        req = urllib.request.Request(
            "https://api.github.com/graphql", data=body,
            headers={"Authorization": f"Bearer {token}",
                     "Content-Type": "application/json",
                     "User-Agent": "pydantic-bench-miner"})
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read())
            if "errors" in data and not data.get("data"):
                raise RuntimeError(f"GraphQL errors: {data['errors']}")
            return data
        except Exception as e:  # noqa: BLE001 - retry any transient failure
            if attempt == retries - 1:
                raise
            wait = 2 ** (attempt + 1)
            print(f"  graphql retry in {wait}s: {e}", file=sys.stderr)
            time.sleep(wait)
    raise RuntimeError("unreachable")


def fetch_all() -> list[dict]:
    token = load_env("GITHUB_TOKEN")
    done = set(WINDOWS_DONE.read_text().split()) if WINDOWS_DONE.exists() else set()
    for start, end in WINDOWS:
        wid = f"{start}..{end}"
        if wid in done:
            print(f"window {wid}: already harvested, skipping")
            continue
        q = f"repo:{REPO} is:pr is:merged merged:{start}..{end}"
        cursor, seen = None, 0
        while True:
            s = graphql(token, QUERY, {"q": q, "cursor": cursor})["data"]["search"]
            if seen == 0:
                print(f"window {wid}: {s['issueCount']} merged PRs")
                assert s["issueCount"] < 1000, f"window {wid} exceeds search cap"
            for node in s["nodes"]:
                if node:
                    node["_window"] = wid
                    append_jsonl(RAW, node)
                    seen += 1
            if not s["pageInfo"]["hasNextPage"]:
                break
            cursor = s["pageInfo"]["endCursor"]
        print(f"window {wid}: harvested {seen}")
        with WINDOWS_DONE.open("a") as f:
            f.write(wid + "\n")
    return read_jsonl(RAW)


def classify(pr: dict) -> str | None:
    """Return a discard reason code, or None if the PR is a keeper."""
    author = pr.get("author") or {}
    login = (author.get("login") or "").lower()
    if author.get("__typename") == "Bot" or any(b in login for b in BOT_LOGINS):
        return "bot_author"
    files = [f for f in (pr.get("files") or {}).get("nodes", []) if f]
    if pr["changedFiles"] > len(files):
        # files(first:30) truncated -- only possible when changedFiles > 30 > MAX_FILES
        return "too_many_files"
    paths = [f["path"] for f in files]
    if not [p for p in paths if p.startswith("tests/")]:
        return "no_test_changes"
    if not [p for p in paths if p.startswith("pydantic/") and p.endswith(".py")]:
        return "no_source_changes"
    if not (MIN_FILES <= pr["changedFiles"] <= MAX_FILES):
        return "file_count_out_of_range"
    lines = sum(f["additions"] + f["deletions"] for f in files
                if f["path"].startswith(("pydantic/", "tests/")))
    if lines < MIN_LINES:
        return "diff_too_small"
    if lines > MAX_LINES:
        return "diff_too_large"
    if not pr.get("mergeCommit"):
        return "no_merge_commit"
    return None


def cmd_harvest(args) -> None:
    discards = WORK / "discards_harvest.jsonl"
    raw = fetch_all()
    seen_nums, kept, reasons = set(), 0, Counter()
    CANDIDATES.write_text("")
    discards.write_text("")
    for pr in sorted(raw, key=lambda p: p["number"]):
        if pr["number"] in seen_nums:
            continue
        seen_nums.add(pr["number"])
        reason = classify(pr)
        if reason:
            reasons[reason] += 1
            log_discard(discards, pr["number"], reason, title=pr.get("title", ""))
            continue
        issues = [i for i in (pr.get("closingIssuesReferences") or {}).get("nodes", []) if i]
        append_jsonl(CANDIDATES, {
            "pr_number": pr["number"],
            "pr_title": pr["title"],
            "pr_body": pr.get("body") or "",
            "merged_at": pr["mergedAt"],
            "url": pr["url"],
            "author": (pr.get("author") or {}).get("login"),
            "merge_commit": pr["mergeCommit"]["oid"],
            "merge_commit_parents": pr["mergeCommit"]["parents"]["totalCount"],
            "changed_files": [f for f in pr["files"]["nodes"] if f],
            "additions": pr["additions"],
            "deletions": pr["deletions"],
            "issues": [{"number": i["number"], "title": i["title"],
                        "body": i.get("body") or ""} for i in issues],
        })
        kept += 1
    print(f"\n{len(seen_nums)} merged PRs -> {kept} candidates")
    for reason, count in reasons.most_common():
        print(f"  discard {reason}: {count}")
    n_issue = sum(1 for r in read_jsonl(CANDIDATES) if r["issues"])
    print(f"candidates with linked issue: {n_issue}/{kept}")


# ---------------------------------------------------------------------------
# stage 2 -- extract patches and prompt text from the local clone (no network)
# ---------------------------------------------------------------------------
#
# Verified assumption: pydantic squash-merges, so base = merge_commit^ and the
# gold patch is the merge commit's own diff, split by path into a code patch
# (everything except tests/) and a test patch (tests/ only).
#
# Prompts are the linked issue's title + body verbatim, scrubbed only of things
# that leak the fix: links or mentions of the fixing PR/commit, and bare 40-hex
# SHAs. Code snippets and tracebacks are kept -- real engineers get those. Issue
# *comment threads* are excluded entirely: that is where a maintainer posts the
# fix inline.

MAX_PROMPT_CHARS = 10_000
SHA_RE = re.compile(r"\b[0-9a-f]{40}\b")
CHECKLIST_RE = re.compile(r"^\s*[-*]\s*\[[ xX]\]\s.*$", re.MULTILINE)
REVIEWER_RE = re.compile(r"^\s*Selected Reviewer:.*$", re.MULTILINE | re.IGNORECASE)


def scrub(text: str, pr_number: int) -> str:
    """Remove references that leak the fixing PR/commit; keep code and tracebacks."""
    text = text.replace("\r\n", "\n")
    text = re.sub(r"https?://github\.com/pydantic/pydantic/(?:pull|commit)/\S+", "", text)
    text = re.sub(rf"(?<![\w#])#{pr_number}\b", "", text)
    text = SHA_RE.sub("", text)
    if len(text) > MAX_PROMPT_CHARS:
        text = text[:MAX_PROMPT_CHARS] + "\n\n[... truncated ...]"
    return text.strip()


def scrub_pr_body(body: str, pr_number: int) -> str:
    """PR bodies additionally carry template boilerplate; drop it."""
    body = CHECKLIST_RE.sub("", body)
    body = REVIEWER_RE.sub("", body)
    body = re.sub(r"(?im)^change summary:?\s*$", "", body)
    body = re.sub(r"(?im)^related issue number:?.*$", "", body)
    body = re.sub(r"(?i)\b(fix(es|ed)?|close[sd]?|resolve[sd]?)\s+#\d+", "", body)
    return scrub(body, pr_number)


def extract_one(cand: dict) -> tuple[dict | None, str | None]:
    """Return (record, None) on success or (None, discard_reason)."""
    if cand["merge_commit_parents"] != 1:
        return None, "non_squash_merge"
    sha = cand["merge_commit"]
    try:
        git("cat-file", "-e", f"{sha}^{{commit}}")
    except subprocess.CalledProcessError:
        return None, "commit_not_in_clone"
    # Keep environments predictable: only main-line history.
    on_main = subprocess.run(
        ["git", "-C", str(CLONE_DIR), "merge-base", "--is-ancestor", sha, "origin/main"],
        capture_output=True).returncode == 0
    if not on_main:
        return None, "not_on_main"
    base = git("rev-parse", f"{sha}^").strip()
    test_patch = git("diff", base, sha, "--", "tests/")
    code_patch = git("diff", base, sha, "--", ".", ":(exclude)tests/")
    if not test_patch.strip():
        return None, "empty_test_patch"
    if not code_patch.strip():
        return None, "empty_code_patch"
    test_files = re.findall(r"^\+\+\+ b/(tests/\S+)", test_patch, re.MULTILINE)
    if not test_files:
        return None, "test_patch_only_deletes"

    if cand["issues"]:
        issue = cand["issues"][0]
        source, title = "issue", issue["title"]
        body = scrub(issue["body"], cand["pr_number"])
    else:
        source, title = "pr_body", cand["pr_title"]
        body = scrub_pr_body(cand["pr_body"], cand["pr_number"])
    if len(body) < 20 and source == "pr_body":
        return None, "prompt_too_thin"

    return {
        "pr_number": cand["pr_number"],
        "pr_title": cand["pr_title"],
        "url": cand["url"],
        "merged_at": cand["merged_at"],
        "base_commit": base,
        "merge_commit": sha,
        "code_patch": code_patch,
        "test_patch": test_patch,
        "test_files": sorted(set(test_files)),
        "code_files": sorted(set(re.findall(r"^\+\+\+ b/(\S+)", code_patch, re.MULTILINE))),
        "prompt_source": source,
        "prompt_title": title,
        "prompt_body": body,
        "issue_numbers": [i["number"] for i in cand["issues"]],
    }, None


def cmd_extract(args) -> None:
    discards = WORK / "discards_extract.jsonl"
    cands = read_jsonl(CANDIDATES)
    done = {r["pr_number"] for r in read_jsonl(EXTRACTED)}
    done |= {d["key"] for d in read_jsonl(discards)}
    reasons, kept = Counter(), len(read_jsonl(EXTRACTED))
    for cand in cands:
        if cand["pr_number"] in done:
            continue
        rec, reason = extract_one(cand)
        if reason:
            reasons[reason] += 1
            log_discard(discards, cand["pr_number"], reason, title=cand["pr_title"])
            continue
        append_jsonl(EXTRACTED, rec)
        kept += 1
    print(f"{len(cands)} candidates -> {kept} extracted")
    for reason, count in reasons.most_common():
        print(f"  discard {reason}: {count}")
    print("prompt sources:",
          dict(Counter(r["prompt_source"] for r in read_jsonl(EXTRACTED))))


# ---------------------------------------------------------------------------
# stage 3 -- validate by actually flipping the tests
# ---------------------------------------------------------------------------
#
# Per candidate, in a disposable worktree with its own uv venv:
#   run1  base + test patch only  -> failures are the F2P candidate set
#   run2  + code patch            -> F2P candidates must pass
#   run3  repeat run2             -> flake check
#
# F2P = failed in run1 (or its file failed *collection*, e.g. tests importing a
# not-yet-existing symbol), passed in run2 AND run3. P2P = passed in all three.
# Tests failing in every run are environment drift: excluded from both sets and
# counted. Any F2P candidate whose two post-fix runs disagree discards the whole
# candidate. Outcomes come from an injected pytest plugin writing exact node IDs
# as JSON (report_plugin.py) -- no output parsing.
#
# pytest is pinned to 8.3.5: >=8.4 raises PytestRemovedIn10Warning for
# generator-based parametrize, which pydantic's `filterwarnings = error` turns
# into collection errors at 2024-era commits.

WT_ROOT = WORK / "wt"
TEST_DEPS = [
    "pytest==8.3.5", "pytest-mock", "dirty-equals", "cloudpickle", "email-validator",
    "faker", "pytest-benchmark", "pytest-examples", "eval-type-backport",
    "jsonschema", "packaging", "rich", "pytest-run-parallel", "pytz",
]
PYTEST_TIMEOUT = 900
PYTHON = "3.12"

_wt_lock = threading.Lock()
_log_lock = threading.Lock()


def worktree_add(path, sha) -> None:
    with _wt_lock:
        r = sh(["git", "-C", str(CLONE_DIR), "worktree", "add", "--detach",
                str(path), sha], timeout=120)
    if r.returncode != 0:
        raise RuntimeError(f"worktree add failed: {r.stderr[-500:]}")


def worktree_remove(path) -> None:
    with _wt_lock:
        sh(["git", "-C", str(CLONE_DIR), "worktree", "remove", "--force",
            str(path)], timeout=120)
    shutil.rmtree(path, ignore_errors=True)


def run_pytest(wt, test_files, out_json) -> dict | None:
    """Run pytest on test_files, return {nodeid: outcome} or None on timeout."""
    existing = [f for f in test_files if (wt / f).exists()]
    if not existing:
        return {}
    env = os.environ.copy()
    env.update({"PYTHONPATH": str(HERE), "REPORT_JSON": str(out_json),
                "PYTHONHASHSEED": "0", "COLUMNS": "120"})
    cmd = [str(wt / ".venv/bin/python"), "-m", "pytest", "-q", "--tb=no",
           "--no-header", "-p", "no:cacheprovider", "-p", "report_plugin",
           "--continue-on-collection-errors", *existing]
    try:
        sh(cmd, cwd=wt, timeout=PYTEST_TIMEOUT, env=env)
    except subprocess.TimeoutExpired:
        return None
    if not out_json.exists():
        return None
    data = json.loads(out_json.read_text())
    out_json.unlink()
    return data


def collect_error_files(outcomes: dict) -> set:
    return {k.split(":", 1)[1].split("::")[0]
            for k in outcomes if k.startswith("__collect_error__:")}


def validate_one(rec: dict) -> dict:
    """Returns {'pr_number', 'status': 'ok'|'discard', 'reason', ...}."""
    pr = rec["pr_number"]
    # The venv installs pydantic-core as a built wheel, so patching its Rust
    # source silently no-ops.
    if any(f.startswith("pydantic-core/") for f in rec["code_files"]):
        return {"pr_number": pr, "status": "discard", "reason": "touches_rust_core"}
    # Only real pytest modules are runnable: tests/mypy/ holds golden outputs
    # pinned to specific mypy versions, and helper files are not test modules.
    runnable = [f for f in rec["test_files"]
                if not f.startswith("tests/mypy/")
                and re.search(r"(^|/)test_[^/]+\.py$", f)]
    if not runnable:
        return {"pr_number": pr, "status": "discard", "reason": "no_runnable_test_files"}
    rec = {**rec, "test_files": runnable}
    wt = WT_ROOT / f"wt-{pr}"
    worktree_remove(wt)  # clear any stale leftover from a killed run
    t0 = time.time()
    try:
        worktree_add(wt, rec["base_commit"])
        r = sh(["uv", "venv", "--quiet", "--python", PYTHON, ".venv"], cwd=wt, timeout=300)
        if r.returncode != 0:
            return {"pr_number": pr, "status": "discard", "reason": "venv_failed",
                    "detail": r.stderr[-300:]}
        r = sh(["uv", "pip", "install", "--quiet", "--python", ".venv/bin/python",
                "-e", ".", *TEST_DEPS], cwd=wt, timeout=1200)
        if r.returncode != 0:
            return {"pr_number": pr, "status": "discard", "reason": "install_failed",
                    "detail": r.stderr[-300:]}
        install_s = time.time() - t0

        patch_file = wt / "_test.patch"
        patch_file.write_text(rec["test_patch"])
        r = sh(["git", "apply", "--whitespace=nowarn", str(patch_file)], cwd=wt)
        if r.returncode != 0:
            return {"pr_number": pr, "status": "discard",
                    "reason": "test_patch_apply_failed", "detail": r.stderr[-300:]}

        t1 = time.time()
        run1 = run_pytest(wt, rec["test_files"], wt / "r1.json")
        if run1 is None:
            return {"pr_number": pr, "status": "discard", "reason": "pytest_timeout_pre"}
        pre_collect_err = collect_error_files(run1)
        pre_fail = {n for n, o in run1.items()
                    if o in ("failed", "error") and not n.startswith("__collect_error__:")}
        if not pre_fail and not pre_collect_err:
            return {"pr_number": pr, "status": "discard", "reason": "no_failing_tests_pre"}

        patch_file.write_text(rec["code_patch"])
        r = sh(["git", "apply", "--whitespace=nowarn", str(patch_file)], cwd=wt)
        if r.returncode != 0:
            return {"pr_number": pr, "status": "discard",
                    "reason": "code_patch_apply_failed", "detail": r.stderr[-300:]}
        if any(f in ("pyproject.toml", "pdm.lock", "uv.lock") for f in rec["code_files"]):
            r = sh(["uv", "pip", "install", "--quiet", "--python", ".venv/bin/python",
                    "-e", "."], cwd=wt, timeout=1200)
            if r.returncode != 0:
                return {"pr_number": pr, "status": "discard",
                        "reason": "reinstall_failed", "detail": r.stderr[-300:]}

        run2 = run_pytest(wt, rec["test_files"], wt / "r2.json")
        run3 = run_pytest(wt, rec["test_files"], wt / "r3.json") if run2 is not None else None
        if run2 is None or run3 is None:
            return {"pr_number": pr, "status": "discard", "reason": "pytest_timeout_post"}
        test_s = time.time() - t1

        # A file that fails collection post-fix but collected fine pre-fix is
        # suspicious (the gold patch or the env breaks it) -> discard. A file
        # broken in BOTH states is env drift: its tests yield no node IDs in any
        # run, so they fall out of F2P/P2P naturally.
        new_collect_err = (collect_error_files(run2) | collect_error_files(run3)) - pre_collect_err
        if new_collect_err:
            return {"pr_number": pr, "status": "discard", "reason": "collect_error_post_fix",
                    "detail": sorted(new_collect_err)[:3]}

        f2p, p2p, env_broken, flaky = [], [], [], []
        all_nodes = {n for n in (set(run1) | set(run2) | set(run3))
                     if not n.startswith("__collect_error__:")}
        for n in sorted(all_nodes):
            o1, o2, o3 = (run1.get(n, "absent"), run2.get(n, "absent"), run3.get(n, "absent"))
            failed_pre = o1 in ("failed", "error") or n.split("::")[0] in pre_collect_err
            if o2 != o3:
                if failed_pre:
                    flaky.append(n)
                continue  # unstable non-candidate: exclude silently
            if failed_pre and o2 == "passed":
                f2p.append(n)
            elif o1 == "passed" and o2 == "passed":
                p2p.append(n)
            elif o1 == "passed" and o2 in ("failed", "error"):
                return {"pr_number": pr, "status": "discard",
                        "reason": "gold_patch_breaks_test", "detail": n}
            elif failed_pre and o2 in ("failed", "error"):
                env_broken.append(n)
        if flaky:
            return {"pr_number": pr, "status": "discard", "reason": "flaky_f2p",
                    "detail": flaky[:5]}
        if not f2p:
            return {"pr_number": pr, "status": "discard", "reason": "no_f2p",
                    "detail": {"pre_fail": len(pre_fail), "env_broken": len(env_broken)}}
        return {"pr_number": pr, "status": "ok", "f2p": f2p, "p2p": p2p,
                "n_env_broken": len(env_broken),
                "install_seconds": round(install_s, 1),
                "test_seconds": round(test_s, 1)}
    except Exception as e:  # noqa: BLE001
        return {"pr_number": pr, "status": "discard", "reason": "exception",
                "detail": repr(e)[-300:]}
    finally:
        worktree_remove(wt)


def cmd_validate(args) -> None:
    discards = WORK / "discards_validate.jsonl"
    WT_ROOT.mkdir(parents=True, exist_ok=True)
    recs = read_jsonl(EXTRACTED)
    done = {r["pr_number"] for r in read_jsonl(VALIDATED)}
    done |= {d["key"] for d in read_jsonl(discards)}
    pending = [r for r in recs if r["pr_number"] not in done]
    if args.limit:
        pending = pending[: args.limit]
    print(f"{len(recs)} extracted, {len(done)} already processed, "
          f"{len(pending)} to validate ({args.workers}-wide)", flush=True)

    n_ok = n_disc = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(validate_one, r) for r in pending]
        for i, fut in enumerate(as_completed(futs), 1):
            res = fut.result()
            with _log_lock:
                if res["status"] == "ok":
                    n_ok += 1
                    append_jsonl(VALIDATED, res)
                    msg = f"OK f2p={len(res['f2p'])} p2p={len(res['p2p'])}"
                else:
                    n_disc += 1
                    log_discard(discards, res["pr_number"], res["reason"],
                                detail=res.get("detail"))
                    msg = f"discard {res['reason']}"
                eta = (time.time() - t0) / i * (len(pending) - i) / 60
                print(f"[{i}/{len(pending)}] PR#{res['pr_number']}: {msg} "
                      f"(ok={n_ok} disc={n_disc}, eta {eta:.0f}m)", flush=True)
    print(f"\nvalidated ok: {n_ok}, discarded: {n_disc}")


# ---------------------------------------------------------------------------
# stage 4 -- assemble the frozen task directories
# ---------------------------------------------------------------------------
#
# Agent-facing text is deliberately provenance-free: it must not name the
# library, state that the tree was rewound to a commit, or imply that a
# reference fix exists. Any of those tells the agent to look for the answer (in
# git history, in a newer release on PyPI, in its own memory of the project)
# instead of solving the problem. Describe only the repo as it stands, and how
# the work is scored.

N_FIX, N_TEST, N_LOC = 60, 20, 20

FIX_INSTRUCTIONS = """\
The repository at /repo is a Python library. Below is a bug report filed
against it. Modify the library source to resolve the reported problem. Do NOT
modify anything under tests/ -- changes there are ignored by scoring. Your
change is graded by tests that are not present in the repository: partial
credit for the fraction of currently-failing tests your change makes pass,
with a penalty if you break tests that currently pass."""

TEST_INSTRUCTIONS = """\
The repository at /repo is a Python library. Below is a description of a bug
or incorrect behavior in it. The code currently exhibits the WRONG behavior.
Write pytest tests (a new file, or added to existing files under tests/) that
capture the CORRECT behavior described. Do NOT modify the library source --
changes outside tests/ are ignored by scoring. Grading: your tests must FAIL
against the code as it stands and PASS once the described problem is fixed.
Tests that pass in both states, or fail in both states, score zero."""

LOC_INSTRUCTIONS = """\
The repository at /repo is a Python library. Below is a bug report filed
against it. Identify which Python source files must be modified to resolve the
reported problem. Write your answer to /repo/localization.json as a JSON list
of repo-relative file paths, e.g. ["package/module.py",
"package/_internal/helper.py"]. Do not include test files. Scoring is F1
overlap against the set of source files that a correct fix must touch."""


def norm_title(t: str) -> set:
    return {w for w in re.findall(r"[a-z0-9_]+", t.lower())
            if w not in {"the", "a", "an", "in", "of", "to", "for", "with",
                         "fix", "fixes", "when", "on", "and", "is", "not"}}


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a | b else 0.0


def gold_loc_files(rec: dict) -> list:
    return sorted(f for f in rec["code_files"]
                  if f.startswith("pydantic/") and f.endswith(".py"))


def dedupe(recs: list[dict], discards: Path) -> list[dict]:
    """Same linked issue, or same gold files + similar title -> keep the richer."""
    recs = sorted(recs, key=lambda r: (-len(r["f2p"]), r["pr_number"]))
    kept, seen_issues, seen_sigs = [], set(), []
    for r in recs:
        issues = frozenset(r["issue_numbers"])
        if issues and issues & seen_issues:
            log_discard(discards, r["pr_number"], "dup_issue")
            continue
        sig = (frozenset(gold_loc_files(r)), norm_title(r["pr_title"]))
        if any(sig[0] and sig[0] == s[0] and jaccard(sig[1], s[1]) >= 0.6 for s in seen_sigs):
            log_discard(discards, r["pr_number"], "dup_files_title")
            continue
        seen_issues |= issues
        seen_sigs.append(sig)
        kept.append(r)
    return kept


def window(merged_at: str) -> str:
    return f"{merged_at[:4]}H{1 if int(merged_at[5:7]) <= 6 else 2}"


def stratified(recs: list[dict], n: int) -> list[dict]:
    """Round-robin across half-year windows, newest first within each."""
    by_win = defaultdict(list)
    for r in recs:
        by_win[window(r["merged_at"])].append(r)
    for lst in by_win.values():
        lst.sort(key=lambda r: r["merged_at"], reverse=True)
    order = sorted(by_win, reverse=True)
    out = []
    while len(out) < n and any(by_win[w] for w in order):
        for w in order:
            if by_win[w] and len(out) < n:
                out.append(by_win[w].pop(0))
    return out


# `category` is the reporting grouping -- the three categories results are broken
# down by. `type` is the finer distinction the graders dispatch on: `fix` and
# `localize` are scored differently but are both "implement what the description
# asks for", so they report together.
CATEGORY = {"fix": "general_feature", "localize": "general_feature",
            # not one of the three reported categories: superseded by
            # stress_test, which asks the harder version of the same question
            "testwrite": "testwrite"}


def emit(task_id: str, ttype: str, rec: dict, instructions: str) -> None:
    """tasks/<id>/task.json is what the agent sees; answers/<id>/answer.json never
    enters a container."""
    (TASKS_DIR / task_id).mkdir(parents=True, exist_ok=True)
    (ANSWERS_DIR / task_id).mkdir(parents=True, exist_ok=True)
    (TASKS_DIR / task_id / "task.json").write_text(json.dumps({
        "id": task_id,
        "category": CATEGORY[ttype],
        "type": ttype,
        "repo": REPO,
        "base_commit": rec["base_commit"],
        "instructions": instructions,
        "prompt": f"# {rec['prompt_title']}\n\n{rec['prompt_body']}".strip(),
    }, indent=1, ensure_ascii=False))
    (ANSWERS_DIR / task_id / "answer.json").write_text(json.dumps({
        "id": task_id,
        "type": ttype,
        "pr_number": rec["pr_number"],
        "pr_url": rec["url"],
        "merged_at": rec["merged_at"],
        "issue_numbers": rec["issue_numbers"],
        "base_commit": rec["base_commit"],
        "merge_commit": rec["merge_commit"],
        "code_patch": rec["code_patch"],
        "test_patch": rec["test_patch"],
        "f2p": rec["f2p"],
        "p2p": rec["p2p"],
        "test_files": rec["test_files"],
        "gold_loc_files": gold_loc_files(rec),
        "test_seconds": rec.get("test_seconds"),
    }, indent=1, ensure_ascii=False))


def cmd_assemble(args) -> None:
    discards = WORK / "discards_assemble.jsonl"
    extracted = {r["pr_number"]: r for r in read_jsonl(EXTRACTED)}
    validated = [r for r in read_jsonl(VALIDATED) if r["status"] == "ok"]

    # Contamination: drop anything whose base commit appears in SWE-Gym's
    # pydantic instances, so the benchmark is not measuring memorised training
    # data. Absent the blocklist we say so rather than silently skipping it.
    swegym_path = HERE / "swegym_pydantic.json"
    if swegym_path.exists():
        swegym = {r["base_commit"] for r in json.loads(swegym_path.read_text())}
    else:
        swegym = set()
        print(f"WARNING: {swegym_path.name} missing -- no contamination filter applied")

    discards.write_text("")
    for d in (TASKS_DIR, ANSWERS_DIR):
        if d.exists():
            shutil.rmtree(d)

    recs = []
    for v in validated:
        r = {**extracted[v["pr_number"]], **v}
        if r["base_commit"] in swegym:
            log_discard(discards, r["pr_number"], "swegym_overlap")
            continue
        recs.append(r)
    print(f"{len(validated)} validated, {len(recs)} after SWE-Gym exclusion")
    recs = dedupe(recs, discards)
    print(f"{len(recs)} after dedupe")

    with_issue = [r for r in recs if r["prompt_source"] == "issue"]
    no_issue = [r for r in recs if r["prompt_source"] == "pr_body"]

    # Test-writing: issue-less candidates first, since what matters there is the
    # behavior delta rather than the prose. Ranked by prompt substance.
    def tw_rank(r):
        return (-min(len(r["prompt_body"]), 2000), -len(r["f2p"]), r["pr_number"])

    test_tasks = sorted(no_issue, key=tw_rank)[:N_TEST]
    used = {r["pr_number"] for r in test_tasks}
    for r in sorted(with_issue, key=tw_rank):  # top up if there were too few
        if len(test_tasks) >= N_TEST:
            break
        if r["pr_number"] not in used:
            test_tasks.append(r)
            used.add(r["pr_number"])

    # Localization: issue-based, multi-file gold patches strongly preferred --
    # single-file localization is trivially saturated.
    loc_pool = [r for r in with_issue if r["pr_number"] not in used and gold_loc_files(r)]
    loc_tasks = stratified([r for r in loc_pool if len(gold_loc_files(r)) >= 2], N_LOC)
    if len(loc_tasks) < N_LOC:
        loc_tasks += stratified([r for r in loc_pool if len(gold_loc_files(r)) == 1],
                                N_LOC - len(loc_tasks))
    used |= {r["pr_number"] for r in loc_tasks}

    # Fix: the issue-based remainder, stratified across half-year windows so the
    # set balances recency against era diversity.
    fix_tasks = stratified([r for r in with_issue if r["pr_number"] not in used], N_FIX)
    used |= {r["pr_number"] for r in fix_tasks}

    assert len(fix_tasks) == N_FIX, f"only {len(fix_tasks)} fix tasks"
    assert len(test_tasks) == N_TEST, f"only {len(test_tasks)} test tasks"
    assert len(loc_tasks) == N_LOC, f"only {len(loc_tasks)} loc tasks"
    shipped = fix_tasks + test_tasks + loc_tasks
    assert not {r["base_commit"] for r in shipped} & swegym, "SWE-Gym overlap!"
    assert len({r["pr_number"] for r in shipped}) == 100

    for kind, tasks, instr in (("fix", fix_tasks, FIX_INSTRUCTIONS),
                               ("testwrite", test_tasks, TEST_INSTRUCTIONS),
                               ("localize", loc_tasks, LOC_INSTRUCTIONS)):
        for i, rec in enumerate(sorted(tasks, key=lambda r: r["pr_number"]), 1):
            emit(f"pydantic-{kind}-{i:03d}", kind, rec, instr)

    # Everything validated but not shipped is a spare: the swap pool for a task
    # that turns out to be unfair, and the source the suite family draws from.
    spares = sorted(r["pr_number"] for r in recs if r["pr_number"] not in used)
    manifest = {
        "tasks": {kind: [f"pydantic-{kind}-{i:03d}" for i in range(1, n + 1)]
                  for kind, n in (("fix", N_FIX), ("testwrite", N_TEST), ("localize", N_LOC))},
        "spare_pr_numbers": spares,
        "counts": {"validated": len(validated), "after_dedupe": len(recs),
                   "shipped": 100, "spares": len(spares)},
    }
    (WORK / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps(manifest["counts"], indent=1))
    print("fix tasks by window:",
          dict(sorted(Counter(window(r["merged_at"]) for r in fix_tasks).items())))


# ---------------------------------------------------------------------------
# stage 5 -- strip template boilerplate out of the prompts
# ---------------------------------------------------------------------------
#
# Issue and PR bodies carry a lot of ritual: checkbox checklists, empty template
# sections ("### Example Code" / "_No response_"), reviewer lines, sign-offs.
# None of it describes the bug, and all of it costs the agent context.
#
# The cleaning is done by a small model, but the model is not trusted: its output
# is accepted ONLY if every non-blank line it kept appears in the original, in
# order. So it can delete lines and nothing else -- never rephrase, reorder, or
# invent. On a failed check it retries once, then keeps the original untouched.
# Originals are backed up before any write.

CLEAN_MODEL = "claude-haiku-4-5-20251001"

CLEAN_INSTRUCTIONS = """\
You are cleaning a coding-benchmark task description (a GitHub issue or PR \
description). Return the SAME text with irrelevant template boilerplate \
DELETED. You may ONLY delete whole lines; never rephrase, reorder, merge, \
or add anything (deleting entire lines, including blank ones, is the only \
allowed edit).

DELETE:
- Template ritual sections: "Initial Checks" / checklists of checkboxes and \
their section headers
- Empty template sections, e.g. a "### Example Code" header followed by \
"_No response_" (delete both)
- "Selected Assignee" / reviewer lines, HTML comments, "Thanks!" sign-offs, \
contribution/CLA boilerplate
- Section headers that head nothing after deletions

KEEP (never delete):
- The title line and all substantive description text
- ALL code blocks that contain actual code, reproduction snippets, tracebacks, \
error messages, expected/actual output
- Version info blocks (pydantic/python/OS versions) - these are relevant
- Anything you are unsure about - when in doubt, keep it

Return ONLY the cleaned text, no commentary."""


def llm_clean(text: str, key: str) -> str:
    body = json.dumps({
        "model": CLEAN_MODEL, "max_tokens": 8000,
        "messages": [{"role": "user", "content": f"{CLEAN_INSTRUCTIONS}\n\n"
                                                 f"<task-description>\n{text}\n</task-description>"}],
    }).encode()
    for attempt in range(4):
        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages", data=body,
            headers={"x-api-key": key, "anthropic-version": "2023-06-01",
                     "content-type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.loads(r.read())["content"][0]["text"]
        except Exception:  # noqa: BLE001
            if attempt == 3:
                raise
            time.sleep(2 ** (attempt + 1))
    raise RuntimeError("unreachable")


def is_deletion_only(original: str, cleaned: str) -> bool:
    """Every non-blank cleaned line must appear in the original, in order."""
    orig = [l.strip() for l in original.splitlines()]
    i = 0
    for line in cleaned.splitlines():
        s = line.strip()
        if not s:
            continue
        while i < len(orig) and orig[i] != s:
            i += 1
        if i == len(orig):
            return False
        i += 1
    return True


def clean_one(task_file: Path, key: str) -> dict:
    t = json.loads(task_file.read_text())
    original = t["prompt"]
    for _ in range(2):
        cleaned = llm_clean(original, key).strip()
        if cleaned and is_deletion_only(original, cleaned) and len(cleaned) >= 0.2 * len(original):
            if cleaned == original.strip():
                return {"id": t["id"], "status": "unchanged"}
            t["prompt"] = cleaned
            task_file.write_text(json.dumps(t, indent=1, ensure_ascii=False))
            return {"id": t["id"], "status": "cleaned",
                    "saved": len(original) - len(cleaned)}
    return {"id": t["id"], "status": "verify_failed_kept_original"}


def cmd_clean(args) -> None:
    key = load_env("ANTHROPIC_API_KEY")
    tasks_dir = Path(args.tasks_dir)
    files = sorted(tasks_dir.glob("*/task.json"))
    backup = WORK / f"prompts_backup_{tasks_dir.name}.jsonl"
    if not backup.exists():
        with backup.open("w") as f:
            for tf in files:
                t = json.loads(tf.read_text())
                f.write(json.dumps({"id": t["id"], "prompt": t["prompt"]},
                                   ensure_ascii=False) + "\n")
        print(f"backed up {len(files)} prompts -> {backup}")

    stats, saved = Counter(), 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for fut in as_completed([pool.submit(clean_one, tf, key) for tf in files]):
            r = fut.result()
            stats[r["status"]] += 1
            saved += r.get("saved", 0)
            if r["status"] == "verify_failed_kept_original":
                print("  verify failed, original kept:", r["id"])
    print(dict(stats), f"| {saved} chars of boilerplate removed")


# ---------------------------------------------------------------------------
# stage 6 -- fairness judge
# ---------------------------------------------------------------------------
#
# The hidden tests come from the PR; the prompt comes from the issue. Sometimes
# the tests assert something the issue never pins down -- a message wording or an
# API name that got settled in the PR's comment thread, which the prompt
# deliberately excludes. Those tasks are unanswerable rather than hard.
#
# The rubric deliberately does NOT require every assertion to be stated in the
# prompt. Tests MAY assert things a competent engineer following the codebase's
# conventions would reasonably do anyway -- symmetric completions, adjacent edge
# cases. UNFAIR is reserved for invented wording, arbitrary picks among equally
# good API designs, orthogonal features, and dependency bumps.
#
# One judge call per task, verdict recorded with its reason so the labels can be
# spot-checked. This rates tasks; it does not delete them -- see `filter`.

JUDGE_MODEL = "claude-opus-5"

JUDGE_INSTRUCTIONS = """\
You are auditing a coding-benchmark task for fairness.

The agent is shown ONLY the task prompt below. It is then graded by hidden tests
taken from the pull request that fixed the issue. Your question is whether a
strong engineer, given only the prompt and the repository, could plausibly write
a change that passes those tests.

Rate FAIR, BORDERLINE or UNFAIR.

The tests MAY legitimately assert things the prompt does not state explicitly, so
long as a competent engineer following the repository's own conventions would
reasonably make them: symmetric completions of a change, adjacent edge cases,
consistency with how neighbouring code behaves. That is normal engineering, not
unfairness.

UNFAIR means the tests require a choice the prompt does not determine and the
codebase does not imply:
- an exact error/warning message wording invented by the PR
- one arbitrary pick among several equally reasonable API names or signatures
- behavior orthogonal to the reported problem
- a dependency or version bump

BORDERLINE is for genuine uncertainty between the two.

Answer with a single line `VERDICT: FAIR|BORDERLINE|UNFAIR`, then one sentence of
justification."""


def judge_one(task_file: Path, answers_dir: Path, key: str) -> dict:
    t = json.loads(task_file.read_text())
    ans = json.loads((answers_dir / t["id"] / "answer.json").read_text())
    content = (f"{JUDGE_INSTRUCTIONS}\n\n<prompt>\n{t['prompt']}\n</prompt>\n\n"
               f"<hidden-tests>\n{ans['test_patch'][:40000]}\n</hidden-tests>\n\n"
               f"<reference-fix>\n{ans['code_patch'][:20000]}\n</reference-fix>")
    body = json.dumps({"model": JUDGE_MODEL, "max_tokens": 2000,
                       "messages": [{"role": "user", "content": content}]}).encode()
    for attempt in range(4):
        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages", data=body,
            headers={"x-api-key": key, "anthropic-version": "2023-06-01",
                     "content-type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                text = json.loads(r.read())["content"][0]["text"]
            break
        except Exception:  # noqa: BLE001
            if attempt == 3:
                raise
            time.sleep(2 ** (attempt + 1))
    m = re.search(r"VERDICT:\s*(FAIR|BORDERLINE|UNFAIR)", text)
    return {"id": t["id"], "type": t["type"],
            "verdict": m.group(1) if m else "UNPARSED",
            "reason": text.split("\n")[-1].strip()[:400]}


def cmd_judge(args) -> None:
    key = load_env("ANTHROPIC_API_KEY")
    tasks_dir, answers_dir = Path(args.tasks_dir), Path(args.answers_dir)
    out_path = WORK / f"fairness_{tasks_dir.name}.json"
    done = json.loads(out_path.read_text()) if out_path.exists() else {}
    files = [f for f in sorted(tasks_dir.glob("*/task.json"))
             if json.loads(f.read_text())["id"] not in done]
    print(f"{len(done)} already judged, {len(files)} to go")

    stats = Counter(r["verdict"] for r in done.values())
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(judge_one, f, answers_dir, key) for f in files]
        for fut in as_completed(futs):
            r = fut.result()
            done[r["id"]] = r
            stats[r["verdict"]] += 1
            out_path.write_text(json.dumps(done, indent=1))
            print(f"  {r['id']:28s} {r['verdict']}")
    print(f"\n{dict(stats)} -> {out_path}")


# ---------------------------------------------------------------------------
# stage 7 -- select the shipped set
# ---------------------------------------------------------------------------
#
# Two filters, both requiring evidence gathered outside this script.
#
# Difficulty: a benchmark everything already passes measures nothing, so the set
# is narrowed to tasks a cheap model could NOT solve. Run the probe model over
# the full pool first (run_agent.py + grade.py), then point `filter` at its score
# file. This is the anti-saturation lever, and it is why the shipped set is
# smaller than the validated pool.
#
# Fairness: drop whatever `judge` rated UNFAIR. Keeping BORDERLINE is the default
# -- those are genuinely uncertain, and partial-credit scoring absorbs them.


def cmd_filter(args) -> None:
    scores = json.loads((WORK / "scores" / f"{args.probe_run}.json").read_text())
    rows = {r["task_id"]: r for r in scores["rows"]}
    fairness = {}
    for p in WORK.glob("fairness_*.json"):
        fairness.update({k: v["verdict"] for k, v in json.loads(p.read_text()).items()})
    if not fairness:
        print("WARNING: no work/fairness_*.json -- run `mine.py judge` first; "
              "selecting on difficulty alone")

    kept, cut = [], Counter()
    for tid, row in sorted(rows.items()):
        if row["score"] > args.max_score:
            cut["solved_by_probe"] += 1
            continue
        verdict = fairness.get(tid)
        if verdict and verdict not in args.keep_fairness:
            cut[f"fairness_{verdict}"] += 1
            continue
        kept.append(tid)

    out = WORK / f"ids_{args.name}.json"
    out.write_text(json.dumps(kept, indent=1))
    print(f"{len(rows)} probed -> {len(kept)} selected  (cut: {dict(cut)})")
    print(f"wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="stage", required=True)
    sub.add_parser("harvest", help="GitHub -> work/candidates.jsonl").set_defaults(fn=cmd_harvest)
    sub.add_parser("extract", help="clone -> work/extracted.jsonl").set_defaults(fn=cmd_extract)
    v = sub.add_parser("validate", help="pytest -> work/validated.jsonl")
    v.add_argument("--workers", type=int, default=4)
    v.add_argument("--limit", type=int, default=0, help="only process the first N pending")
    v.set_defaults(fn=cmd_validate)
    sub.add_parser("assemble", help="-> tasks/ + answers/").set_defaults(fn=cmd_assemble)
    c = sub.add_parser("clean", help="delete template boilerplate from the prompts")
    c.add_argument("--tasks-dir", default=str(TASKS_DIR))
    c.add_argument("--workers", type=int, default=8)
    c.set_defaults(fn=cmd_clean)

    j = sub.add_parser("judge", help="rate each task FAIR / BORDERLINE / UNFAIR")
    j.add_argument("--tasks-dir", default=str(TASKS_DIR))
    j.add_argument("--answers-dir", default=str(ANSWERS_DIR))
    j.add_argument("--workers", type=int, default=8)
    j.set_defaults(fn=cmd_judge)

    s = sub.add_parser("filter", help="select the shipped set: unsolved by the "
                                      "probe model, and not rated UNFAIR")
    s.add_argument("--probe-run", required=True,
                   help="name of a graded run over the full pool, e.g. haiku-probe")
    s.add_argument("--name", default="selected", help="output as work/ids_<name>.json")
    s.add_argument("--max-score", type=float, default=0.0,
                   help="keep tasks the probe scored at or below this (default: 0)")
    s.add_argument("--keep-fairness", nargs="*", default=["FAIR", "BORDERLINE"])
    s.set_defaults(fn=cmd_filter)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
