#!/usr/bin/env python3
"""Category 2, stress-testing test cases: write tests that catch a wrong implementation.

Given a PR description and the PR's change (tests excluded), the agent writes a
comprehensive test suite. It is scored on the percentage of deliberately buggy
implementations its suite successfully fails.

The obvious version of this task -- "write tests that fail before the fix and
pass after it" -- is easy: one assertion touching the changed code path scores
nearly full marks, and a thorough suite scores the same as a lazy one. This
category asks the question a test suite actually exists to answer: **would it
catch a wrong implementation of the change?** That cannot be measured against a
single gold patch, so the wrong implementations are supplied explicitly.

The agent gets the source with the PR's fix already applied and tests/ rolled
back to the parent commit, writes a test suite, and scores on how many hidden
plausible-but-buggy reimplementations of that change its suite kills.

    select     spares          ->  work/suite_candidates.jsonl
    variants   Claude Opus 5   ->  work/variants_raw.jsonl      (8 proposals/task)
    validate   pytest          ->  work/variants_validated.jsonl
    assemble   pure python     ->  tasks_suite/ + answers_suite/

The adversary is *proven*, not trusted. Every variant must keep all pre-existing
tests in the PR's touched test files green (plausible) and must fail its own
model-authored witness test, which itself must pass on the correct code (wrong).
Variants are then tiered by whether the PR's own tests would have caught them;
the `gold_caught: false` ones are what keeps the family unsaturated.

    export ANTHROPIC_API_KEY=...
    python3 mine_suite.py select --n 15
    python3 mine_suite.py variants --workers 5     # ~20 min/task, Opus 5
    python3 mine_suite.py validate --workers 3     # ~1 min/task
    python3 mine_suite.py assemble
    python3 mine.py clean --tasks-dir tasks_suite  # strip prompt boilerplate

`--set NAME` builds a second, independent adversary from the same source PRs
using a different generator model, so an agent's suites can be re-scored against
both. If the generating model also scores highest, that lead may be familiarity
with its own idea of a plausible mistake rather than better test design; a second
adversary is how you tell those apart.
"""

import argparse
import json
import re
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from mine import (CLONE_DIR, HERE, PYTHON, TEST_DEPS, WORK, append_jsonl,
                  collect_error_files, forbidden_introspection, git, load_env,
                  read_jsonl, run_pytest, sh, worktree_add, worktree_remove)

WT_ROOT = WORK / "wt_suite"
DEFAULT_MODEL = "claude-opus-5"
N_VARIANTS = 8
CONTEXT_PAD = 70  # lines of post-fix source shown around each changed hunk


class SuiteSet:
    """Every path this pipeline reads or writes, for one named adversary.

    A "set" is one adversary: the same source PRs mutated by a different
    generator model live in a different set, so two can coexist and the same
    agent submission can be scored against both.
    """

    def __init__(self, name: str = ""):
        self.name = name
        sfx = f"_{name}" if name else ""
        self.candidates = WORK / f"suite_candidates{sfx}.jsonl"
        self.raw = WORK / f"variants_raw{sfx}.jsonl"
        self.validated = WORK / f"variants_validated{sfx}.jsonl"
        self.discards = WORK / f"discards_suite{sfx}.jsonl"
        self.tasks = HERE / f"tasks_suite{sfx}"
        self.answers = HERE / f"answers_suite{sfx}"
        self.id_map = WORK / f"suite_id_map{sfx}.json"
        self.ids = WORK / f"suite_ids{sfx}.json"


# ---------------------------------------------------------------------------
# select -- which validated PRs become suite tasks
# ---------------------------------------------------------------------------
#
# Drawn from the stage-3 survivors that mine.py did NOT ship in the 100, so this
# family shares no PR with fix / testwrite / localize and all four can be scored
# on the same run.
#
# Every criterion is about making mutation testing meaningful:
#   - exactly one changed source file, 3-40 added lines: a single well-localized
#     behavior to mutate. Multi-file rewrites give the generator nowhere to hide
#     a subtle defect and make "keeps the old suite green" ambiguous.
#   - >= 20 P2P tests and <= 15s gold runtime: a real regression set to prove
#     plausibility against, at a runtime we can afford ~10x per task.
#   - a prompt long enough to specify behavior: the agent must know WHICH
#     behavior to pin down.

# PRs about test infrastructure, dependency bumps, typing-only changes or docs
# describe no user-visible behavior, so no suite could be expected to cover the
# corners a mutant hides in. Excluded rather than shipped and apologised for.
NON_BEHAVIOR_TITLE = re.compile(
    r"test regression|^tests?\b|\btest suite\b|\bci\b|\bbump\b|dependabot"
    r"|\btypo\b|\bdocs?\b|docstring|changelog|lint|ruff|mypy|pyright"
    r"|coverage|benchmark|pre-commit|\bpin\b|\brelease\b",
    re.IGNORECASE,
)


def is_behavior_prompt(prompt: str) -> bool:
    return not NON_BEHAVIOR_TITLE.search(prompt.splitlines()[0].strip("# ").strip())


def added_lines(patch: str) -> int:
    return sum(1 for l in patch.splitlines()
               if l.startswith("+") and not l.startswith("+++"))


def patch_files(patch: str) -> set:
    return set(re.findall(r"^\+\+\+ b/(\S+)", patch or "", re.MULTILINE))


def spare_records() -> list[dict]:
    """The validated survivors mine.py left unshipped, as full task records.

    Issue-sourced only. A PR body describes the change its author made, which
    leaks implementation intent into a prompt that is supposed to specify
    behavior and nothing else; an issue describes the behavior a user wanted.
    """
    manifest = json.loads((WORK / "manifest.json").read_text())
    spare_prs = set(manifest["spare_pr_numbers"])
    extracted = {r["pr_number"]: r for r in read_jsonl(WORK / "extracted.jsonl")}
    out = []
    for v in read_jsonl(WORK / "validated.jsonl"):
        if v["status"] != "ok" or v["pr_number"] not in spare_prs:
            continue
        e = extracted[v["pr_number"]]
        if e["prompt_source"] != "issue":
            continue
        out.append({
            "id": f"pr-{v['pr_number']}",
            "pr_number": v["pr_number"],
            "pr_url": e["url"],
            "merged_at": e["merged_at"],
            "base_commit": e["base_commit"],
            "merge_commit": e["merge_commit"],
            "code_patch": e["code_patch"],
            "test_patch": e["test_patch"],
            "test_files": e["test_files"],
            "gold_loc_files": sorted(f for f in e["code_files"]
                                     if f.startswith("pydantic/") and f.endswith(".py")),
            "f2p": v["f2p"],
            "p2p": v["p2p"],
            "test_seconds": v["test_seconds"],
            "prompt": f"# {e['prompt_title']}\n\n{e['prompt_body']}".strip(),
        })
    # validated.jsonl is in thread-completion order; sort so that selection --
    # whose round-robin breaks ties by insertion order -- is reproducible.
    out.sort(key=lambda r: r["pr_number"])
    return out


def cmd_select(args) -> None:
    paths = SuiteSet(args.set)
    force = set(args.include)
    if args.include_set:
        # Force-include every source PR already shipped in another set, so the
        # two adversaries cover the same tasks and are directly comparable.
        base = SuiteSet("" if args.include_set == "default" else args.include_set)
        if base.id_map.exists():
            force |= {cid for cid, tid in json.loads(base.id_map.read_text()).items()
                      if (base.answers / tid).exists()}

    rows = []
    for rec in spare_records():
        cfiles = patch_files(rec["code_patch"])
        if len(cfiles) != 1:
            continue
        path = next(iter(cfiles))
        if not path.startswith("pydantic/") or not path.endswith(".py"):
            continue
        n_add = added_lines(rec["code_patch"])
        if not (3 <= n_add <= 40):
            continue
        if len(rec["p2p"]) < args.min_p2p or rec["test_seconds"] > args.max_test_seconds:
            continue
        if len(rec["prompt"]) < 400 or not is_behavior_prompt(rec["prompt"]):
            continue
        rows.append({**rec, "path": path, "n_added": n_add})

    picked = [r for r in rows if r["id"] in force]
    missing = force - {r["id"] for r in picked}
    if missing:
        raise SystemExit(f"forced ids are not eligible under the current filters: "
                         f"{sorted(missing)}")
    rows = [r for r in rows if r["id"] not in force]

    # Diversity: round-robin over the changed file so one hot module cannot
    # dominate; within a module, prefer the largest regression set.
    by_mod: dict[str, list] = {}
    for r in rows:
        by_mod.setdefault(r["path"], []).append(r)
    for v in by_mod.values():
        v.sort(key=lambda r: -len(r["p2p"]))
    mods = sorted(by_mod, key=lambda m: -len(by_mod[m]))
    while len(picked) < args.n and any(by_mod.values()):
        for m in mods:
            if by_mod[m] and len(picked) < args.n:
                picked.append(by_mod[m].pop(0))

    paths.candidates.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in picked))
    print(f"{len(rows) + len(force)} eligible spares -> picked {len(picked)}")
    for r in picked:
        print(f"  {r['id']:12s} PR#{r['pr_number']:<6} {r['path']:<48} "
              f"+{r['n_added']:<3} p2p={len(r['p2p'])} {r['test_seconds']}s")
    print(f"\nwrote {paths.candidates}")


# ---------------------------------------------------------------------------
# variants -- propose plausible-but-buggy alternatives (one LLM call per task)
# ---------------------------------------------------------------------------

VARIANT_SCHEMA = {
    "type": "object",
    "properties": {
        "variants": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string",
                             "description": "short snake_case id, e.g. only_handles_str_subclass"},
                    "defect_class": {
                        "type": "string",
                        "description": "one of: boundary, inverted_condition, missing_case, "
                        "wrong_default, incomplete_propagation, wrong_scope, "
                        "swallowed_error, wrong_error_type, ordering, other"},
                    "rationale": {
                        "type": "string",
                        "description": "what an engineer would plausibly believe when writing "
                        "this, and the concrete input where it diverges from correct behavior"},
                    "edits": {
                        "type": "array",
                        "items": {"type": "object",
                                  "properties": {"path": {"type": "string"},
                                                 "old": {"type": "string"},
                                                 "new": {"type": "string"}},
                                  "required": ["path", "old", "new"],
                                  "additionalProperties": False}},
                    "witness_test": {"type": "string",
                                     "description": "complete standalone pytest module source"},
                },
                "required": ["name", "defect_class", "rationale", "edits", "witness_test"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["variants"],
    "additionalProperties": False,
}

SYSTEM = """\
You build mutation-testing benchmarks for the pydantic library. You are given a
merged pull request that changed pydantic's behavior, and the post-merge source
of the file it changed. You produce plausible-but-wrong alternative
implementations of that same change ("variants"), each paired with a test that
proves it is wrong.

A variant is only useful to us if BOTH hold:

1. It LOOKS RIGHT. It reads as a competent engineer's implementation of the
   described behavior -- no stubs, no `pass`, no deleted function bodies, no
   syntax errors, no obviously arbitrary constants. It must keep the rest of
   pydantic working: it may not break unrelated behavior, and it must still
   pass pydantic's large pre-existing test suite. Prefer defects that hide in a
   case the obvious test would not reach.

2. It IS WRONG. There must be a concrete input for which observable behavior
   through pydantic's public API differs from the correct implementation --
   different validation outcome, different error type/location/message,
   different JSON schema, different repr, different warning. A pure refactor,
   or a change only visible by reading the source, is worthless to us.

Each variant is expressed as exact-string edits against the post-merge source
shown to you, and comes with a `witness_test`: a complete standalone pytest
module that PASSES on the correct implementation and FAILS on that variant. The
witness is our proof the variant is genuinely broken, so keep it narrow and
deterministic.

Witness test rules:
- One module per variant, self-contained: its own imports, no fixtures from
  pydantic's test suite, no helpers defined elsewhere.
- Assert the behavior a user observes. Never inspect source text, bytecode, or
  file contents (no `inspect.getsource`, no reading files under `pydantic/`, no
  `dis`, no `ast`), and never shell out.
- Deterministic: no randomness, no network, no clocks, no dict-ordering luck.
- Use only pydantic, pytest, and the standard library.
"""

PROMPT = """\
# Pull request under test

{pr_text}

## The change (unified diff, already applied in the source below)

```diff
{code_patch}
```

## Post-merge source: `{path}`

This is the CORRECT implementation. Your edits' `old` strings must be copied
verbatim from this text, and each must occur exactly once in the full file.
{trunc_note}

```python
{source}
```

# Your task

Produce exactly {n} variants of this change, spanning distinct defect classes
(do not give us {n} versions of one off-by-one). Each must satisfy both
conditions from your instructions.

Notes specific to this repository:
- Edits must stay inside `{path}`. Do not touch `tests/`, `pydantic-core/`,
  or packaging files.
- Behavior is usually observed by building a `BaseModel` / `TypeAdapter` and
  validating, serializing, or asking for `model_json_schema()`.
- pydantic's test suite runs with `filterwarnings = error`, so a variant that
  emits a new warning on a common path will break unrelated tests: that makes
  it implausible, not clever.
- Assume the witness module lives at `tests/test_witness.py` and runs against
  an installed pydantic.

Return JSON only, matching the required schema.
"""


def salvage_variants(text: str) -> list[dict]:
    """Pull the complete objects out of a `variants` array cut off mid-stream."""
    i = text.find('"variants"')
    i = text.find("[", i) if i >= 0 else -1
    if i < 0:
        return []
    dec, out, pos = json.JSONDecoder(), [], i + 1
    while True:
        while pos < len(text) and text[pos] in ", \n\r\t":
            pos += 1
        try:
            obj, pos = dec.raw_decode(text, pos)
        except json.JSONDecodeError:
            break
        if not isinstance(obj, dict):
            break
        out.append(obj)
    required = {"name", "defect_class", "rationale", "edits", "witness_test"}
    return [o for o in out if required <= set(o)]


def changed_line_ranges(code_patch: str, path: str) -> list[tuple[int, int]]:
    """Post-image line ranges of each hunk touching `path`."""
    ranges, in_file = [], False
    for line in code_patch.splitlines():
        if line.startswith("+++ b/"):
            in_file = line[6:].strip() == path
        elif in_file and line.startswith("@@"):
            m = re.search(r"\+(\d+)(?:,(\d+))?", line)
            if m:
                start, count = int(m.group(1)), int(m.group(2) or 1)
                ranges.append((start, start + count))
    return ranges


def window_source(src: str, ranges: list[tuple[int, int]], pad: int) -> tuple[str, bool]:
    """Full file if small, else the hunks with `pad` lines of context each."""
    lines = src.splitlines()
    if len(lines) <= 700 or not ranges:
        return src, False
    keep = set()
    for a, b in ranges:
        keep.update(range(max(1, a - pad), min(len(lines), b + pad) + 1))
    out, prev = [], 0
    for i in sorted(keep):
        if prev and i != prev + 1:
            out.append(f"# ... lines {prev + 1}-{i - 1} elided ...")
        out.append(lines[i - 1])
        prev = i
    return "\n".join(out), True


def generate_one(client, rec: dict, effort: str, max_tokens: int, model: str) -> dict:
    path = rec["gold_loc_files"][0]
    src = git("show", f"{rec['merge_commit']}:{path}")
    shown, truncated = window_source(
        src, changed_line_ranges(rec["code_patch"], path), CONTEXT_PAD)
    prompt = PROMPT.format(
        pr_text=rec["prompt"].strip()[:12000],
        code_patch=rec["code_patch"][:20000],
        path=path, source=shown, n=N_VARIANTS,
        trunc_note=("Long file: only the regions around the change are shown, with "
                    "elisions marked." if truncated else ""))
    # Deliberately NO server-side fallbacks: a refusal must surface as a refusal.
    # Silently re-running on another model would put a different model's variants
    # into a set labelled with this one -- exactly the confound a second adversary
    # is meant to remove.
    with client.messages.stream(
        model=model,
        # Generous: adaptive thinking at high effort can spend tens of thousands
        # of tokens before the JSON, and a truncated response yields no text block.
        max_tokens=max_tokens,
        system=SYSTEM,
        output_config={"effort": effort,
                       "format": {"type": "json_schema", "schema": VARIANT_SCHEMA}},
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        msg = stream.get_final_message()
    usage = {"input": msg.usage.input_tokens, "output": msg.usage.output_tokens}
    if msg.stop_reason == "refusal":
        return {"id": rec["id"], "status": "refusal", "usage": usage,
                "detail": str(getattr(msg, "stop_details", None))}
    text = next((b.text for b in msg.content if b.type == "text"), "")
    if not text:
        return {"id": rec["id"], "status": "no_output", "usage": usage,
                "detail": f"stop_reason={msg.stop_reason}"}
    try:
        variants, salvaged = json.loads(text)["variants"], False
    except json.JSONDecodeError:
        # Output cap reached mid-array: keep the variants that did finish rather
        # than paying for the whole call again. They get validated anyway.
        variants, salvaged = salvage_variants(text), True
        if not variants:
            return {"id": rec["id"], "status": "truncated", "usage": usage,
                    "detail": f"stop_reason={msg.stop_reason}, {len(text)} chars"}
    for i, v in enumerate(variants):
        v["index"] = i
    return {"id": rec["id"], "status": "ok", "pr_number": rec["pr_number"],
            "path": path, "n": len(variants), "variants": variants, "model": model,
            "usage": usage, "source_truncated": truncated, "salvaged": salvaged}


def cmd_variants(args) -> None:
    import anthropic

    client = anthropic.Anthropic(api_key=load_env("ANTHROPIC_API_KEY"))
    paths = SuiteSet(args.set)
    cands = read_jsonl(paths.candidates)
    done = {r["id"] for r in read_jsonl(paths.raw) if r.get("status") == "ok"}
    pending = [c for c in cands if c["id"] not in done]
    if args.limit:
        pending = pending[: args.limit]
    print(f"{len(cands)} candidates, {len(done)} already generated, {len(pending)} to go "
          f"(set={args.set or 'default'}, model={args.model})", flush=True)

    tok_in = tok_out = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(generate_one, client, c, args.effort, args.max_tokens,
                            args.model): c["id"] for c in pending}
        for i, fut in enumerate(as_completed(futs), 1):
            cid = futs[fut]
            try:
                res = fut.result()
            except Exception as e:  # noqa: BLE001
                res = {"id": cid, "status": "error", "detail": repr(e)[-300:]}
            append_jsonl(paths.raw, res)
            u = res.get("usage") or {}
            tok_in += u.get("input", 0)
            tok_out += u.get("output", 0)
            print(f"[{i}/{len(pending)}] {cid}: {res['status']} n={res.get('n', 0)} "
                  f"{res.get('detail', '')}", flush=True)
    print(f"\ntokens in={tok_in} out={tok_out}")


# ---------------------------------------------------------------------------
# validate -- keep only the variants we can prove are plausible AND wrong
# ---------------------------------------------------------------------------
#
# One disposable worktree + venv per candidate, pinned exactly like mine.py's
# validate stage. The worktree is put in the CORRECT state once (base commit +
# the PR's code patch, tests left at base), then each variant is applied and
# reverted in place; pydantic is installed editable, so a source edit takes
# effect immediately and no reinstall is needed.
#
#   ref   run the PR's touched test files on the correct code -> ref_pass, the
#         regression set a plausible variant may not break.
#   wit0  run every proposed witness on the correct code; a witness that fails
#         here is wrong about correct behavior -> its variant is dropped.
#
# Then per variant: edits must apply and pydantic must still import; every test
# in ref_pass must still pass (plausible); its own witness must now fail (wrong
# -- a collection error does not count, that is a broken test, not a caught bug);
# and finally the PR's gold test patch is applied to tier it as gold_caught or
# gold_missed.


def apply_edits(wt: Path, path: str, edits: list[dict]) -> str | None:
    """Apply exact-string edits in place. Returns an error reason, or None.

    grade.py imports this, so the mutation semantics of the validator that proved
    a variant plausible-and-wrong and of the grader that scores against it cannot
    drift apart.
    """
    target = wt / path
    text = target.read_text()
    for e in edits:
        if e["path"] != path:
            return f"edit outside target file: {e['path']}"
        n = text.count(e["old"])
        if n == 0:
            return "old string not found"
        if n > 1:
            return f"old string not unique ({n} occurrences)"
        text = text.replace(e["old"], e["new"], 1)
    target.write_text(text)
    return None


def failed(outcomes: dict, nodes) -> set:
    cerr = collect_error_files(outcomes)
    return {n for n in nodes
            if outcomes.get(n, "absent") in ("failed", "error", "absent")
            or n.split("::")[0] in cerr}


def validate_variants(rec: dict, proposals: list[dict], min_variants: int,
                      set_name: str = "") -> dict:
    cid, path = rec["id"], rec["path"]
    wt = WT_ROOT / f"wt-{set_name}{rec['pr_number']}"
    worktree_remove(wt)
    t0 = time.time()
    per_variant = []
    try:
        worktree_add(wt, rec["base_commit"])
        r = sh(["uv", "venv", "--quiet", "--python", PYTHON, ".venv"], cwd=wt, timeout=300)
        if r.returncode != 0:
            return {"id": cid, "status": "discard", "reason": "venv_failed"}
        r = sh(["uv", "pip", "install", "--quiet", "--python", ".venv/bin/python",
                "-e", ".", *TEST_DEPS], cwd=wt, timeout=1800)
        if r.returncode != 0:
            return {"id": cid, "status": "discard", "reason": "install_failed",
                    "detail": r.stderr[-300:]}

        # --- correct state: base + code patch, tests untouched ----------------
        pf = wt / "_p.patch"
        pf.write_text(rec["code_patch"])
        if sh(["git", "apply", "--whitespace=nowarn", str(pf)], cwd=wt).returncode != 0:
            return {"id": cid, "status": "discard", "reason": "code_patch_apply_failed"}
        correct_src = (wt / path).read_text()

        ref = run_pytest(wt, rec["test_files"], wt / "ref.json")
        if ref is None:
            return {"id": cid, "status": "discard", "reason": "ref_pytest_timeout"}
        ref_pass = {n for n, o in ref.items() if o == "passed"}
        if len(ref_pass) < 20:
            return {"id": cid, "status": "discard", "reason": "ref_suite_too_small",
                    "detail": len(ref_pass)}

        # --- witness sanity on the correct implementation ---------------------
        wit_files = {}
        for v in proposals:
            bad = forbidden_introspection(v["witness_test"])
            if bad:
                per_variant.append({"index": v["index"], "name": v["name"], "ok": False,
                                    "reason": f"witness_introspects: {bad}"})
                continue
            f = f"tests/test_witness_{cid.split('-')[-1]}_{v['index']:02d}.py"
            (wt / f).write_text(v["witness_test"])
            wit_files[v["index"]] = f
        wit_src = {f: p["witness_test"] for p in proposals
                   for f in [wit_files.get(p["index"])] if f}
        if not wit_files:
            return {"id": cid, "status": "discard", "reason": "no_usable_witnesses",
                    "variants": per_variant}
        wit0 = run_pytest(wt, sorted(wit_files.values()), wt / "w0.json") or {}
        wit0_cerr = collect_error_files(wit0)
        wit_ok = {}
        for idx, f in wit_files.items():
            nodes = [n for n in wit0 if n.split("::")[0] == f
                     and not n.startswith("__collect_error__:")]
            if f in wit0_cerr or not nodes:
                per_variant.append({"index": idx, "ok": False,
                                    "reason": "witness_does_not_collect"})
            elif any(wit0.get(n) != "passed" for n in nodes):
                per_variant.append({"index": idx, "ok": False,
                                    "reason": "witness_fails_on_correct_code"})
            else:
                wit_ok[idx] = (f, nodes)

        # --- per variant ------------------------------------------------------
        kept = []
        for v in proposals:
            idx = v["index"]
            if idx not in wit_ok:
                continue
            wfile, wnodes = wit_ok[idx]
            (wt / path).write_text(correct_src)  # start from correct every time
            err = apply_edits(wt, path, v["edits"])
            if err:
                per_variant.append({"index": idx, "name": v["name"], "ok": False,
                                    "reason": f"edits_failed: {err}"})
                continue
            if sh([str(wt / ".venv/bin/python"), "-c", "import pydantic"], cwd=wt,
                  timeout=120).returncode != 0:
                per_variant.append({"index": idx, "name": v["name"], "ok": False,
                                    "reason": "import_failed"})
                continue

            reg = run_pytest(wt, rec["test_files"], wt / "reg.json")
            if reg is None:
                per_variant.append({"index": idx, "name": v["name"], "ok": False,
                                    "reason": "regression_pytest_timeout"})
                continue
            broke = failed(reg, ref_pass)
            if broke:
                per_variant.append({"index": idx, "name": v["name"], "ok": False,
                                    "reason": "breaks_existing_tests",
                                    "detail": f"{len(broke)} of {len(ref_pass)}: "
                                              f"{sorted(broke)[:3]}"})
                continue

            wit = run_pytest(wt, [wfile], wt / "w.json") or {}
            if wfile in collect_error_files(wit):
                per_variant.append({"index": idx, "name": v["name"], "ok": False,
                                    "reason": "witness_collect_error_on_variant"})
                continue
            if not [n for n in wnodes if wit.get(n, "absent") in ("failed", "error")]:
                per_variant.append({"index": idx, "name": v["name"], "ok": False,
                                    "reason": "witness_passes_on_variant"})
                continue

            # Tier against the PR's own tests. gold_missed variants are what keep
            # the family unsaturated, and they are only admissible because the
            # witness already proved this variant broken independently of them.
            pf.write_text(rec["test_patch"])
            gold_applied = sh(["git", "apply", "--whitespace=nowarn", str(pf)],
                              cwd=wt).returncode == 0
            gold_caught = None
            if gold_applied:
                g = run_pytest(wt, rec["test_files"], wt / "g.json") or {}
                gold_caught = bool(failed(g, rec["f2p"]))
                sh(["git", "checkout", rec["base_commit"], "--", "tests/"], cwd=wt)
                sh(["git", "clean", "-fdq", "tests/"], cwd=wt)
                for f2, src2 in wit_src.items():
                    (wt / f2).write_text(src2)

            kept.append({**v, "witness_file": wfile, "witness_nodes": wnodes,
                         "gold_caught": gold_caught})
            per_variant.append({"index": idx, "name": v["name"], "ok": True,
                                "defect_class": v["defect_class"],
                                "gold_caught": gold_caught})

        (wt / path).write_text(correct_src)
        n_gold = sum(1 for k in kept if k["gold_caught"])
        res = {"id": cid, "pr_number": rec["pr_number"], "path": path,
               "status": "ok" if len(kept) >= min_variants else "discard",
               "n_kept": len(kept), "n_proposed": len(proposals),
               "gold_kill_rate": round(n_gold / len(kept), 3) if kept else None,
               "ref_pass": len(ref_pass), "variants": kept, "audit": per_variant,
               "seconds": round(time.time() - t0, 1)}
        if res["status"] == "discard":
            res["reason"] = "too_few_valid_variants"
        return res
    except Exception as e:  # noqa: BLE001
        return {"id": cid, "status": "discard", "reason": "exception",
                "detail": repr(e)[-400:], "audit": per_variant}
    finally:
        worktree_remove(wt)


def cmd_validate(args) -> None:
    paths = SuiteSet(args.set)
    WT_ROOT.mkdir(parents=True, exist_ok=True)
    cands = {c["id"]: c for c in read_jsonl(paths.candidates)}
    raw = {r["id"]: r for r in read_jsonl(paths.raw) if r.get("status") == "ok"}
    done = {r["id"] for r in read_jsonl(paths.validated)}
    done |= {d["id"] for d in read_jsonl(paths.discards)}
    todo = [cid for cid in raw if cid not in done
            and (not args.only or cid in set(args.only))]
    print(f"{len(raw)} generated, {len(done)} already validated, {len(todo)} to go "
          f"({args.workers}-wide)", flush=True)

    n_ok = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(validate_variants, cands[cid], raw[cid]["variants"],
                            args.min_variants, args.set) for cid in todo]
        for i, fut in enumerate(as_completed(futs), 1):
            res = fut.result()
            if res["status"] == "ok":
                n_ok += 1
                append_jsonl(paths.validated, res)
            else:
                append_jsonl(paths.discards, res)
            print(f"[{i}/{len(todo)}] {res['id']}: {res['status']} "
                  f"kept={res.get('n_kept', 0)}/{res.get('n_proposed', 0)} "
                  f"gold_kill={res.get('gold_kill_rate')} "
                  f"{res.get('reason', '')} ({res.get('seconds', '?')}s)", flush=True)
    print(f"\nshipped {n_ok}/{len(todo)}")


# ---------------------------------------------------------------------------
# assemble -- emit the task family
# ---------------------------------------------------------------------------
#
# The instructions say nothing about how the task was built: no PR, no commits,
# no "tests were rolled back", and they do not name the library. Construction
# detail is not just noise -- an agent told that tests/ was rewound to a parent
# commit has been handed the hint that a matching upstream test file exists to
# reconstruct, which is the opposite of what the task measures.

INSTRUCTIONS = """\
The repository at /repo is a Python library. The behavior described below is
already implemented in its source.

Your job is to write a comprehensive pytest test suite for that behavior, in new
file(s) under tests/. Read the implementation to work out what is worth
asserting, and cover the edge cases -- not just the happy path. Write your tests
to disk as you go rather than only at the end.

Work by reading only. Do not run any tests and do not execute the library:
navigating and reading files is fine, running code is not. Do not modify the
library source or any existing test file. Your tests must assert behavior through
the library's public API, never by inspecting its source text or bytecode.
"""


def near_change(src: str, code_patch: str, path: str, edits: list[dict],
                pad: int = 60) -> bool | None:
    """Do this variant's edits sit at the change the prompt describes?

    Recorded per variant rather than enforced: a variant that mutates code far
    from the PR's hunks would be wrong about behavior the prompt never mentions,
    so no reasonable suite could be expected to kill it.
    """
    ranges = changed_line_ranges(code_patch, path)
    if not ranges:
        return None
    for e in edits:
        i = src.find(e["old"])
        if i < 0:
            return None
        a = src.count("\n", 0, i) + 1
        b = a + e["old"].count("\n")
        if not any(a <= hi + pad and b >= lo - pad for lo, hi in ranges):
            return False
    return True


def cmd_assemble(args) -> None:
    paths = SuiteSet(args.set)
    cands = {c["id"]: c for c in read_jsonl(paths.candidates)}
    validated = {r["id"]: r for r in read_jsonl(paths.validated)}
    paths.tasks.mkdir(exist_ok=True)
    paths.answers.mkdir(exist_ok=True)

    # Task ids are assigned once and never reshuffled: a later validation pass
    # adding tasks must not renumber ones models have already been run on.
    id_map = json.loads(paths.id_map.read_text()) if paths.id_map.exists() else {}
    if args.base_id_map:
        base = SuiteSet("" if args.base_id_map == "default" else args.base_id_map).id_map
        if base.exists():  # shared source PRs keep the other set's task id
            id_map = {**json.loads(base.read_text()), **id_map}
    taken = {int(t.rsplit("-", 1)[1]) for t in id_map.values()}
    for cid in sorted(validated):
        if cid not in id_map:
            n = next(i for i in range(1, 999) if i not in taken)
            taken.add(n)
            id_map[cid] = f"pydantic-suite-{n:03d}"
    paths.id_map.write_text(json.dumps(id_map, indent=1))

    shipped, dropped = [], []
    for cid, val in sorted(validated.items(), key=lambda kv: id_map[kv[0]]):
        tid = id_map[cid]
        if cid not in cands:
            # Validated under looser filters than the ones in force now. Not
            # shippable; say so loudly rather than emitting it anyway.
            dropped.append((tid, cid, "no longer an eligible candidate"))
            continue
        c = cands[cid]
        if not is_behavior_prompt(c["prompt"]):
            dropped.append((tid, cid, "prompt specifies no behavior"))
            continue
        merged_src = git("show", f"{c['merge_commit']}:{val['path']}")
        scored = [{**v, "near_change": near_change(merged_src, c["code_patch"],
                                                   val["path"], v["edits"])}
                  for v in val["variants"][: args.max_variants]]
        # Off-hunk variants break behavior the prompt never describes: kept in the
        # answer for inspection, but they must not count against a suite.
        variants = [v for v in scored if v["near_change"] is not False]
        excluded = [v for v in scored if v["near_change"] is False]
        if len(variants) < args.min_variants:
            dropped.append((tid, cid, f"only {len(variants)} on-topic variants"))
            continue

        (paths.tasks / tid).mkdir(exist_ok=True)
        (paths.tasks / tid / "task.json").write_text(json.dumps({
            "id": tid,
            "category": "stress_test",
            "type": "suite",
            "repo": "pydantic/pydantic",
            "base_commit": c["merge_commit"],   # source: the fix already applied
            "tests_commit": c["base_commit"],   # tests/: rolled back
            "instructions": INSTRUCTIONS,
            "prompt": c["prompt"],
        }, indent=1))
        (paths.answers / tid).mkdir(exist_ok=True)
        (paths.answers / tid / "answer.json").write_text(json.dumps({
            "id": tid,
            "type": "suite",
            "source_id": cid,
            "pr_number": c["pr_number"],
            "pr_url": c["pr_url"],
            "merged_at": c["merged_at"],
            # Grading rebuilds the correct state as base_commit + code_patch, the
            # same way validation did, so variant edits match byte for byte.
            "base_commit": c["base_commit"],
            "merge_commit": c["merge_commit"],
            "code_patch": c["code_patch"],
            "test_patch": c["test_patch"],
            "path": val["path"],
            "test_files": c["test_files"],
            "f2p": c["f2p"],
            "p2p": c["p2p"],
            "gold_loc_files": c["gold_loc_files"],
            "variants": variants,
            "n_variants": len(variants),
            "excluded_variants": excluded,
            "gold_kill_rate": round(sum(1 for v in variants if v["gold_caught"])
                                    / len(variants), 3),
            "ref_pass": val["ref_pass"],
        }, indent=1))
        shipped.append((tid, cid, c["pr_number"], val["path"], len(variants),
                        sum(1 for v in variants if v["gold_caught"]), len(excluded)))

    print(f"shipped {len(shipped)} suite tasks -> {paths.tasks}")
    print(f"{'task':22s} {'source':12s} {'PR':>6} {'variants':>9} {'gold-caught':>12} "
          f"{'off-topic':>10}  path")
    for tid, cid, pr, path, n, ng, nf in shipped:
        print(f"{tid:22s} {cid:12s} {pr:>6} {n:>9} {ng:>12} {nf:>10}  {path}")
    for tid, cid, why in dropped:
        print(f"dropped {tid} ({cid}): {why}")
    tot = sum(s[4] for s in shipped)
    gold = sum(s[5] for s in shipped)
    if tot:
        print(f"\n{tot} variants total; the PR's own tests would kill {gold} "
              f"({gold / tot:.0%}) -- that is the gold suite's score on this family.")
    paths.ids.write_text(json.dumps(sorted(s[0] for s in shipped), indent=1))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--set", default="", help="named adversary (suffixes every path)")
    sub = ap.add_subparsers(dest="stage", required=True)

    s = sub.add_parser("select", help="spares -> work/suite_candidates.jsonl")
    s.add_argument("--n", type=int, default=15)
    s.add_argument("--min-p2p", type=int, default=20)
    s.add_argument("--max-test-seconds", type=float, default=15.0)
    s.add_argument("--include", nargs="*", default=[], help="source ids to force-include")
    s.add_argument("--include-set", default="",
                   help="force-include every source id already shipped in this set")
    s.set_defaults(fn=cmd_select)

    g = sub.add_parser("variants", help="LLM -> work/variants_raw.jsonl")
    g.add_argument("--model", default=DEFAULT_MODEL)
    g.add_argument("--workers", type=int, default=4)
    g.add_argument("--limit", type=int, default=0)
    # Adaptive thinking shares max_tokens with the answer, so a truncated reply is
    # usually thinking crowding out the JSON: lower the effort or raise the cap.
    g.add_argument("--effort", default="high",
                   choices=["low", "medium", "high", "xhigh", "max"])
    g.add_argument("--max-tokens", type=int, default=64000)
    g.set_defaults(fn=cmd_variants)

    v = sub.add_parser("validate", help="prove each variant plausible AND wrong")
    v.add_argument("--workers", type=int, default=3)
    v.add_argument("--min-variants", type=int, default=4)
    v.add_argument("--only", nargs="*")
    v.set_defaults(fn=cmd_validate)

    a = sub.add_parser("assemble", help="-> tasks_suite/ + answers_suite/")
    a.add_argument("--max-variants", type=int, default=8)
    a.add_argument("--min-variants", type=int, default=4)
    a.add_argument("--base-id-map", default="",
                   help="reuse another set's task ids for shared source PRs")
    a.set_defaults(fn=cmd_assemble)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
