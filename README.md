# pydantic-bench

An evaluation benchmark for coding agents, mined from [pydantic/pydantic](https://github.com/pydantic/pydantic)
— chosen because it has a large volume of recent merged PRs. Every task has a
frozen starting state, a prompt, and an automatic scoring procedure returning a
number in [0, 1]. Agents are run with
[mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent) in Docker.

Eight scripts and a Dockerfile, no package to install. Three scripts mine tasks
out of the repository, three run and score them, and the rest is the container
image.

## The three task categories

| # | category | the agent starts with | it is asked to | score |
|---|---|---|---|---|
| 1 | **General feature / spec implementation** | the PR's parent commit | implement what the PR description asks for | percentage of the PR's test cases that pass |
| 2 | **Stress-testing test cases** | the change already applied, `tests/` rolled back | write a comprehensive test suite for the described behavior | percentage of deliberately buggy implementations the suite fails |
| 3 | **Performance optimization** | the parent of a performance PR | make the described slow path faster | 0 unless the tests still pass *and* it beats the human optimization |

One script builds each category — `mine.py`, `mine_suite.py`, `mine_perf.py` —
and every script's docstring covers its own design decisions in detail.

Each `task.json` carries a `category`, the reporting grouping above, and a
`type`, the internal label the graders dispatch on.

| category | n | task dir | emitted by | grader |
|---|---|---|---|---|
| `general_feature` | 82 | `tasks/` | `mine.py` | `grade.py` |
| `stress_test` | 12 | `tasks_suite/` | `mine_suite.py` | `grade.py` |
| `performance` | 6 | `tasks_perf/` | `mine_perf.py` | `grade_perf.py` |

`mine.py assemble` can also emit a `testwrite` type, which is **not** a reported
category: it saturates, which is exactly what `stress_test` exists to fix.

## Environment

`Dockerfile` builds `pydantic-bench:latest`: python 3.12-slim, `uv`, a Rust
toolchain (for pydantic-core workspace builds at 2025+ commits), and a full
pydantic clone baked in at `/opt/pydantic-src`.

`setup_task.sh <base_commit> [tests_commit]` reverts the codebase to the base
commit inside the container, erasing every later commit:

- `git archive` the base commit into `/repo` — a snapshot, so there is **no git
  history to mine the fix out of**; then `git init` a single commit tagged `BASE`
  so the agent's diff can be extracted later.
- for category 2, `tests_commit` swaps `tests/` for another commit's copy. The
  state is still a pure git snapshot; no patch file ever enters the container.
- delete `/opt/pydantic-src`, so nothing newer than the base commit remains.
- build the pinned venv and warm the bytecode and pytest caches.

`run_agent.py` then hands the agent its task description and **severs container
networking after setup** — with the network left up, agents did in fact cheat, by
fetching a newer pydantic from PyPI and by finding the fixing PR on GitHub. It
also extracts the working-tree diff from the container itself, rather than
trusting the agent to format a patch.

Everything under `answers*/` — gold patches, hidden test node IDs, buggy-variant
edits — stays on the host and is only ever read by the graders.

## Setup

Docker, git, Python 3.12, [`uv`](https://docs.astral.sh/uv/), and mini-swe-agent
(`uv tool install mini-swe-agent`). `valgrind` only for `mine_perf.py measure --ir`.

Credentials come from the environment or a `.env` file next to these scripts:
`GITHUB_TOKEN` to mine, `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` to judge and to
benchmark.

```bash
git clone https://github.com/pydantic/pydantic.git pydantic-clone
docker build -t pydantic-bench:latest .
```

The clone (~440MB) is not in this archive; it is an unmodified copy of a public
repo. Everything else — mining, grading, worktrees — reads from it, and the image
bakes it in.

## Category 1 — general feature / spec implementation

```bash
python3 mine.py harvest              # GitHub GraphQL -> work/candidates.jsonl
python3 mine.py extract              # local git      -> work/extracted.jsonl
python3 mine.py validate --workers 4 # pytest         -> work/validated.jsonl  (~40 min)
python3 mine.py assemble             #                -> tasks/ + answers/
python3 mine.py clean                # delete template boilerplate from the prompts
python3 mine.py judge                # LLM judge: FAIR / BORDERLINE / UNFAIR
# probe with a cheap model first (see "Running and scoring"), then:
python3 mine.py filter --probe-run haiku-probe
```

**Harvest** keeps small merged PRs from 2024 onwards touching both `pydantic/`
and `tests/`; `pydantic-core/` PRs are dropped, since the venv installs it as a
built wheel and patching its Rust source silently no-ops. **Validate** runs each
candidate three times in a disposable worktree and keeps only tests that flip
deterministically, so flakes and environment drift are never charged to the
agent. **Judge** rates whether the hidden tests assert something the prompt
cannot determine — invented wording, an arbitrary API name. **Filter** is the
anti-saturation lever: keep only tasks a cheap model (claude-haiku-4.5) failed,
minus anything rated UNFAIR, which is why the shipped set is much smaller than
the validated pool.

## Category 2 — stress-testing test cases

```bash
python3 mine_suite.py select --n 15  # draws from the validated PRs mine.py did not ship
python3 mine_suite.py variants       # one Opus 5 call per task, 8 proposals each
python3 mine_suite.py validate       # prove each variant plausible AND wrong
python3 mine_suite.py assemble       #                -> tasks_suite/ + answers_suite/
python3 mine.py clean --tasks-dir tasks_suite        # same boilerplate pass
```

The buggy implementations are LLM-generated from the PR and then **proven**
rather than trusted: each must keep the PR's pre-existing tests green (otherwise
the repo's own suite would already catch it) and must fail its own witness test.
Anything unprovable is discarded, so a task ships ~5–8 variants rather than a
fixed count; the ones the PR's own tests would have missed are what keeps the
category unsaturated. Sources are validated PRs that category 1 did *not* ship,
so the two share no PR. `--set NAME` builds a second set with a different
generator model — if the generating model also scores highest, that lead may be
familiarity with its own idea of a plausible mistake rather than better test
design.

## Category 3 — performance optimization

```bash
python3 mine_perf.py harvest         # perf-labelled PRs
python3 mine_perf.py measure --pr 13523 11244 7947   # triage: does it beat the noise floor?
python3 mine_perf.py calibrate       # each task's driver on base vs gold
python3 mine_perf.py assemble --blind
```

**Wall clock is not good enough on its own.** The same commit run twice varies by
up to 3.8% — the magnitude of a typical constant-factor perf PR. `measure`
therefore also supports callgrind instruction counts, deterministic to ~3 parts
per million at ~25s a run; use wall clock only when the gold speedup is ≥ 1.20x,
and calibrate on an idle machine (one task's noise floor went from 1.0067x quiet
to 1.074x under load, enough to swallow its 1.275x signal).

Each task ships a self-contained driver printing elapsed milliseconds, so the
metric does not depend on the repo's benchmark files existing at the base commit.
`--blind` emits a variant with no benchmark, so the agent must build its own
repro; everything else is held equal, so the delta isolates the effect of handing
the agent the thing it is measured on.

## Running and scoring

```bash
MSA=~/.local/share/uv/tools/mini-swe-agent/bin/python

# category 1 (--tasks-dir defaults to tasks/; omit --ids-file to run everything)
$MSA run_agent.py --model anthropic/claude-sonnet-5 --run-name sonnet5 \
    --workers 4 --step-limit 200 --cost-limit 2.0
python3 grade.py --run-name sonnet5                     # -> work/scores/sonnet5.json

# category 2
$MSA run_agent.py --model anthropic/claude-sonnet-5 --run-name sonnet5-suite \
    --tasks-dir tasks_suite --step-limit 80
python3 grade.py --run-name sonnet5-suite --answers-dir answers_suite

# category 3
$MSA run_agent.py --model anthropic/claude-sonnet-5 --run-name sonnet5-perf \
    --tasks-dir tasks_perf --step-limit 200
python3 grade_perf.py --run-name sonnet5-perf
```

Grading runs **host-side**, in fresh git worktrees built from the clone with the
same pinned recipe as mining, so nothing the agent did to its own environment can
affect its score. Agent edits outside the writable area are discarded: `tests/`
is reset for categories 1 and 3, source edits are ignored for category 2.

How each score is computed:

- **Category 1** — the fraction of the PR's fail-to-pass tests that now pass,
  multiplied by the fraction of its pass-to-pass tests still passing and by a
  wider regression check over related test files.
- **Category 2** — the fraction of buggy implementations the suite fails on, times
  the fraction of the agent's generated tests that actually pass against the correct
  code. The second factor is there to ensure that the tests are not overly broad in catching edge cases (e.g. a test that always asserts False).
- **Category 3** — 0 or 1, with no partial credit. It is 1 only if every test
  passing at base still passes, the change is as fast as the maintainers' own
  (measured *in the same run*), and no held-out benchmark the agent never saw
  regresses. A crash counts as a failure — the cheap wrong answer to one of these
  tasks segfaults pydantic-core.

Before trusting any run, self-test the graders against the gold patches — the
perf grader scored the maintainers' own patch 0.0 on its first version, for two
separate reasons, both found this way:

```bash
python3 grade_perf.py --self-test perf-001              # expect ~1.0
python3 grade_perf.py --self-test perf-001 --no-sanity --patch <known-bad>.diff  # expect 0
```

## Files

```
Dockerfile          the evaluation environment
setup_task.sh       per-task container setup (base-commit snapshot + pinned venv)

mine.py             category 1, and the shared helpers (paths, JSONL I/O, the
                    introspection blocklist) the other scripts import
mine_suite.py       category 2, including generating the buggy implementations
mine_perf.py        category 3, including the measurement harness

run_agent.py        mini-swe-agent driver: container, isolation, diff extraction
grade.py            grader for categories 1 and 2
grade_perf.py       grader for category 3

report_plugin.py    pytest plugin dumping {nodeid: outcome} as JSON. Separate
                    because `pytest -p` needs an importable module; it is how
                    both the validator and the graders read test outcomes
                    without parsing pytest's console output.
drivers/            the six perf benchmarks, one per task, plus two profiling
                    drivers used during triage
swegym_pydantic.json  SWE-Gym's pydantic instances, used as a contamination
                    blocklist when assembling category 1
```

Generated, not shipped: `tasks*/`, `answers*/`, and `work/` (JSONL artifacts,
worktrees, venvs, agent trajectories, scores).
