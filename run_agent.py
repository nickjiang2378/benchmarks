#!/usr/bin/env python
"""Run sampled benchmark tasks with mini-swe-agent in Docker containers.

Must be run with mini-swe-agent's interpreter:
  ~/.local/share/uv/tools/mini-swe-agent/bin/python run_agent.py --model anthropic/claude-haiku-4-5-20251001

Per task: start a pydantic-bench container, run setup_task.sh <base_commit>
(history-free snapshot + venv), hand the task to DefaultAgent, then extract
the working-tree diff (and localization.json) directly from the container --
extraction does not depend on the agent formatting a patch correctly.

Writes: work/agent_out/<run>/<task_id>.json  (+ .traj.json trajectories)
"""

import argparse
import base64
import json
import subprocess
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from minisweagent.agents.default import DefaultAgent
from minisweagent.environments.docker import DockerEnvironment
from minisweagent.models import get_model

HERE = Path(__file__).resolve().parent
TASKS = HERE / "tasks"  # overridden by --tasks-dir

# Container setup (uv install, sometimes a Rust build) is the CPU/memory-heavy
# phase; agents themselves are API-latency-bound. Cap concurrent setups so a
# wide worker pool doesn't stack 20 installs into an OOM storm.
SETUP_SEMAPHORE = threading.Semaphore(4)

SYSTEM_TEMPLATE = """\
You are a software engineer solving a task in a Python repository checkout
by interacting with a Linux shell. Think briefly, then issue at least one
bash tool call per response. Every command runs in a fresh subshell in /repo;
directory changes and shell variables do not persist between commands."""

ENV_RULES = """\

## Environment

- The repository is at /repo (your working directory), with a ready virtualenv
  at /repo/.venv already on PATH -- `python -m pytest tests/test_x.py -x -q`
  just works. Do not reinstall dependencies.
- Every command runs in a fresh subshell; prefix with `cd /repo && ...` if in
  doubt. Use non-interactive commands only (no vi/less/pagers).
- Keep test runs targeted (single files), the full suite is slow."""

# perf tasks get the opposite rule: the test suite is off limits entirely. Scoring
# is zero-tolerance on regressions, so the temptation is to spend the whole budget
# running tests -- and the default rules above actively invite that by advertising
# pytest. Do not append both: they contradict each other.
ENV_RULES_PERF = """\

## Environment

- The repository is at /repo (your working directory), with a ready virtualenv
  at /repo/.venv already on PATH. Do not reinstall dependencies.
- Every command runs in a fresh subshell; prefix with `cd /repo && ...` if in
  doubt. Use non-interactive commands only (no vi/less/pagers).
- Do NOT run pytest, and do not run the test suite or any part of it. Your job
  here is to read and modify the library source. Reason about correctness from
  the code itself.
- You may read any file, search the tree, and run your own scripts to measure
  and profile."""

SUBMISSION_RULES = """\


## Submission

When you are completely done, submit by running exactly this single command
with no other command in the same response:

echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT

Your working tree is collected automatically; you do not need to produce a
patch file. You cannot continue working after submitting."""


# Recovered from the rendered prompts stored in work/agent_out/*-suite/*.traj.json
# after an over-broad edit here removed its definition: all four suite runs render
# byte-identical rules, so this is exactly the default environment + submission text.
COMMON_RULES_READONLY = ENV_RULES + SUBMISSION_RULES


def instance_template(task: dict) -> str:
    """Environment rules differ by task type; never append two sets that disagree.

    perf   pytest forbidden outright (see ENV_RULES_PERF)
    suite  pre-existing read-only rules
    other  default rules, which advertise pytest
    """
    if task["type"] == "perf":
        return "{{task}}\n" + ENV_RULES_PERF + SUBMISSION_RULES
    if task["type"] == "suite":
        return "{{task}}\n" + COMMON_RULES_READONLY
    return "{{task}}\n" + ENV_RULES + SUBMISSION_RULES


def load_task(task_id: str) -> dict:
    return json.load(open(TASKS / task_id / "task.json"))


def ship_asset(env, task_id: str, name: str) -> bool:
    """Copy a task asset (e.g. perf_bench.py) into /repo and fold it into BASE.

    It has to land *inside* the BASE commit, otherwise it shows up as an addition
    in the agent's extracted diff and looks like part of their change.
    """
    src = TASKS / task_id / name
    if not src.exists():
        return False
    blob = base64.b64encode(src.read_bytes()).decode()
    env.execute({"command": f"printf %s {blob} | base64 -d > /repo/{name}"}, timeout=120)
    env.execute({"command": f"cd /repo && git add {name} && "
                            "git commit -q --amend --no-edit && git tag -f BASE"},
                timeout=300)
    r = env.execute({"command": f"test -s /repo/{name} && echo ASSET_OK"}, timeout=60)
    return "ASSET_OK" in r.get("output", "")


def build_task_text(task: dict) -> str:
    # suite tasks describe a change that is already implemented, not a bug report.
    header = "The change" if task["type"] == "suite" else "The report"
    return f"{task['instructions']}\n\n## {header}\n\n{task['prompt']}"


def run_one(task_id: str, model_name: str, out_dir: Path, step_limit: int,
            cost_limit: float, allow_network: bool = False,
            model_kwargs: dict | None = None) -> dict:
    task = load_task(task_id)
    result = {"task_id": task_id, "type": task["type"],
              "base_commit": task["base_commit"]}
    env = None
    try:
        env = DockerEnvironment(
            image="pydantic-bench:latest",
            cwd="/repo",
            timeout=120,
            interpreter=["bash", "-c"],
            container_timeout="3h",
            env={
                "PATH": "/repo/.venv/bin:/root/.cargo/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
                "VIRTUAL_ENV": "/repo/.venv",
                "PAGER": "cat", "GIT_PAGER": "cat", "PIP_PROGRESS_BAR": "off",
                # NOTE: PYTHONDONTWRITEBYTECODE deliberately NOT set -- agents
                # invoke pytest dozens of times per task, and .pyc caching cuts
                # each invocation's import cost substantially (setup_task.sh
                # pre-compiles and warms the caches).
            },
        )
        # suite tasks start from the fixed source with tests/ rolled back to the
        # pre-PR commit; every other type passes no tests_commit and gets a plain
        # snapshot of base_commit.
        with SETUP_SEMAPHORE:
            setup = env.execute(
                {"command": f"setup_task.sh {task['base_commit']} "
                            f"{task.get('tests_commit', '')} 2>&1 | tail -3"},
                timeout=1800)
        if "SETUP_OK" not in setup.get("output", ""):
            result.update(status="setup_failed",
                          detail=setup.get("output", "")[-500:])
            return result

        # perf tasks ship their benchmark as a task asset; it must exist before the
        # agent starts and be part of BASE so it is not mistaken for their edit.
        if task.get("ships_benchmark") and not ship_asset(env, task_id, "perf_bench.py"):
            result.update(status="asset_failed", detail="perf_bench.py not delivered")
            return result

        # Sever network AFTER setup (which needs PyPI): agents must not fetch
        # future pydantic releases or the fixing PR from GitHub. Empirically
        # sonnet-5 did both when the network was left open.
        if not allow_network:
            r = subprocess.run(
                ["docker", "network", "disconnect", "bridge", env.container_id],
                capture_output=True, text=True)
            if r.returncode != 0:
                result.update(status="isolation_failed", detail=r.stderr[-300:])
                return result

        # Some models reject litellm's default params. gpt-5.6-* refuses
        # litellm's default `reasoning_effort` alongside function tools on
        # /v1/chat/completions, and mini-swe-agent always sends a bash tool, so an
        # explicit value is required: --model-kwargs '{"reasoning_effort":"high"}'.
        # Measured: minimal is rejected outright; none/low/medium/high/xhigh all work.
        model = get_model(model_name,
                          {"model_kwargs": model_kwargs} if model_kwargs else None)
        agent = DefaultAgent(
            model, env,
            system_template=SYSTEM_TEMPLATE,
            instance_template=instance_template(task),
            step_limit=step_limit,
            cost_limit=cost_limit,
            output_path=out_dir / f"{task_id}.traj.json",
        )
        extra = agent.run(task=build_task_text(task))
        result.update(
            status="ran",
            exit_status=extra.get("exit_status", "unknown"),
            n_calls=agent.n_calls,
            cost=round(agent.cost, 4),
        )
        # Extract the working tree regardless of how the run ended.
        diff = env.execute(
            {"command": "cd /repo && git add -A >/dev/null 2>&1; "
                        "git -c core.quotepath=false diff --cached BASE"},
            timeout=300)
        result["patch"] = diff.get("output", "")
        if task["type"] == "localize":
            loc = env.execute(
                {"command": "cat /repo/localization.json 2>/dev/null || echo MISSING"})
            result["localization"] = loc.get("output", "")
    except Exception:
        result.update(status="error", detail=traceback.format_exc()[-800:])
    finally:
        if env is not None:
            try:
                env.cleanup()
            except Exception:
                pass
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--step-limit", type=int, default=50)
    ap.add_argument("--cost-limit", type=float, default=1.0)
    global TASKS
    ap.add_argument("--only", nargs="*", help="subset of task ids")
    ap.add_argument("--allow-network", action="store_true",
                    help="leave container networking up after setup (default: severed)")
    ap.add_argument("--model-kwargs", default="",
                    help='JSON dict of extra API params, e.g. \'{"reasoning_effort":"none"}\'')
    ap.add_argument("--tasks-dir", default=str(TASKS))
    ap.add_argument("--ids-file", help="JSON list of task ids (default: every task "
                                       "in --tasks-dir)")
    args = ap.parse_args()

    TASKS = Path(args.tasks_dir)
    sample = (json.load(open(args.ids_file)) if args.ids_file
              else sorted(p.parent.name for p in TASKS.glob("*/task.json")))
    if args.only:
        sample = [s for s in sample if s in set(args.only)]
    out_dir = HERE / "work" / "agent_out" / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    pending = [s for s in sample if not (out_dir / f"{s}.json").exists()]
    print(f"{len(sample)} sampled, {len(pending)} to run "
          f"({args.workers}-wide, model={args.model})", flush=True)

    lock = threading.Lock()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(run_one, t, args.model, out_dir,
                            args.step_limit, args.cost_limit,
                            args.allow_network,
                            json.loads(args.model_kwargs) if args.model_kwargs else None): t
                for t in pending}
        for fut in as_completed(futs):
            r = fut.result()
            with lock:
                (out_dir / f"{r['task_id']}.json").write_text(
                    json.dumps(r, indent=1))
                print(f"  {r['task_id']}: {r.get('status')}/"
                      f"{r.get('exit_status','-')} calls={r.get('n_calls','-')} "
                      f"cost=${r.get('cost','-')} patch={len(r.get('patch') or '')}B",
                      flush=True)
    print("done")


if __name__ == "__main__":
    main()
