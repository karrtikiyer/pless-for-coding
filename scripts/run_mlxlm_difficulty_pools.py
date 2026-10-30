"""mlx-lm difficulty-pool runner (todos E24,
docs/mlx_seq_groups_difficulty_pool_scheduling_proposal.md).

pless_default, Qwen3-8B, the 60 LCB_v6_stratified60 tasks, 10 samples each.
Runs ONE phase per process (easy -> medium -> hard) and exits;
`scripts/run_mlxlm_difficulty_pools.sh` relaunches it per phase so mlx's
allocator resets between phases.

Every batch: one `insert()`, drained to completion, no backfill
(`completion_batch_size >= rows`), fresh BatchGenerator per batch. Caps come
from `bench.mlx_difficulty_pools` (easy 25 / medium 20 / hard 10).

Sample-level resume reuses `bench.mlx_seq_groups_resume` UNCHANGED (banking
sidecar + record assembly): every finished sample is appended to
`_partial/` the moment it finishes; a task's record is written once it has 10.

Env overrides (for kill-tests against a scratch dir; never set in production):
  MLXLM_POOLS_OUT_DIR, MLXLM_POOLS_MAX_NEW_TOKENS, MLXLM_POOLS_ONLY_PHASE,
  MLXLM_POOLS_MAX_BATCHES, MLXLM_POOLS_EASY_CAP / _MEDIUM_CAP / _HARD_CAP,
  MLXLM_POOLS_OMLX_SITE_PACKAGES, MLXLM_POOLS_CHECKPOINT,
  MLXLM_POOLS_ALLOW_DIRTY=1 (skip the git-cleanliness check, local dev only).
"""
from __future__ import annotations

import gc
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
# mlx/mlx-lm live only in the separate omlx venv. APPEND, never prepend: this
# repo's venv has the datasets==2.21.0 that load_lcb()'s cache was built with,
# and omlx's datasets 5.x must not shadow it.
sys.path.append(os.environ.get(
    "MLXLM_POOLS_OMLX_SITE_PACKAGES",
    "/Users/karrtikiyer/projects/omlx/.venv/lib/python3.12/site-packages",
))

from bench.generator import _strip_think_content  # noqa: E402
from bench.livecodebench.dataset import load_lcb  # noqa: E402
from bench.livecodebench.prompts import _SYSTEM, _user_message  # noqa: E402
from bench.mlx_difficulty_pools import (  # noqa: E402
    EASY_CAP_DEFAULT, HARD_CAP_DEFAULT, MEDIUM_CAP_DEFAULT,
    PoolBatch, build_pool_batches, select_next_phase,
)
from bench.mlx_pless_logits_processors import make_pless_logits_processor  # noqa: E402
from bench.mlx_seq_groups_resume import (  # noqa: E402
    assemble_record, banked_sample_record, load_banked_samples,
    remaining_samples_needed, truncate_to_last_newline,
)

from mlx_lm import load  # noqa: E402
from mlx_lm.generate import BatchGenerator  # noqa: E402
from mlx_lm.sample_utils import make_sampler  # noqa: E402
import mlx.core as mx  # noqa: E402


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


CHECKPOINT_PATH = os.environ.get("MLXLM_POOLS_CHECKPOINT", "/Users/karrtikiyer/omlx_models/Qwen3-8B-bf16")
MODEL_ID = "Qwen/Qwen3-8B"
N_SAMPLES = 10
TEMPERATURE = 1.0
SEED = 42
MAX_NEW_TOKENS = _env_int("MLXLM_POOLS_MAX_NEW_TOKENS", 25000)
EASY_CAP = _env_int("MLXLM_POOLS_EASY_CAP", EASY_CAP_DEFAULT)
MEDIUM_CAP = _env_int("MLXLM_POOLS_MEDIUM_CAP", MEDIUM_CAP_DEFAULT)
HARD_CAP = _env_int("MLXLM_POOLS_HARD_CAP", HARD_CAP_DEFAULT)
ONLY_PHASE = os.environ.get("MLXLM_POOLS_ONLY_PHASE") or None
MAX_BATCHES = _env_int("MLXLM_POOLS_MAX_BATCHES", 0) or None
POLL_EVERY_N_STEPS = 50
RUN_ID = str(int(time.time()))
# Per-process seed: a bare SEED would replay the same RNG state on every
# restart and risk duplicate samples for the first resumed batch.
RUN_SEED = SEED + (int(RUN_ID) % 100_000)

OUT_DIR = Path(os.environ.get(
    "MLXLM_POOLS_OUT_DIR",
    str(REPO / "results/pless_variants_mlxlm_pools/Qwen--Qwen3-8B/LCB_v6_stratified60"),
))
PLATFORMS = ("atcoder", "leetcode")
OUT_PATHS = {p: OUT_DIR / f"pless_default_{p}_think_t1.0.jsonl" for p in PLATFORMS}
PARTIAL_DIR = OUT_DIR / "_partial"
PARTIAL_PATHS = {p: PARTIAL_DIR / f"pless_default_{p}_think_t1.0.partial.jsonl" for p in PLATFORMS}
BASELINE_DIR = REPO / "results/pless_variants_cuda/Qwen--Qwen3-8B/LCB_v6_stratified60"

_GATED_FILES = (
    "bench/mlx_difficulty_pools.py",
    "bench/mlx_seq_groups_resume.py",
    "bench/mlx_pless_logits_processors.py",
    "scripts/run_mlxlm_difficulty_pools.py",
)


def log(msg: str) -> None:
    print(msg, flush=True)


def _check_deployed_code() -> None:
    """Log hashes + git HEAD and refuse to run if the scheduling/bookkeeping
    files differ from HEAD: deploys go through `git pull`, so a dirty file
    means this machine is running code that was not committed and tested."""
    for rel in _GATED_FILES:
        digest = hashlib.sha256((REPO / rel).read_bytes()).hexdigest()
        log(f"{rel} sha256={digest}")
    head = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                          capture_output=True, text=True).stdout.strip()
    log(f"git HEAD={head or '<unknown>'}")
    if os.environ.get("MLXLM_POOLS_ALLOW_DIRTY") == "1":
        log("MLXLM_POOLS_ALLOW_DIRTY=1 -- skipping git-cleanliness check.")
        return
    # `git diff HEAD` ignores untracked files, so check tracked-ness and status explicitly.
    tracked = subprocess.run(["git", "-C", str(REPO), "ls-files", "--error-unmatch", "--", *_GATED_FILES],
                             capture_output=True, text=True)
    status = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain", "--", *_GATED_FILES],
                            capture_output=True, text=True)
    if tracked.returncode != 0 or status.stdout.strip():
        raise RuntimeError(
            f"gated files are untracked or differ from git HEAD on this machine -- refusing "
            f"to run uncommitted code. Deploy via commit + git pull.\n{tracked.stderr}{status.stdout}")


def _load_done_task_ids() -> set[str]:
    done: set[str] = set()
    for p in OUT_PATHS.values():
        if p.exists():
            with p.open() as f:
                for line in f:
                    if line.strip():
                        done.add(json.loads(line)["task_id"])
    return done


def _load_all_banked() -> dict[str, list[dict]]:
    banked: dict[str, list[dict]] = {}
    for p in PARTIAL_PATHS.values():
        for task_id, samples in load_banked_samples(p).items():
            banked.setdefault(task_id, []).extend(samples)
    return banked


def _base_record_fields(problem) -> dict:
    return {
        "model": MODEL_ID, "backend": "mlx-lm", "method": "pless_default",
        "sampler_name": "pless", "renyi_k": None,
        "temperature": TEMPERATURE, "top_p": 1.0, "top_k": -1,
        "source": problem.platform,
        "difficulty": problem.difficulty, "contest_date": problem.contest_date,
        "prompt_text": problem.question,
        "enable_thinking": True, "seed": SEED,
        "max_new_tokens": MAX_NEW_TOKENS,
        "architecture": "difficulty_pools_v1_no_backfill",
        "pool_caps": {"easy": EASY_CAP, "medium": MEDIUM_CAP, "hard": HARD_CAP},
    }


def _write_main_record(problem, record: dict) -> None:
    with OUT_PATHS[problem.platform].open("a") as fh:
        fh.write(json.dumps(record) + "\n")
    log(f"  [pless_default/{problem.platform}] task {problem.task_id} DONE "
        f"(attempt_ids={record['attempt_ids']})")


def run_batch(model, tokenizer, label: str, batch: PoolBatch,
              problem_by_id: dict, prompt_ids_by_task: dict,
              banked_by_task: dict[str, list[dict]]) -> None:
    gen = BatchGenerator(
        model,
        stop_tokens=[[t] for t in tokenizer.eos_token_ids],
        sampler=make_sampler(temp=TEMPERATURE, top_p=1.0, top_k=0),
        logits_processors=[make_pless_logits_processor()],
        max_tokens=MAX_NEW_TOKENS,
        completion_batch_size=batch.completion_batch_size,
    )
    try:
        prompts: list[list[int]] = []
        task_for_prompt: list = []
        for tid, n in batch.rows:
            for _ in range(n):
                prompts.append(prompt_ids_by_task[tid])
                task_for_prompt.append(problem_by_id[tid])

        log(f"\n=== {label}: {batch.rows} ({len(prompts)} rows, "
            f"completion_batch_size={batch.completion_batch_size}) "
            f"active_before={mx.get_active_memory()/1e9:.3f}GB "
            f"peak_before_reset={mx.get_peak_memory()/1e9:.3f}GB ===")
        mx.reset_peak_memory()

        uids = gen.insert(prompts, [MAX_NEW_TOKENS] * len(prompts))
        uid_tokens: dict[int, list[int]] = {uid: [] for uid in uids}
        uid_done: dict[int, bool] = {uid: False for uid in uids}
        uid_problem = dict(zip(uids, task_for_prompt))
        fresh: dict[str, list[dict]] = {tid: [] for tid, _ in batch.rows}
        written: set[str] = set()
        attempt_id = f"{label.replace(' ', '')}_run{RUN_ID}"

        t0 = time.time()
        step = 0
        with gen.stats():
            while responses := gen.next_generated():
                step += 1
                for r in responses:
                    if r.uid not in uid_tokens:
                        continue
                    if r.finish_reason != "stop":
                        uid_tokens[r.uid].append(r.token)
                    if r.finish_reason is None or uid_done[r.uid]:
                        continue
                    uid_done[r.uid] = True
                    problem = uid_problem[r.uid]
                    with_thinking = tokenizer.decode(uid_tokens[r.uid], skip_special_tokens=True)
                    rec = banked_sample_record(
                        problem.task_id, with_thinking, _strip_think_content(with_thinking),
                        r.finish_reason, len(uid_tokens[r.uid]), attempt_id, seed=RUN_SEED,
                    )
                    with PARTIAL_PATHS[problem.platform].open("a") as pf:
                        pf.write(json.dumps(rec) + "\n")
                    fresh[problem.task_id].append(rec)
                    banked = banked_by_task.get(problem.task_id, [])
                    if (len(banked) + len(fresh[problem.task_id]) >= N_SAMPLES
                            and problem.task_id not in written):
                        _write_main_record(problem, assemble_record(
                            problem.task_id, banked, fresh[problem.task_id],
                            N_SAMPLES, _base_record_fields(problem)))
                        written.add(problem.task_id)
                if step % POLL_EVERY_N_STEPS == 0:
                    log(f"    step={step:6d} active={mx.get_active_memory()/1e9:7.3f}GB "
                        f"peak={mx.get_peak_memory()/1e9:7.3f}GB "
                        f"cache={mx.get_cache_memory()/1e9:7.3f}GB "
                        f"samples_done={sum(uid_done.values())}/{len(uids)} "
                        f"tasks_written={len(written)} elapsed={time.time()-t0:.0f}s")

        log(f"  batch drain complete: steps={step} elapsed={time.time()-t0:.0f}s "
            f"peak_during={mx.get_peak_memory()/1e9:.3f}GB")
        # A task split across batches is legitimately incomplete after its
        # first batch; only assert the ones that reached N_SAMPLES here.
        expected = {tid for tid, n in batch.rows
                    if len(banked_by_task.get(tid, [])) + n >= N_SAMPLES}
        if not expected <= written:
            raise RuntimeError(f"{label}: tasks {expected - written} reached N_SAMPLES "
                               f"but were not written -- investigate before resuming.")
    finally:
        gen.close()
        del gen
        gc.collect()
        mx.clear_cache()
        log(f"  post-cleanup active={mx.get_active_memory()/1e9:.3f}GB "
            f"(baseline creep across batches in this phase)")


def main() -> None:
    _check_deployed_code()
    PARTIAL_DIR.mkdir(parents=True, exist_ok=True)
    for p in list(OUT_PATHS.values()) + list(PARTIAL_PATHS.values()):
        dropped = truncate_to_last_newline(p)
        if dropped:
            log(f"WARNING: {p} did not end cleanly, dropped {dropped} trailing bytes.")

    log(f"CONFIG: OUT_DIR={OUT_DIR} MAX_NEW_TOKENS={MAX_NEW_TOKENS} "
        f"CAPS easy={EASY_CAP} medium={MEDIUM_CAP} hard={HARD_CAP} "
        f"ONLY_PHASE={ONLY_PHASE} MAX_BATCHES={MAX_BATCHES}")

    stratified60: set[str] = set()
    for plat in PLATFORMS:
        with (BASELINE_DIR / f"pless_default_{plat}_think_t1.0.jsonl").open() as f:
            for line in f:
                stratified60.add(json.loads(line)["task_id"])
    problems = [p for p in load_lcb(platforms=PLATFORMS, with_tests=False)
                if p.task_id in stratified60]
    assert len(problems) == 60, f"expected 60 tasks, got {len(problems)}"
    problem_by_id = {p.task_id: p for p in problems}

    done_ids = _load_done_task_ids()
    banked_by_task = _load_all_banked()
    tasks: list[tuple[str, str, int]] = []
    for p in problems:
        rem = remaining_samples_needed(p.task_id, done_ids, banked_by_task, N_SAMPLES)
        if rem == 0 and p.task_id not in done_ids:
            # Fully banked but never assembled (crash after the 10th sample).
            _write_main_record(p, assemble_record(
                p.task_id, banked_by_task[p.task_id], [], N_SAMPLES, _base_record_fields(p)))
            done_ids.add(p.task_id)
            continue
        tasks.append((p.task_id, p.difficulty, rem))

    batches = select_next_phase(
        build_pool_batches(tasks, easy_cap=EASY_CAP, medium_cap=MEDIUM_CAP, hard_cap=HARD_CAP),
        only_phase=ONLY_PHASE,
    )
    log(f"{len(done_ids)}/60 tasks done, {len(banked_by_task)} with banked samples.")
    if not batches:
        log("ALL_PHASES_DONE" if ONLY_PHASE is None else f"PHASE_EMPTY {ONLY_PHASE}")
        return
    phase = batches[0].phase
    if MAX_BATCHES is not None:
        batches = batches[:MAX_BATCHES]
    log(f"PHASE_START {phase}: {len(batches)} batch(es), "
        f"{sum(n for b in batches for _, n in b.rows)} rows")

    log(f"Loading model: {CHECKPOINT_PATH}")
    model, tokenizer = load(path_or_hf_repo=CHECKPOINT_PATH)
    log(f"After model load: active={mx.get_active_memory()/1e9:.3f}GB "
        f"RUN_ID={RUN_ID} RUN_SEED={RUN_SEED}")
    mx.random.seed(RUN_SEED)

    prompt_ids_by_task = {
        p.task_id: tokenizer.apply_chat_template(
            [{"role": "system", "content": _SYSTEM},
             {"role": "user", "content": _user_message(p)}],
            add_generation_prompt=True, enable_thinking=True)
        for p in problems
    }

    for i, batch in enumerate(batches):
        run_batch(model, tokenizer, f"{phase} batch {i+1}/{len(batches)}", batch,
                  problem_by_id, prompt_ids_by_task, banked_by_task)
        # Next batch must see what this one banked/wrote (a task can span batches).
        banked_by_task = _load_all_banked()
    log(f"PHASE_DONE {phase} -- exiting for a fresh process.")


if __name__ == "__main__":
    main()
