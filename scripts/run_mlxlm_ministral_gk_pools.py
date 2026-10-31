"""mlx-lm runner: Ministral-3-3B G_k (pless_renyi) sweep on LCB v6 full1046,
batches 1+2 (420 tasks), difficulty-pool batching (todos E26).

Matches the vLLM runner `scripts/run_pless_variants_ministral_3b_lcb_cuda.py`
where it affects results:
  * prompt ids come from transformers' `MistralCommonBackend` (mistral-common),
    the tokenizer vLLM resolves for this model in tokenizer_mode=auto -- verified
    identical to gpu2's vLLM prompt ids on all 420 tasks (2026-10-30);
  * same `MINISTRAL_SYSTEM_PROMPT` and `_user_message`;
  * output decoded with the same tokenizer, `skip_special_tokens=False`, then
    `_strip_think_content(marker="[/THINK]")`;
  * T=1.0, no top-p/top-k, 10 samples/task, 35000 new tokens, stop on EOS;
  * same output file names and record fields.

Easy rows: chunks of 120 per insert with at most 12 running (backfill); medium
chunks of 120 with at most 8 running (backfill); hard 10 with no backfill.
Runs ONE (config, phase) per process and exits; the shell wrapper relaunches.
Configs run in CONFIGS order (gk_0_1 last: the same-backend anchor).
Each batch gets a deterministic seed from its absolute row ranges
(`bench.mlx_difficulty_pools.batch_seed`): crash-free reruns reproduce, and a
resumed batch can never replay already-banked samples.

Env overrides (kill-tests / scratch only, never in production):
  MLXLM_MGK_OUT_DIR, MLXLM_MGK_MAX_NEW_TOKENS, MLXLM_MGK_ONLY_CONFIG,
  MLXLM_MGK_ONLY_PHASE, MLXLM_MGK_MAX_BATCHES, MLXLM_MGK_TASK_LIMIT,
  MLXLM_MGK_EASY_CAP / _EASY_BACKFILL / _MEDIUM_CAP / _MEDIUM_BACKFILL / _HARD_CAP, MLXLM_MGK_OMLX_SITE_PACKAGES,
  MLXLM_MGK_ALLOW_DIRTY=1 (skip the git check, local dev only).
"""
from __future__ import annotations

import csv
import fcntl
import gc
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
# APPEND, never prepend: this venv's datasets==2.21.0 must win over omlx's.
sys.path.append(os.environ.get(
    "MLXLM_MGK_OMLX_SITE_PACKAGES",
    "/Users/karrtikiyer/projects/omlx/.venv/lib/python3.12/site-packages",
))

from bench.generator import _strip_think_content  # noqa: E402
from bench.livecodebench.dataset import load_lcb  # noqa: E402
from bench.livecodebench.prompts import _user_message  # noqa: E402
from bench.mlx_difficulty_pools import (  # noqa: E402
    EASY_CAP_DEFAULT, HARD_CAP_DEFAULT, MEDIUM_CAP_DEFAULT,
    PoolBatch, assign_row_ranges, batch_seed, build_pool_batches, select_next_phase,
)
from bench.mlx_pless_logits_processors import make_pless_renyi_logits_processor  # noqa: E402
from bench.mlx_seq_groups_resume import (  # noqa: E402
    assemble_record, banked_sample_record, load_banked_samples,
    remaining_samples_needed, truncate_to_last_newline,
)


def _vllm_runner_system_prompt() -> str:
    """The exact system prompt string the vLLM runner uses (imported, not copied)."""
    argv, sys.argv = sys.argv, [sys.argv[0]]
    try:
        from scripts.run_pless_variants_ministral_3b_lcb_cuda import MINISTRAL_SYSTEM_PROMPT
    finally:
        sys.argv = argv
    return MINISTRAL_SYSTEM_PROMPT


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


MODEL_ID = "mistralai/Ministral-3-3B-Reasoning-2512"
THINK_END_MARKER = "[/THINK]"
N_SAMPLES = 10
TEMPERATURE = 1.0
BASE_SEED = 42
BATCHES = (1, 2)
PLATFORMS = ("atcoder", "leetcode")
MAX_NEW_TOKENS = _env_int("MLXLM_MGK_MAX_NEW_TOKENS", 35000)
# Easy: chunks of 120 rows per insert, at most 12 running at once (mlx-lm backfills
# the rest as rows finish). Worst case ~6.9GB + 12 x ~4.3GB steady; a brief copy when
# a row finishes can exceed 83.49GB if a long easy row is resident -> crash + resume.
EASY_CAP = _env_int("MLXLM_MGK_EASY_CAP", 120)
EASY_BACKFILL = _env_int("MLXLM_MGK_EASY_BACKFILL", 12)
# Medium: chunks of 120 rows per insert, at most 8 running (backfill). Worst case
# ~6.9GB + (2x8-1) x ~4.3GB ~= 71GB < 83.49GB even with every resident row long, so
# no OOM crashes (crash-resume redraws killed long rows and biases pass@1 up).
MEDIUM_CAP = _env_int("MLXLM_MGK_MEDIUM_CAP", 120)
MEDIUM_BACKFILL = _env_int("MLXLM_MGK_MEDIUM_BACKFILL", 8)
HARD_CAP = _env_int("MLXLM_MGK_HARD_CAP", HARD_CAP_DEFAULT)
ONLY_CONFIG = os.environ.get("MLXLM_MGK_ONLY_CONFIG") or None
ONLY_PHASE = os.environ.get("MLXLM_MGK_ONLY_PHASE") or None
MAX_BATCHES = _env_int("MLXLM_MGK_MAX_BATCHES", 0) or None
TASK_LIMIT = _env_int("MLXLM_MGK_TASK_LIMIT", 0) or None
POLL_EVERY_N_STEPS = 50
RUN_ID = str(int(time.time()))
# Set by the wrapper: consecutive attempts that banked nothing. >0 salts the batch
# seed so an OOM before any row finished is not replayed bit-for-bit forever.
RETRY = _env_int("MLXLM_MGK_RETRY", 0)

# (config name, renyi k). Order is run order; gk_0_1 last on purpose.
CONFIGS: list[tuple[str, float]] = [
    ("gk_0_025", 0.025), ("gk_0_075", 0.075), ("gk_0_15", 0.15), ("gk_0_25", 0.25),
    ("gk_0_3", 0.3), ("gk_0_35", 0.35), ("gk_0_45", 0.45),
    ("gk_0_1", 0.1),
]

OUT_DIR = Path(os.environ.get(
    "MLXLM_MGK_OUT_DIR",
    str(REPO / "results/pless_variants_mlxlm_pools/mistralai--Ministral-3-3B-Reasoning-2512/LCB_v6_full1046"),
))
PARTIAL_DIR = OUT_DIR / "_partial"
# One line per row still unfinished when a batch dies (OOM etc.): the rows that
# get redrawn on resume. Lets crash-redraw bias be audited per config.
KILLED_DIR = OUT_DIR / "_killed"
BATCH_ASSIGNMENT_CSV = REPO / "results/lcb_v6_full1046/batch_assignment.csv"

_GATED_FILES = (
    "bench/mlx_difficulty_pools.py",
    "bench/mlx_seq_groups_resume.py",
    "bench/mlx_pless_logits_processors.py",
    "scripts/run_mlxlm_ministral_gk_pools.py",
    "scripts/run_pless_variants_ministral_3b_lcb_cuda.py",
)


def log(msg: str) -> None:
    print(msg, flush=True)


def out_path(config: str, platform: str) -> Path:
    return OUT_DIR / f"{config}_{platform}_think_t{TEMPERATURE}.jsonl"


def partial_path(config: str, platform: str) -> Path:
    return PARTIAL_DIR / f"{config}_{platform}_think_t{TEMPERATURE}.partial.jsonl"


def killed_path(config: str, platform: str) -> Path:
    return KILLED_DIR / f"{config}_{platform}_think_t{TEMPERATURE}.killed.jsonl"


def _write_killed_lines(config, platform, lines):
    KILLED_DIR.mkdir(parents=True, exist_ok=True)
    with killed_path(config, platform).open("a") as fh:
        fh.write("".join(lines))


def _log_killed_rows(config, k, phase, seed, attempt_id, toks, banked_uids, slot, problem_by_id, exc):
    """Append one line per row not banked when the batch died; these are exactly
    the rows redrawn on resume. Audit by count per task, not row_index (resume
    renumbers from the banked count). Never raises: a logging failure must not
    hide the original error."""
    try:
        ts = datetime.now(timezone.utc).isoformat()
        by_platform: dict[str, list[str]] = {}
        for uid, (tid, row) in slot.items():
            if uid in banked_uids:
                continue
            rec = {"task_id": tid, "row_index": row, "config": config, "renyi_k": k, "phase": phase,
                   "seed": seed, "attempt_id": attempt_id, "n_tokens_so_far": len(toks[uid]),
                   "started": len(toks[uid]) > 0, "error": repr(exc)[:300], "timestamp": ts}
            by_platform.setdefault(problem_by_id[tid].platform, []).append(json.dumps(rec) + "\n")
        for platform, lines in by_platform.items():
            _write_killed_lines(config, platform, lines)
        log(f"  KILLED: logged {sum(map(len, by_platform.values()))} unbanked row(s) to {KILLED_DIR} "
            f"before exiting ({exc!r:.120})")
    except Exception as log_exc:
        log(f"  KILLED: FAILED to log unbanked rows ({log_exc!r:.120}); original error follows")


def _check_deployed_code() -> None:
    for rel in _GATED_FILES:
        log(f"{rel} sha256={hashlib.sha256((REPO / rel).read_bytes()).hexdigest()}")
    head = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                          capture_output=True, text=True).stdout.strip()
    log(f"git HEAD={head or '<unknown>'}")
    if os.environ.get("MLXLM_MGK_ALLOW_DIRTY") == "1":
        log("MLXLM_MGK_ALLOW_DIRTY=1 -- skipping git-cleanliness check.")
        return
    tracked = subprocess.run(["git", "-C", str(REPO), "ls-files", "--error-unmatch", "--", *_GATED_FILES],
                             capture_output=True, text=True)
    status = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain", "--", *_GATED_FILES],
                            capture_output=True, text=True)
    if tracked.returncode != 0 or status.stdout.strip():
        raise RuntimeError(
            "gated files are untracked or differ from git HEAD -- refusing to run "
            f"uncommitted code. Deploy via commit + git pull.\n{tracked.stderr}{status.stdout}")


def _load_done(config: str) -> set[str]:
    done: set[str] = set()
    for plat in PLATFORMS:
        p = out_path(config, plat)
        if p.exists():
            for line in p.open():
                if line.strip():
                    done.add(json.loads(line)["task_id"])
    return done


def _load_banked(config: str) -> dict[str, list[dict]]:
    banked: dict[str, list[dict]] = {}
    for plat in PLATFORMS:
        for tid, recs in load_banked_samples(partial_path(config, plat)).items():
            banked.setdefault(tid, []).extend(recs)
    return banked


def _base_record_fields(config: str, k: float, problem) -> dict:
    return {
        "model": MODEL_ID, "backend": "mlx-lm", "method": config,
        "sampler_name": "pless_renyi", "renyi_k": k,
        "temperature": TEMPERATURE, "top_p": 1.0, "top_k": 0,
        "source": problem.platform, "difficulty": problem.difficulty,
        "contest_date": problem.contest_date, "prompt_text": problem.question,
        "enable_thinking": True,
        "max_new_tokens": MAX_NEW_TOKENS,
        "architecture": "mlx_difficulty_pools_v3_easy_medium_backfill",
        "pool_caps": {"easy": EASY_CAP, "medium": MEDIUM_CAP, "hard": HARD_CAP},
        "easy_backfill": EASY_BACKFILL,
        "medium_backfill": MEDIUM_BACKFILL,
    }


def _write_record(config: str, k: float, problem, banked: list[dict], fresh: list[dict]) -> None:
    rec = assemble_record(problem.task_id, banked, fresh, N_SAMPLES,
                          _base_record_fields(config, k, problem))
    rec["row_indices"] = [s.get("row_index") for s in (banked + fresh)[:N_SAMPLES]]
    rec["timestamp"] = datetime.now(timezone.utc).isoformat()
    with out_path(config, problem.platform).open("a") as fh:
        fh.write(json.dumps(rec) + "\n")
    log(f"  [{config}/{problem.platform}] task {problem.task_id} DONE")


def _load_problems() -> list:
    batch_of = {r["task_id"]: int(r["batch"]) for r in csv.DictReader(BATCH_ASSIGNMENT_CSV.open())}
    problems = []
    for plat in PLATFORMS:
        problems += [p for p in load_lcb(platforms=(plat,), with_tests=False)
                     if batch_of.get(p.task_id) in BATCHES]
    problems.sort(key=lambda p: (p.platform, p.task_id))
    if TASK_LIMIT is None:
        assert len(problems) == 420, f"expected 420 batch 1+2 tasks, got {len(problems)}"
    else:
        problems = problems[:TASK_LIMIT]
    return problems


def _plan(problems: list) -> tuple[str, float, list[PoolBatch], list, dict] | None:
    """First (config, phase) in run order that still has work."""
    for config, k in CONFIGS:
        if ONLY_CONFIG is not None and config != ONLY_CONFIG:
            continue
        done, banked = _load_done(config), _load_banked(config)
        tasks = []
        for p in problems:
            rem = remaining_samples_needed(p.task_id, done, banked, N_SAMPLES)
            if rem == 0 and p.task_id not in done:  # fully banked, never assembled
                _write_record(config, k, p, banked[p.task_id], [])
                done.add(p.task_id)
                continue
            tasks.append((p.task_id, p.difficulty, rem))
        batches = select_next_phase(
            build_pool_batches(tasks, easy_cap=EASY_CAP, medium_cap=MEDIUM_CAP, hard_cap=HARD_CAP,
                               easy_backfill=EASY_BACKFILL, medium_backfill=MEDIUM_BACKFILL),
            only_phase=ONLY_PHASE,
        )
        if batches:
            counts = {tid: len(v) for tid, v in banked.items()}
            return config, k, batches, assign_row_ranges(batches, counts), banked
    return None


def run_batch(model, tok, eos_ids, config, k, label, batch, ranges, seed,
              problem_by_id, prompt_ids, banked):
    import mlx.core as mx
    from mlx_lm.generate import BatchGenerator
    from mlx_lm.sample_utils import make_sampler

    mx.random.seed(seed)
    gen = BatchGenerator(
        model,
        stop_tokens=[[t] for t in eos_ids],
        sampler=make_sampler(temp=TEMPERATURE, top_p=1.0, top_k=0),
        logits_processors=[make_pless_renyi_logits_processor(k)],
        max_tokens=MAX_NEW_TOKENS,
        completion_batch_size=batch.completion_batch_size,
        prefill_batch_size=batch.prefill_batch_size,
    )
    try:
        prompts, row_of = [], []
        for tid, start, n in ranges:
            for j in range(n):
                prompts.append(prompt_ids[tid])
                row_of.append((tid, start + j))
        log(f"\n=== {label}: {ranges} ({len(prompts)} rows, "
            f"completion_batch_size={batch.completion_batch_size}, "
            f"prefill_batch_size={batch.prefill_batch_size}, seed={seed}, "
            f"active_before={mx.get_active_memory()/1e9:.3f}GB) ===")
        mx.reset_peak_memory()
        uids = gen.insert(prompts, [MAX_NEW_TOKENS] * len(prompts))
        toks = {u: [] for u in uids}
        finished = {u: False for u in uids}
        slot = dict(zip(uids, row_of))
        fresh: dict[str, list[dict]] = {tid: [] for tid, _, _ in ranges}
        written: set[str] = set()
        banked_uids: set[int] = set()
        attempt_id = f"{config}:{label.replace(' ', '')}_run{RUN_ID}"
        t0, step, max_resident = time.time(), 0, 0
        try:
            while responses := gen.next_generated():
                step += 1
                max_resident = max(max_resident, len(responses))
                for r in responses:
                    if r.uid not in toks:
                        continue
                    if r.finish_reason != "stop":
                        toks[r.uid].append(r.token)
                    if r.finish_reason is None or finished[r.uid]:
                        continue
                    finished[r.uid] = True
                    tid, row = slot[r.uid]
                    problem = problem_by_id[tid]
                    text = tok.decode(toks[r.uid], skip_special_tokens=False)
                    rec = banked_sample_record(tid, text, _strip_think_content(text, marker=THINK_END_MARKER),
                                               r.finish_reason, len(toks[r.uid]), attempt_id, seed=seed)
                    rec["row_index"] = row
                    with partial_path(config, problem.platform).open("a") as pf:
                        pf.write(json.dumps(rec) + "\n")
                    banked_uids.add(r.uid)
                    fresh[tid].append(rec)
                    prior = banked.get(tid, [])
                    if len(prior) + len(fresh[tid]) >= N_SAMPLES and tid not in written:
                        _write_record(config, k, problem, prior, fresh[tid])
                        written.add(tid)
                if step % POLL_EVERY_N_STEPS == 0:
                    log(f"    step={step:6d} active={mx.get_active_memory()/1e9:7.3f}GB "
                        f"peak={mx.get_peak_memory()/1e9:7.3f}GB "
                        f"samples_done={sum(finished.values())}/{len(uids)} resident={len(responses)} "
                        f"tasks_written={len(written)} elapsed={time.time()-t0:.0f}s")
        except BaseException as exc:
            # Rows still unfinished here are lost and redrawn on resume; log them first.
            _log_killed_rows(config, k, batch.phase, seed, attempt_id, toks, banked_uids, slot,
                             problem_by_id, exc)
            raise
        log(f"  batch drain complete: steps={step} max_resident={max_resident} elapsed={time.time()-t0:.0f}s "
            f"peak_during={mx.get_peak_memory()/1e9:.3f}GB")
        expected = {tid for tid, _, n in ranges if len(banked.get(tid, [])) + n >= N_SAMPLES}
        if not expected <= written:
            raise RuntimeError(f"{label}: {expected - written} reached N_SAMPLES but were not written")
    finally:
        gen.close()
        del gen
        gc.collect()
        mx.clear_cache()
        log(f"  post-cleanup active={mx.get_active_memory()/1e9:.3f}GB")


def main() -> None:
    _check_deployed_code()
    PARTIAL_DIR.mkdir(parents=True, exist_ok=True)
    lock = (OUT_DIR / ".runner.lock").open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit(f"another runner holds {OUT_DIR}/.runner.lock -- refusing to start (would duplicate records)")
    for config, _ in CONFIGS:
        for plat in PLATFORMS:
            for p in (out_path(config, plat), partial_path(config, plat)):
                if (dropped := truncate_to_last_newline(p)):
                    log(f"WARNING: {p} did not end cleanly, dropped {dropped} trailing bytes.")

    log(f"CONFIG: OUT_DIR={OUT_DIR} MAX_NEW_TOKENS={MAX_NEW_TOKENS} "
        f"CAPS easy={EASY_CAP} (backfill {EASY_BACKFILL}) medium={MEDIUM_CAP} (backfill {MEDIUM_BACKFILL}) hard={HARD_CAP} "
        f"ONLY_CONFIG={ONLY_CONFIG} ONLY_PHASE={ONLY_PHASE} MAX_BATCHES={MAX_BATCHES} TASK_LIMIT={TASK_LIMIT} RETRY={RETRY}")
    problems = _load_problems()
    problem_by_id = {p.task_id: p for p in problems}

    planned = _plan(problems)
    if planned is None:
        log("ALL_CONFIGS_DONE")
        return
    config, k, batches, ranges, banked = planned
    phase = batches[0].phase
    if MAX_BATCHES is not None:
        batches, ranges = batches[:MAX_BATCHES], ranges[:MAX_BATCHES]
    log(f"PHASE_START {config} (k={k}) {phase}: {len(batches)} batch(es), "
        f"{sum(n for b in batches for _, n in b.rows)} rows")

    import mlx.core as mx
    from mlx_lm import load
    from transformers.tokenization_mistral_common import MistralCommonBackend

    tok = MistralCommonBackend.from_pretrained(MODEL_ID)
    model, mlx_tok = load(path_or_hf_repo=MODEL_ID)
    eos_ids = sorted({tok.eos_token_id})
    if eos_ids != sorted(mlx_tok.eos_token_ids):
        raise RuntimeError(f"EOS mismatch: mistral-common {eos_ids} vs mlx-lm {sorted(mlx_tok.eos_token_ids)}")
    log(f"Model loaded: active={mx.get_active_memory()/1e9:.3f}GB eos={eos_ids} RUN_ID={RUN_ID}")

    system_prompt = _vllm_runner_system_prompt()
    prompt_ids = {
        p.task_id: list(tok.apply_chat_template(
            conversation=[{"role": "system", "content": system_prompt},
                          {"role": "user", "content": _user_message(p)}],
            tokenize=True, return_dict=False))
        for p in problems
    }

    for i, (batch, rr) in enumerate(zip(batches, ranges)):
        seed = batch_seed(BASE_SEED, config, phase if RETRY == 0 else f"{phase}#retry{RETRY}", rr)
        run_batch(model, tok, eos_ids, config, k, f"{phase} batch {i+1}/{len(batches)}",
                  batch, rr, seed, problem_by_id, prompt_ids, banked)
        banked = _load_banked(config)
    log(f"PHASE_DONE {config} {phase} -- exiting for a fresh process.")


if __name__ == "__main__":
    main()
