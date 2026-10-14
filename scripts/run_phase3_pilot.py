"""Phase 3 pilot: multi-arm vs single-arm equivalence test for
bench/apps/multi_arm_runner.py vs bench/apps/runner.py.

Reviewed design (2026-10-14, multiple advisor rounds -- see conversation):
  - Model: mistralai/Ministral-3-3B-Reasoning-2512 (the one model already
    proven to fit on this exact GPU, via a real multi-day production job).
  - --seed 42, --n-samples 10, --max-new-tokens 35000 -- matching the real
    production job's own settings on this same machine/model (not a
    reduced pilot budget: an earlier, smaller budget was found to risk
    Ministral burning its whole budget on reasoning before any code, since
    enable_thinking=False is a no-op for this model -- it's a Qwen3-only
    chat-template flag).
  - 3 arms: pless (custom logits-processor path), temp / topp0.9 (native
    vLLM sampler path via generate_samples_standard_vllm) -- config-parity
    against runner.py's real CLI already verified by
    scripts/compare_apps_arm_config_parity.py.
  - 10 real APPS problems, stratified round-robin across all 6
    (source, difficulty) combinations.
  - A cache-miss-vs-cache-miss baseline pair (same arm, same ONE problem,
    two SEPARATE runner.py invocations with DISTINCT --results-dir) to
    establish vLLM's own self-determinism before trusting any multi-arm
    cache-hit-subset comparison -- needed because vLLM's own cache-hit
    GEMM-shape effect (GH #54490) can legitimately flip tokens at a fixed
    seed, so a cache-hit-subset mismatch is uninterpretable without this
    control.
  - Real num_cached_tokens instrumentation on the multi-arm run (via
    multi_arm_runner.py's --cache-stats-file, added this session) --
    cache-hit/miss classification is NOT inferred from loop position.

Output layout (all under --out-root, default results/_phase3_pilot/):
  single_arm/   -- bench.apps.runner, one process per (stratum, arm)
  multi_arm/    -- bench.apps.multi_arm_runner, one process per stratum
                   (all 3 arms internally), plus cache_stats_<stratum>.jsonl
  baseline_run1/, baseline_run2/ -- the cache-miss sanity pair

Worst-case time disclosed to the user before approval: 10 problems x 3 arms
x 2 paths + 2 baseline = 62 sequential engine.generate() calls, each up to
the full 35000-token budget; at this project's own observed production tail
latency (~55-60 min for a single slow task), worst case is ~57-62 hours.
Realistic time is expected to be much lower (most tasks finish far faster),
but the ceiling is real and the user has been told explicitly.

Usage::

    # Show every command that would run, without running anything:
    uv run python scripts/run_phase3_pilot.py --dry-run

    # Actually execute (only after the GPU has been freed -- see the
    # separate kill/restart plan for the production job sharing this GPU):
    uv run python scripts/run_phase3_pilot.py

    # Compare-only, once generation has finished:
    uv run python scripts/run_phase3_pilot.py --compare-only
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MODEL = "mistralai/Ministral-3-3B-Reasoning-2512"
SEED = 42
N_SAMPLES = 10
MAX_NEW_TOKENS = 35000
# Matches the production value in docs/theory/laptop_batch0_pause_checkpoint.md
# (--max-model-len 37500) -- proven to fit this model/GPU combo. Without this,
# vLLM defaults to Ministral-3's native 262144 context, which needs 26GiB of
# KV cache vs. the ~16GiB available (confirmed via a real engine-init crash).
MAX_MODEL_LEN = 37500
ARMS = ["pless", "temp", "topp0.9"]

# Stratified selection (round-robin across 6 strata, 10 total problems),
# fetched live via load_apps() on 2026-10-14 -- see conversation for the
# exact command used.
STRATA: dict[tuple[str, str], list[int]] = {
    ("ATCODER", "competition"): [3722, 3724],
    ("ATCODER", "interview"): [117, 160],
    ("ATCODER", "introductory"): [4035, 4048],
    ("CODEFORCES", "competition"): [3689, 3690],
    ("CODEFORCES", "interview"): [0],
    ("CODEFORCES", "introductory"): [4000],
}

# Baseline cache-miss-vs-cache-miss pair.
BASELINE_SOURCE, BASELINE_DIFFICULTY = "ATCODER", "introductory"
BASELINE_TASK_ID = 4035
BASELINE_ARM = "pless"  # method=pless, temperature=1.0, top_p=1.0, top_k=0


def _single_arm_cmd(out_root: Path, source: str, difficulty: str,
                     task_ids: list[int], method: str, extra: dict,
                     results_dir: Path) -> list[str]:
    cmd = [
        sys.executable, "-m", "bench.apps.runner",
        "--model", MODEL, "--source", source, "--difficulty", difficulty,
        "--method", method,
        "--task-ids", *[str(t) for t in task_ids],
        "--n-samples", str(N_SAMPLES), "--max-new-tokens", str(MAX_NEW_TOKENS),
        "--seed", str(SEED), "--backend", "vllm",
        "--results-dir", str(results_dir),
        "--treat-as-instruct", "--max-model-len", str(MAX_MODEL_LEN),
    ]
    if "temperature" in extra:
        cmd += ["--temperature", str(extra["temperature"])]
    if "top_p" in extra:
        cmd += ["--top-p", str(extra["top_p"])]
    if "top_k" in extra:
        cmd += ["--top-k", str(extra["top_k"])]
    return cmd


# (arm name) -> (method, extra CLI knobs) -- mirrors _preset_arm_spec exactly,
# re-derived independently here rather than imported, so this script doesn't
# silently inherit a future drift in multi_arm_runner.py's own preset table
# without a human noticing the two have diverged (compare_apps_arm_config_parity.py
# is the authority on whether they currently agree; this file does not re-run it).
ARM_TO_RUNNER_CLI: dict[str, tuple[str, dict]] = {
    "pless": ("pless", {"temperature": 1.0}),
    "temp": ("temp", {"temperature": 1.0}),
    "topp0.9": ("temp", {"temperature": 1.0, "top_p": 0.9, "top_k": 0}),
}


def _multi_arm_cmd(out_root: Path, source: str, difficulty: str,
                    task_ids: list[int], results_dir: Path,
                    cache_stats_file: Path) -> list[str]:
    return [
        sys.executable, "-m", "bench.apps.multi_arm_runner",
        "--model", MODEL, "--source", source, "--difficulty", difficulty,
        "--arms", *ARMS,
        "--task-ids", *[str(t) for t in task_ids],
        "--n-samples", str(N_SAMPLES), "--max-new-tokens", str(MAX_NEW_TOKENS),
        "--seed", str(SEED), "--backend", "vllm",
        "--results-dir", str(results_dir),
        "--cache-stats-file", str(cache_stats_file),
        "--treat-as-instruct", "--max-model-len", str(MAX_MODEL_LEN),
    ]


def build_all_commands(out_root: Path) -> list[tuple[str, list[str]]]:
    """Returns [(label, cmd), ...] in execution order."""
    single_dir = out_root / "single_arm"
    multi_dir = out_root / "multi_arm"
    cmds: list[tuple[str, list[str]]] = []

    for (source, difficulty), task_ids in STRATA.items():
        stratum_label = f"{source}_{difficulty}"
        cache_stats_file = multi_dir / f"cache_stats_{stratum_label}.jsonl"
        cmds.append((
            f"multi-arm[{stratum_label}]",
            _multi_arm_cmd(out_root, source, difficulty, task_ids, multi_dir, cache_stats_file),
        ))
        for arm_name in ARMS:
            method, extra = ARM_TO_RUNNER_CLI[arm_name]
            cmds.append((
                f"single-arm[{stratum_label}][{arm_name}]",
                _single_arm_cmd(out_root, source, difficulty, task_ids, method, extra, single_dir),
            ))

    # Baseline pair.
    method, extra = ARM_TO_RUNNER_CLI[BASELINE_ARM]
    for run_label in ["baseline_run1", "baseline_run2"]:
        cmds.append((
            run_label,
            _single_arm_cmd(
                out_root, BASELINE_SOURCE, BASELINE_DIFFICULTY, [BASELINE_TASK_ID],
                method, extra, out_root / run_label,
            ),
        ))
    return cmds


def _method_key_str(arm_name: str) -> str:
    """Calls the REAL bench.apps.runner._method_key on a minimal constructed
    Namespace -- NOT a second hand-duplicated copy of its logic. An earlier
    version of this function hardcoded stems independently and got them
    wrong (missed that _method_key only appends "_think_t{temp}" when
    enable_thinking=True, which this pilot never sets) -- exactly the kind
    of drift this file's own docstring elsewhere warns about avoiding by
    re-deriving things independently. Caught by review 2026-10-14; fixed by
    calling the real function instead of re-deriving it a second time."""
    from argparse import Namespace
    from bench.apps.runner import _method_key
    method, extra = ARM_TO_RUNNER_CLI[arm_name]
    ns = Namespace(
        method=method,
        temperature=extra.get("temperature", 1.0),
        top_p=extra.get("top_p", 1.0),
        top_k=extra.get("top_k", 0),
        repetition_penalty=1.0,
        enable_thinking=False,
        post_temperature=None,
        alpha=None,
        renyi_k=None,
        sampler_think=None, sampler_code=None,
        temp_think=None, temp_code=None,
    )
    return _method_key(ns)


def _load_jsonl_by_task(path: Path) -> dict[int, dict]:
    if not path.exists():
        return {}
    out = {}
    with path.open() as f:
        for line in f:
            d = json.loads(line)
            out[d["task_id"]] = d
    return out


def compare(out_root: Path) -> bool:
    """Returns True if any mismatch/problem found."""
    any_fail = False
    single_dir = out_root / "single_arm"
    multi_dir = out_root / "multi_arm"

    # Load cache-stats for every stratum up front: (task_id, arm_name) -> num_cached.
    cache_stats: dict[tuple[int, str], int | None] = {}
    for (source, difficulty) in STRATA:
        stats_path = multi_dir / f"cache_stats_{source}_{difficulty}.jsonl"
        if stats_path.exists():
            for line in stats_path.open():
                d = json.loads(line)
                cache_stats[(d["problem_id"], d["arm_name"])] = d["num_cached_tokens"]

    print("=== Main comparison: single-arm vs multi-arm, per (problem, arm) ===")
    cold_results, hot_results = [], []
    for (source, difficulty), task_ids in STRATA.items():
        model_dir = MODEL.replace("/", "--")
        stratum_sub = f"{source}_{difficulty}"
        for arm_name in ARMS:
            key = _method_key_str(arm_name)
            single_path = single_dir / model_dir / stratum_sub / f"{key}.jsonl"
            multi_path = multi_dir / model_dir / stratum_sub / f"{key}.jsonl"
            single_data = _load_jsonl_by_task(single_path)
            multi_data = _load_jsonl_by_task(multi_path)
            for task_id in task_ids:
                s = single_data.get(task_id)
                m = multi_data.get(task_id)
                if s is None or m is None:
                    print(f"  [MISSING] {stratum_sub}/{arm_name}/task={task_id}: "
                          f"single={'ok' if s else 'MISSING'} multi={'ok' if m else 'MISSING'}")
                    any_fail = True
                    continue
                match = s["samples"] == m["samples"]
                num_cached = cache_stats.get((task_id, arm_name))
                bucket = cold_results if not num_cached else hot_results
                bucket.append((stratum_sub, arm_name, task_id, match, num_cached))
                if not match:
                    any_fail = True
                    print(f"  [MISMATCH] {stratum_sub}/{arm_name}/task={task_id} "
                          f"(num_cached_tokens={num_cached})")

    print(f"\nCold (num_cached_tokens falsy) pairs: {len(cold_results)}, "
          f"all match: {all(r[3] for r in cold_results) if cold_results else 'N/A'}")
    print(f"Hot (real cache hit) pairs: {len(hot_results)}, "
          f"all match: {all(r[3] for r in hot_results) if hot_results else 'N/A'}")

    print("\n=== Baseline: cache-miss vs cache-miss (vLLM self-determinism) ===")
    b1 = _load_jsonl_by_task(
        out_root / "baseline_run1" / MODEL.replace("/", "--")
        / f"{BASELINE_SOURCE}_{BASELINE_DIFFICULTY}" / f"{_method_key_str(BASELINE_ARM)}.jsonl"
    )
    b2 = _load_jsonl_by_task(
        out_root / "baseline_run2" / MODEL.replace("/", "--")
        / f"{BASELINE_SOURCE}_{BASELINE_DIFFICULTY}" / f"{_method_key_str(BASELINE_ARM)}.jsonl"
    )
    s1, s2 = b1.get(BASELINE_TASK_ID), b2.get(BASELINE_TASK_ID)
    if s1 is None or s2 is None:
        print("  [MISSING] baseline pair incomplete")
        any_fail = True
    elif s1["samples"] != s2["samples"]:
        print("  [MISMATCH] baseline_run1 vs baseline_run2 differ -- vLLM is "
              "NOT self-deterministic for this model/seed/length on this "
              "hardware. Do not trust ANY part of the main comparison above "
              "until this is understood -- the whole premise (fixed seed = "
              "reproducible) is in question.")
        any_fail = True
    else:
        print("  [PASS] baseline_run1 == baseline_run2 (vLLM self-determinism confirmed)")

    print()
    if any_fail:
        print("RESULT: at least one problem found -- see above before drawing any "
              "conclusion about multi-arm vs single-arm equivalence.")
    else:
        print("RESULT: baseline self-determinism confirmed, and every single-arm "
              "vs multi-arm pair matched exactly (both cold and hot cache subsets).")
    return any_fail


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-root", type=Path, default=REPO / "results/_phase3_pilot")
    ap.add_argument("--dry-run", action="store_true", help="Print commands, run nothing.")
    ap.add_argument("--compare-only", action="store_true", help="Skip generation, just diff existing output.")
    args = ap.parse_args()

    if args.compare_only:
        sys.exit(1 if compare(args.out_root) else 0)

    cmds = build_all_commands(args.out_root)
    print(f"{len(cmds)} commands planned (6 multi-arm + 18 single-arm + 2 baseline = 26).\n")
    for i, (label, cmd) in enumerate(cmds, 1):
        print(f"[{i}/{len(cmds)}] {label}")
        print("  " + " ".join(cmd))
        if not args.dry_run:
            result = subprocess.run(cmd, cwd=REPO)
            if result.returncode != 0:
                print(f"\nFATAL: command failed (exit {result.returncode}): {label}")
                sys.exit(result.returncode)

    if args.dry_run:
        print("\n--dry-run: nothing executed.")
        return

    print("\nAll generation complete. Running comparison...\n")
    sys.exit(1 if compare(args.out_root) else 0)


if __name__ == "__main__":
    main()
