"""Multi-arm APPS runner — one vLLM process, N arms, shared prefix cache.

Loads vLLM ONCE and iterates ``(prompt, arm)`` pairs through the shared engine
so vLLM's prefix cache is reused across arms sharing the same prompt (every arm
in a sweep processes the same 252 / 748 prompts).

Why: ``run_cf748_apps_qwen3.sh`` and friends currently invoke
``python -m bench.apps ...`` once per arm; each spawn re-initializes vLLM
(model load, CUDA graphs, torch.compile) and cold-starts the prefix cache.
On a 16-arm sweep that is ~10-20% wasted compute (warmup + redundant prefill).
This module runs all arms in one process so the warmup is paid once and the
prefix cache is reused.

Output JSONL layout, per-arm filenames, record fields, and resume semantics
are byte-equivalent to ``bench.apps.runner`` (single-arm) — the paper
extractor and ``bench.eval`` read the outputs unchanged.

Only vLLM backend is supported. All per-arm generation knobs flow through
unchanged; no implicit defaults introduced (vLLM default sampling, unseeded,
same as single-arm path).

Usage::

    uv run python -m bench.apps.multi_arm_runner \\
        --model Qwen/Qwen3-8B --backend vllm \\
        --source CODEFORCES --difficulty interview \\
        --task-ids-file data/cf_interview_748_ids.txt \\
        --enable-thinking \\
        --n-samples 10 --max-new-tokens 32768 \\
        --results-dir results/_cf_interview_748 \\
        --arms pless pless_norm renyi_k0.1 rec topp0.9 temp

    # Custom (non-preset) arms via --arm-raw (repeatable):
    #     --arm-raw "name=custom1,method=temp,temperature=1.5,top_p=0.85"
"""

from __future__ import annotations

import argparse
import json
from argparse import Namespace
from datetime import datetime, timezone
from pathlib import Path

from tqdm import tqdm

from bench.apps.dataset import DIFFICULTIES, SOURCES, load_apps
from bench.apps.prompts import format_prompt_apps_instruct
from bench.apps.runner import _method_key, _output_path
from bench.checkpointing import append_result, load_completed_ids
from bench.generator import _strip_think_content
from bench.prompts import is_instruct_model


# ---------------------------------------------------------------------------
# Arm spec parsing. Preset names mirror ``run_cf748_apps_qwen3.sh:arm_args()``
# byte-for-byte so the multi-arm runner reproduces the single-arm shell flow.
# ---------------------------------------------------------------------------


def _preset_arm_spec(arm_name: str) -> dict:
    """Return a dict with generation params for a preset arm name.

    Mirrors ``run_cf748_apps_qwen3.sh:arm_args()``:

    * ``pless``        → ``--method pless --temperature 1.0``
    * ``pless_norm``   → ``--method pless_norm --temperature 1.0``
    * ``renyi_k<K>``   → ``--method pless_renyi --renyi-k <K> --temperature 1.0``
    * ``rec``          → ``--method temp --temperature 0.6 --top-p 0.95 --top-k 20``
    * ``topp<P>``      → ``--method temp --top-p <P> --top-k 0 --temperature 1.0``
    * ``temp``         → ``--method temp --temperature 1.0``

    ``top_p=1.0`` and ``top_k=0`` are the single-arm argparse defaults; set
    explicitly here so :func:`_method_key` builds the same filename stem.
    """
    base = {"top_p": 1.0, "top_k": 0, "renyi_k": None, "alpha": None}
    if arm_name == "pless":
        return {**base, "name": arm_name, "method": "pless", "temperature": 1.0}
    if arm_name == "pless_norm":
        return {**base, "name": arm_name, "method": "pless_norm", "temperature": 1.0}
    if arm_name.startswith("renyi_k"):
        try:
            k = float(arm_name[len("renyi_k"):])
        except ValueError as exc:
            raise ValueError(
                f"renyi_k arm must look like 'renyi_k0.1', got {arm_name!r}"
            ) from exc
        return {**base, "name": arm_name, "method": "pless_renyi",
                "temperature": 1.0, "renyi_k": k}
    if arm_name == "rec":
        return {**base, "name": arm_name, "method": "temp",
                "temperature": 0.6, "top_p": 0.95, "top_k": 20}
    if arm_name.startswith("topp"):
        try:
            p = float(arm_name[len("topp"):])
        except ValueError as exc:
            raise ValueError(
                f"topp arm must look like 'topp0.9', got {arm_name!r}"
            ) from exc
        return {**base, "name": arm_name, "method": "temp",
                "temperature": 1.0, "top_p": p, "top_k": 0}
    if arm_name == "temp":
        return {**base, "name": arm_name, "method": "temp", "temperature": 1.0}
    raise ValueError(
        f"unknown preset arm {arm_name!r}; use --arm-raw for custom specs"
    )


def _parse_arm_raw(spec: str) -> dict:
    """Parse ``name=X,method=Y,temperature=Z,top_p=A,top_k=B,renyi_k=C,alpha=D``.

    ``name`` and ``method`` are required. Other keys default to the single-arm
    runner's argparse defaults: ``temperature=1.0``, ``top_p=1.0``, ``top_k=0``,
    ``renyi_k=None``, ``alpha=None``.
    """
    parts: dict[str, str] = {}
    for kv in spec.split(","):
        kv = kv.strip()
        if not kv:
            continue
        if "=" not in kv:
            raise ValueError(
                f"malformed arm-raw segment {kv!r}; expected key=value"
            )
        key, val = kv.split("=", 1)
        parts[key.strip()] = val.strip()
    if "name" not in parts or "method" not in parts:
        raise ValueError(
            f"arm-raw requires at least name= and method=: {spec!r}"
        )
    return {
        "name": parts["name"],
        "method": parts["method"],
        "temperature": float(parts.get("temperature", 1.0)),
        "top_p": float(parts.get("top_p", 1.0)),
        "top_k": int(parts.get("top_k", 0)),
        "renyi_k": float(parts["renyi_k"]) if "renyi_k" in parts else None,
        "alpha": float(parts["alpha"]) if "alpha" in parts else None,
    }


def _arm_to_namespace(arm: dict, base_args: argparse.Namespace) -> Namespace:
    """Build a per-arm Namespace shaped like what ``bench.apps.runner`` expects.

    Only the fields :func:`_method_key` reads are populated. Loop-force,
    scaffolds, split-sampler, post-temperature, and repetition_penalty are
    left at their single-arm defaults — the multi-arm runner does not expose
    them.
    """
    return Namespace(
        method=arm["method"],
        temperature=arm["temperature"],
        top_p=arm["top_p"],
        top_k=arm["top_k"],
        renyi_k=arm["renyi_k"],
        alpha=arm["alpha"],
        repetition_penalty=1.0,
        post_temperature=None,
        sampler_think=None,
        sampler_code=None,
        temp_think=None,
        temp_code=None,
        enable_thinking=base_args.enable_thinking,
    )


def _arm_output_path(arm: dict, base_args: argparse.Namespace) -> Path:
    """Resolve the output JSONL path for a given arm (same layout as single-arm)."""
    ns = _arm_to_namespace(arm, base_args)
    return _output_path(
        base_args.results_dir, base_args.model, base_args.source,
        base_args.difficulty, _method_key(ns), arm["temperature"],
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Base flags mirror bench.apps.runner — same semantics, same defaults.
    p.add_argument("--model", required=True, help="HuggingFace model ID")
    p.add_argument("--source", required=True, choices=list(SOURCES))
    p.add_argument("--difficulty", required=True, choices=list(DIFFICULTIES))
    p.add_argument("--n-samples", type=int, default=10)
    p.add_argument("--max-new-tokens", type=int, default=8192,
                   help="Default matches bench.apps.runner (8192). Shell scripts "
                        "pass 32768 via --max-new-tokens.")
    p.add_argument("--results-dir", required=True)
    p.add_argument(
        "--backend", choices=["vllm"], default="vllm",
        help="Only vllm is supported — multi-arm is a vLLM-prefix-cache "
             "optimization and depends on a single long-lived engine.",
    )
    p.add_argument("--max-problems", type=int, default=None)
    p.add_argument("--task-ids", type=int, nargs="+", default=None)
    p.add_argument("--task-ids-file", type=Path, default=None)
    p.add_argument("--enable-thinking", action="store_true")
    p.add_argument("--treat-as-instruct", action="store_true",
                   help="Force instruct-prompt formatting even when the model "
                        "id does not contain 'Instruct' or 'Chat'. Mirrors "
                        "bench.apps.runner's flag of the same name. Use for "
                        "models like Ministral-3-3B-Reasoning that are "
                        "chat-tuned but not named accordingly.")
    p.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    p.add_argument("--max-model-len", type=int, default=None,
                   help="Caps the engine's max sequence length (prompt + "
                        "generation). Mirrors bench.apps.runner's flag of the "
                        "same name. Omitted (default) lets vLLM use the "
                        "model's native max, which can exceed available KV "
                        "cache memory for long-context models.")
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--seed", type=int, default=None,
                   help="Applied to every arm's SamplingParams. Omitted "
                        "(default) preserves existing unseeded behavior.")
    p.add_argument("--cache-stats-file", type=Path, default=None,
                   help="Optional. When set, writes one JSON line per "
                        "(problem_id, arm_name, num_cached_tokens) to this "
                        "path -- real vLLM RequestOutput.num_cached_tokens "
                        "via a spy on engine.generate, NOT inferred from "
                        "loop position. Diagnostic only; omitted (default) "
                        "is a complete no-op, zero behavior change.")

    # Arm specification.
    p.add_argument(
        "--arms", nargs="+", default=[], metavar="ARM",
        help="Preset arm names (space-separated). See _preset_arm_spec for the "
             "preset → generation-params mapping.",
    )
    p.add_argument(
        "--arm-raw", action="append", default=[], metavar="SPEC",
        help="Per-arm raw spec 'name=X,method=Y,temperature=Z,top_p=A,top_k=B,"
             "renyi_k=C,alpha=D'. Repeatable.",
    )

    return p


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = _build_argparser()
    args = parser.parse_args()

    if not is_instruct_model(args.model) and not args.treat_as_instruct:
        raise SystemExit(
            f"Multi-arm runner expects an instruct model; got {args.model!r}. "
            "Pass --treat-as-instruct if the model is chat-tuned but not "
            "named accordingly."
        )

    arms: list[dict] = [_preset_arm_spec(name) for name in args.arms]
    arms.extend(_parse_arm_raw(spec) for spec in args.arm_raw)
    if not arms:
        raise SystemExit("At least one arm required via --arms or --arm-raw.")

    # Resolve per-arm output paths + load completed-task-id sets for resume.
    arm_out_paths: dict[str, Path] = {}
    arm_completed: dict[str, set[int]] = {}
    for arm in arms:
        p = _arm_output_path(arm, args)
        if args.no_resume and p.exists():
            p.unlink()
        arm_out_paths[arm["name"]] = p
        arm_completed[arm["name"]] = load_completed_ids(p)
        print(f"[arm {arm['name']:<22}] out={p}  "
              f"completed={len(arm_completed[arm['name']])}")

    # Load vLLM once. Explicitly request prefix caching — the WHOLE POINT of
    # multi-arm is cross-arm prefix-cache reuse; silently falling back to
    # no-caching (if vLLM's default ever changes or is overridden) would
    # eliminate the ~10-20% speedup without any error.
    print(f"\nLoading vLLM engine for {args.model} (enable_prefix_caching=True) ...")
    from bench.generator_vllm import (
        encode_prompt_for_vllm,
        generate_samples_standard_vllm,
        generate_samples_vllm,
        load_engine,
    )
    engine = load_engine(args.model, dtype=args.dtype, enable_prefix_caching=True,
                          max_model_len=args.max_model_len)
    tokenizer = getattr(engine, "_safe_tokenizer", None) or engine.get_tokenizer()

    # Optional diagnostic: spy on engine.generate to capture each call's real
    # RequestOutput.num_cached_tokens (same pattern as
    # scripts/test_prefix_caching_determinism.py). Only active when
    # --cache-stats-file is passed; otherwise engine.generate is untouched
    # and captured_outputs/cache_stats_fh stay unused (zero behavior change).
    captured_outputs: list = []
    cache_stats_fh = None
    if args.cache_stats_file is not None:
        _real_generate = engine.generate

        def _spying_generate(*a, **kw):
            outs = _real_generate(*a, **kw)
            captured_outputs.append(outs)
            return outs

        engine.generate = _spying_generate
        cache_stats_fh = args.cache_stats_file.open("a")

    # Runtime assertion: verify prefix caching is actually on. In vLLM 0.21
    # this lives on cache_config (NOT model_config — that was removed/never
    # existed as a stable field). If the attribute path renames in a future
    # vLLM, this probe will warn and continue rather than falsely assert ON.
    try:
        prefix_cache_on = engine.llm_engine.cache_config.enable_prefix_caching
    except Exception:
        prefix_cache_on = None
    if prefix_cache_on is True:
        print(f"[multi-arm] prefix caching: ON (cache_config.enable_prefix_caching={prefix_cache_on!r})")
    elif prefix_cache_on is False:
        raise SystemExit(
            "FATAL: vLLM engine reports enable_prefix_caching=False. Multi-arm "
            "requires prefix caching to be enabled; otherwise there is no speedup "
            "vs the single-arm per-process path. Aborting."
        )
    else:
        print(
            "[multi-arm] WARNING: could not probe cache_config.enable_prefix_caching "
            "— cannot verify prefix caching is on. Proceeding on the assumption "
            "that `load_engine(..., enable_prefix_caching=True)` took effect. "
            "If this warning fires, open bench/apps/multi_arm_runner.py and update "
            "the probe path for your vLLM version."
        )

    # Load APPS problems, filtered to the requested bucket + ids.
    problems = list(load_apps(source=args.source, difficulty=args.difficulty))
    wanted: set[int] | None = None
    if args.task_ids is not None:
        wanted = set(args.task_ids)
    if args.task_ids_file is not None:
        file_ids = {int(tok) for tok in args.task_ids_file.read_text().split()}
        wanted = file_ids if wanted is None else (wanted | file_ids)
    if wanted is not None:
        problems = [p for p in problems if p.problem_id in wanted]
    if args.max_problems is not None:
        problems = problems[: args.max_problems]

    print(f"\nProcessing {len(problems)} problems × {len(arms)} arms "
          f"= {len(problems) * len(arms)} (prompt, arm) pairs")
    print("Prompt-outer, arm-inner loop (maximizes prefix-cache reuse).\n")

    total = len(problems) * len(arms)
    pbar = tqdm(total=total, desc="prompts × arms")

    # Accumulate per-arm failure counts and failed (arm, task_id) pairs so a
    # silent systemic failure (e.g. one arm OOMs on every task) is visible at
    # run end rather than hidden behind tqdm.write lines.
    failed_pairs: list[tuple[str, int, str]] = []

    for problem in problems:
        prompt_text, code_prefix = format_prompt_apps_instruct(
            problem, tokenizer, enable_thinking=args.enable_thinking,
        )
        prompt_text_encoded = encode_prompt_for_vllm(
            prompt_text, getattr(engine, "_safe_tokenizer", None)
        )

        for arm in arms:
            arm_name = arm["name"]
            # Skip if this (prompt, arm) was already completed in a prior run.
            if problem.problem_id in arm_completed[arm_name]:
                pbar.update(1)
                continue

            try:
                _cache_stats_before = len(captured_outputs)
                if arm["method"] == "temp":
                    raw_samples = generate_samples_standard_vllm(
                        engine=engine, tokenizer=tokenizer,
                        prompt_text=prompt_text_encoded,
                        n_samples=args.n_samples,
                        max_new_tokens=args.max_new_tokens,
                        temperature=arm["temperature"], stop_strings=None,
                        top_p=arm["top_p"], top_k=arm["top_k"],
                        repetition_penalty=1.0,
                        seed=args.seed,
                    )
                else:
                    raw_samples = generate_samples_vllm(
                        engine=engine, tokenizer=tokenizer,
                        prompt_text=prompt_text_encoded,
                        sampler_name=arm["method"],
                        n_samples=args.n_samples,
                        max_new_tokens=args.max_new_tokens,
                        temperature=arm["temperature"], stop_strings=None,
                        alpha=arm["alpha"], renyi_k=arm["renyi_k"],
                        loop_ngram_n=None, loop_ngram_k=None, loop_window=None,
                        seed=args.seed,
                    )

                if cache_stats_fh is not None:
                    new_outputs = captured_outputs[_cache_stats_before:]
                    assert len(new_outputs) == 1, (
                        f"expected exactly one engine.generate() call per "
                        f"(problem, arm), spy captured {len(new_outputs)}"
                    )
                    num_cached = getattr(new_outputs[0][0], "num_cached_tokens", None)
                    cache_stats_fh.write(json.dumps({
                        "problem_id": problem.problem_id,
                        "arm_name": arm_name,
                        "num_cached_tokens": num_cached,
                    }) + "\n")
                    cache_stats_fh.flush()

                has_cot = args.enable_thinking
                samples_with_think = [code_prefix + s for s in raw_samples]
                if has_cot:
                    samples = [_strip_think_content(s) for s in samples_with_think]
                else:
                    samples = samples_with_think

                record = {
                    "model": args.model,
                    "backend": "vllm",
                    "method": arm["method"],
                    "temperature": arm["temperature"],
                    "top_p": arm["top_p"],
                    "top_k": arm["top_k"],
                    "task_id": problem.problem_id,
                    "source": problem.source,
                    "difficulty": problem.difficulty,
                    "prompt_text": problem.question,
                    "samples": samples,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                if arm["alpha"] is not None:
                    record["alpha"] = arm["alpha"]
                if arm["renyi_k"] is not None:
                    record["renyi_k"] = arm["renyi_k"]
                if has_cot:
                    record["samples_with_thinking"] = samples_with_think

                # Per-prompt-per-arm flush: write before advancing to the next
                # arm so a mid-sweep crash leaves survivor arms intact for resume.
                append_result(arm_out_paths[arm_name], record)
                arm_completed[arm_name].add(problem.problem_id)

            except Exception as exc:
                tqdm.write(f"Error on arm={arm_name} task={problem.problem_id}: {exc}")
                failed_pairs.append((arm_name, problem.problem_id, repr(exc)))
            finally:
                pbar.update(1)

    pbar.close()
    if cache_stats_fh is not None:
        cache_stats_fh.close()
        engine.generate = _real_generate  # restore, tidiness only (process exits next)
    print("\nDone. Per-arm JSONLs:")
    for arm in arms:
        print(f"  {arm['name']:<22} → {arm_out_paths[arm['name']]}")

    # End-of-run failure summary. If any (arm, task_id) failed, print the full
    # list grouped by arm so a systemic single-arm failure (e.g. OOM on every
    # task of one config) is obvious even in a mostly-successful sweep.
    #
    # Exit-code policy (per advisor): one-off per-task failures (OOM, timeout
    # on one of N thousand pairs) are normal — exit 0 so the shell driver
    # doesn't abort downstream eval + report. Non-zero exit ONLY when:
    #   (a) at least 1% of (prompt, arm) pairs failed, OR
    #   (b) any single arm failed on 100% of its prompts (systemic to that arm).
    # The JSONL is the source of truth; partial sweeps are resumable.
    if failed_pairs:
        print(f"\n[multi-arm] {len(failed_pairs)} / {total} (arm, task_id) pair(s) FAILED:")
        per_arm: dict[str, list[tuple[int, str]]] = {}
        for arm_name, task_id, exc_repr in failed_pairs:
            per_arm.setdefault(arm_name, []).append((task_id, exc_repr))
        for arm_name, items in sorted(per_arm.items()):
            print(f"  arm={arm_name}: {len(items)} / {len(problems)} failures")
            for task_id, exc_repr in items[:5]:  # first 5 per arm
                print(f"    task_id={task_id}: {exc_repr}")
            if len(items) > 5:
                print(f"    ... (+{len(items) - 5} more)")
        failure_rate = len(failed_pairs) / total if total else 0.0
        systemic_arm = next(
            (name for name, items in per_arm.items() if len(items) == len(problems)),
            None,
        )
        if failure_rate >= 0.01 or systemic_arm is not None:
            if systemic_arm is not None:
                print(f"\n[multi-arm] FATAL: arm {systemic_arm!r} failed on 100% of prompts "
                      f"(systemic failure). Exiting with code 2.")
            else:
                print(f"\n[multi-arm] FATAL: {failure_rate*100:.2f}% failure rate "
                      f"≥ 1% threshold. Exiting with code 2.")
            raise SystemExit(2)
        print(f"\n[multi-arm] Failure rate {failure_rate*100:.2f}% below 1% threshold; "
              f"treating as successful sweep. JSONLs are the source of truth; "
              f"re-run to pick up unfinished pairs via resume.")


if __name__ == "__main__":
    main()
