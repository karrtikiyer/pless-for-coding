"""CUDA/vLLM runner: pless variants vs baselines on LiveCodeBench v6, Qwen3-8B.

Companion to ``scripts/run_pless_variants_qwen3_8b_cuda.py`` (the APPS version)
— same generation machinery, different dataset. Reuses:
  * ``bench.livecodebench.dataset.load_lcb`` for problems
  * ``bench.livecodebench.prompts.format_prompt_lcb_instruct`` for prompts
  * ``bench.generator_vllm.{load_engine,generate_samples_vllm,encode_prompt_for_vllm}``
  * ``bench.pless_variants_vllm.register()`` + ``bench.topnsigma_vllm.register()``
    for the 5 pless variants + 3 top-nsigma points (splices into
    ``bench.generator_vllm._SAMPLER_LOGIT_FN`` at import time)
  * the canonical ``"pless"`` sampler and ``"pless_renyi"`` (G_k) already
    registered in ``bench.generator_vllm`` itself — no new sampler code needed.

Does NOT modify or import ``bench/livecodebench/runner.py`` (that module is the
HF-oriented CLI used by other experiments; leaving it untouched avoids any
regression risk to work that depends on it).

**LCB eval constraint** (see ``bench/eval/__main__.py``'s livecodebench branch):
a results JSONL must be single-platform — ``records[0]["source"]`` picks which
test map loads. Since our stratified task set spans AtCoder + LeetCode, this
runner writes ONE FILE PER (config, platform) pair, not one file per config.

10 configs (name -> sampler_name/temperature/renyi_k), matching the user's
request verbatim:
  top_nsigma (n=1.0), top_nsigma_1.5, G_k=0.1, G_k=0.35, Qwen recipe
  (temp_standard @ T=0.6), pless_default (canonical, "pless" sampler),
  pless_effk, pless_topmass, pless_gini, pless_top1half.

Example::

    uv run --project pyproject-vllm.toml python scripts/run_pless_variants_qwen3_8b_lcb_cuda.py \\
        --model Qwen/Qwen3-8B --dtype bfloat16 \\
        --n-samples 10 --max-new-tokens 25000 --seed 42 \\
        --configs pless_gini top_nsigma
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path

# Order matters: register() mutates the vLLM dispatch table before we import
# generate_samples_vllm's consumers below.
from bench.pless_variants_vllm import register as _register_variants
_register_variants()
from bench.topnsigma_vllm import register as _register_topnsigma
_register_topnsigma()

from bench.generator import _strip_think_content
from bench.generator_vllm import (
    encode_prompt_for_vllm,
    generate_samples_vllm,
    load_engine,
)
from bench.livecodebench.dataset import load_lcb
from bench.livecodebench.prompts import format_prompt_lcb_instruct

REPO = Path(__file__).resolve().parent.parent
TASK_SELECTION_CSV = REPO / "results/lcb_v6_stratified60/task_selection.csv"
OUT_ROOT_TEMPLATE = "results/pless_variants_cuda/{model_slug}/LCB_v6_stratified60"

# (name, sampler_name, temperature, renyi_k_or_None)
CONFIGS: list[tuple[str, str, float, float | None]] = [
    ("top_nsigma",       "top_nsigma",       1.0, None),
    ("top_nsigma_1_5",   "top_nsigma_1_5",   1.0, None),
    ("gk_0_1",           "pless_renyi",      1.0, 0.1),
    ("gk_0_35",          "pless_renyi",      1.0, 0.35),
    ("qwen_recipe",      "temp_standard",    0.6, None),
    ("pless_default",    "pless",            1.0, None),
    ("pless_effk",       "pless_effk",       1.0, None),
    ("pless_topmass",    "pless_topmass",    1.0, None),
    ("pless_gini",       "pless_gini",       1.0, None),
    ("pless_top1half",   "pless_top1half",   1.0, None),
    # T1-only sweep: canonical pless with the pre-truncation temperature raised
    # above 1.0 to flatten the (often very peaked, RL-sharpened) distribution
    # before the collision-entropy threshold is applied. No T2 (post-pless
    # re-temperature, see sampler_bridge.make_pless_post_temp_sampler) here —
    # user asked for T1-only. Same "pless" sampler_name as pless_default;
    # temperature is what varies.
    ("pless_default_t1_5", "pless",          1.5, None),
    ("pless_default_t2_0", "pless",          2.0, None),
    ("pless_default_t2_5", "pless",          2.5, None),
    ("pless_default_t3_0", "pless",          3.0, None),
]
CONFIG_NAMES = [c[0] for c in CONFIGS]


def _load_task_selection() -> dict[str, list[dict]]:
    """Return {platform: [ {task_id, difficulty, contest_date}, ... ]}."""
    by_platform: dict[str, list[dict]] = {}
    with TASK_SELECTION_CSV.open() as f:
        for row in csv.DictReader(f):
            by_platform.setdefault(row["platform"], []).append(row)
    return by_platform


def _output_path(out_root: Path, config_name: str, platform: str,
                  temperature: float) -> Path:
    return out_root / f"{config_name}_{platform}_think_t{temperature}.jsonl"


def _run_config_on_platform(
    *, config_name: str, sampler_name: str, temperature: float,
    renyi_k: float | None, platform: str, problems: list, engine, tokenizer,
    n_samples: int, max_new_tokens: int, out_root: Path, model_id: str,
    seed: int | None, per_config_deadline: float, global_deadline: float,
) -> Path:
    out_root.mkdir(parents=True, exist_ok=True)
    out_path = _output_path(out_root, config_name, platform, temperature)

    done_ids: set[str] = set()
    if out_path.exists():
        for line in out_path.open():
            try:
                done_ids.add(str(json.loads(line)["task_id"]))
            except Exception:
                pass
        if done_ids:
            print(f"[{config_name}/{platform}] resume: {len(done_ids)} done")
    print(f"[{config_name}/{platform}] writing -> {out_path}")

    for idx, problem in enumerate(problems, 1):
        if problem.task_id in done_ids:
            continue
        now = time.time()
        if now > per_config_deadline or now > global_deadline:
            print(f"[{config_name}/{platform}] budget exhausted after "
                  f"{idx-1}/{len(problems)} tasks")
            break
        t0 = time.time()
        prompt_text, code_prefix = format_prompt_lcb_instruct(
            problem, tokenizer, enable_thinking=True,
        )
        prompt_text = encode_prompt_for_vllm(
            prompt_text, getattr(engine, "_safe_tokenizer", None)
        )
        kwargs = {}
        if renyi_k is not None:
            kwargs["renyi_k"] = renyi_k
        try:
            raw_samples = generate_samples_vllm(
                engine=engine, tokenizer=tokenizer, prompt_text=prompt_text,
                sampler_name=sampler_name, n_samples=n_samples,
                max_new_tokens=max_new_tokens, temperature=temperature,
                stop_strings=None, seed=seed, **kwargs,
            )
        except Exception as exc:
            print(f"[{config_name}/{platform}] task {problem.task_id} crashed: {exc!r}")
            continue

        elapsed = time.time() - t0
        samples_with_think = [code_prefix + s for s in raw_samples]
        samples = [_strip_think_content(s) for s in samples_with_think]

        record = {
            "model": model_id, "backend": "vllm", "method": config_name,
            "sampler_name": sampler_name, "renyi_k": renyi_k,
            "temperature": temperature, "top_p": 1.0, "top_k": 0,
            "task_id": problem.task_id, "source": problem.platform,
            "difficulty": problem.difficulty, "contest_date": problem.contest_date,
            "prompt_text": problem.question,
            "samples": samples, "samples_with_thinking": samples_with_think,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "enable_thinking": True,
        }
        with out_path.open("a") as fh:
            fh.write(json.dumps(record) + "\n")
        print(f"[{config_name}/{platform}] task {problem.task_id}: {elapsed:.0f}s")

    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--quantization", default=None,
                    help="vLLM quantization flag: 'awq', 'gptq', 'fp8' etc. "
                         "Required if model+KV won't fit unquantized (e.g. "
                         "16 GB card + Qwen3-8B needs fp8).")
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.92)
    ap.add_argument("--max-model-len", type=int, default=27000)
    ap.add_argument("--n-samples", type=int, default=10)
    ap.add_argument("--max-new-tokens", type=int, default=25000)
    ap.add_argument("--configs", nargs="+", default=CONFIG_NAMES, choices=CONFIG_NAMES)
    ap.add_argument("--platforms", nargs="+", default=["atcoder", "leetcode"],
                    choices=["atcoder", "leetcode"])
    ap.add_argument("--only-task-ids", nargs="+", default=None)
    ap.add_argument("--budget-minutes", type=float, default=3000.0)
    ap.add_argument("--per-config-minutes", type=float, default=400.0)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    global_start = time.time()
    global_deadline = global_start + args.budget_minutes * 60.0

    selection = _load_task_selection()
    model_slug = args.model.replace("/", "--")
    out_root = REPO / OUT_ROOT_TEMPLATE.format(model_slug=model_slug)

    load_kwargs = {"dtype": args.dtype,
                   "gpu_memory_utilization": args.gpu_memory_utilization,
                   "max_model_len": args.max_model_len}
    if args.quantization:
        load_kwargs["quantization"] = args.quantization
    print(f"Loading engine: {args.model} (kwargs={load_kwargs})")
    t0 = time.time()
    engine = load_engine(args.model, **load_kwargs)
    print(f"Engine ready in {time.time()-t0:.0f}s")
    tokenizer = getattr(engine, "_safe_tokenizer", None) or engine.get_tokenizer()

    config_by_name = {c[0]: c for c in CONFIGS}
    written = []
    for platform in args.platforms:
        rows = selection.get(platform, [])
        wanted_ids = {r["task_id"] for r in rows}
        if args.only_task_ids is not None:
            wanted_ids &= set(args.only_task_ids)
        problems = list(load_lcb(
            platforms=(platform,), task_ids=wanted_ids, with_tests=False,
        ))
        problems.sort(key=lambda p: p.task_id)
        print(f"Platform {platform}: matched {len(problems)}/{len(wanted_ids)} problems")

        for name in args.configs:
            _, sampler_name, temperature, renyi_k = config_by_name[name]
            per_config_deadline = min(
                time.time() + args.per_config_minutes * 60.0, global_deadline
            )
            p = _run_config_on_platform(
                config_name=name, sampler_name=sampler_name,
                temperature=temperature, renyi_k=renyi_k, platform=platform,
                problems=problems, engine=engine, tokenizer=tokenizer,
                n_samples=args.n_samples, max_new_tokens=args.max_new_tokens,
                out_root=out_root, model_id=args.model, seed=args.seed,
                per_config_deadline=per_config_deadline,
                global_deadline=global_deadline,
            )
            written.append(p)

    total = time.time() - global_start
    print(f"\nDone in {total/60:.1f} min. Wrote:")
    for p in written:
        n = sum(1 for _ in p.open()) if p.exists() else 0
        print(f"  {p}   ({n} tasks)")


if __name__ == "__main__":
    main()
