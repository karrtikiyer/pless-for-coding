"""Signal diagnostic: does Σpᵢ² rise before the n-gram loop onset?

Phase-0 experiment for the adaptive loop-escape project.

For each sampled thinking-chain from a given JSONL:
  1. Classify as clean / looping_completed / looping_truncated using the
     validated per-model n-gram detector params.
  2. Tokenize the <think> block.
  3. Simulate the streaming n-gram detector to find onset_token — the exact
     token index where the live detector would have fired.
  4. Teacher-force the model to get per-token Σpᵢ² (collision probability)
     and Shannon H at each think-block position.
  5. Fit a per-sample baseline (mean + std of Σpᵢ² over the first BASELINE_FRAC
     of the trace) and find signal_rise_token: the first position where a
     rolling-mean crosses baseline + RISE_SIGMA * std and stays elevated
     for SUSTAIN_TOKENS consecutive steps.
  6. lead_time = onset_token - signal_rise_token

Outputs (all under --output-dir):
  per_sample.jsonl          — raw per-sample results (resumable)
  lead_time_stats.json      — aggregate statistics
  mean_trajectory.png       — mean Σpᵢ² aligned to onset (looping vs clean)
  individual_traces.png     — 20 individual looping traces with onset marked

Usage
-----
# classify only (no GPU — checks n-gram labels, prints class counts):
  HF_HUB_OFFLINE=1 uv run python scripts/signal_diagnostic.py \\
      --jsonl results/pless_cot_efficiency_vllm/Qwen--Qwen3-8B/ATCODER_interview_all_252/pless_think_t1.0_t1.0.jsonl \\
      --model Qwen/Qwen3-8B \\
      --classify-only

# full run (GPU required):
  HF_HUB_OFFLINE=1 uv run python scripts/signal_diagnostic.py \\
      --jsonl results/pless_cot_efficiency_vllm/Qwen--Qwen3-8B/ATCODER_interview_all_252/pless_think_t1.0_t1.0.jsonl \\
      --model Qwen/Qwen3-8B

Validated n-gram params (from run_loop_forcethink_apps_qwen3.sh):
  Qwen3-8B:                n=30 k=6 window=1200
  DeepSeek-R1-Distill-*:  n=30 k=6 window=3000
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Per-model validated detector params (DO NOT change without re-running
# detector_nk_grid.py / detector_falsepos_check.py)
# ---------------------------------------------------------------------------
LOOP_PARAMS: dict[str, dict] = {
    "Qwen/Qwen3-8B":                              dict(n=30, k=6, window=1200),
    "Qwen/Qwen3-8B-Instruct":                     dict(n=30, k=6, window=1200),
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B":  dict(n=30, k=6, window=3000),
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B":  dict(n=30, k=6, window=3000),
}

# Signal-rise detection hyperparameters
BASELINE_FRAC   = 0.10   # fraction of trace used to estimate per-sample baseline
ROLLING_WINDOW  = 50     # tokens in the rolling-mean window
RISE_SIGMA      = 2.0    # std-deviations above baseline to call a "rise"
SUSTAIN_TOKENS  = 20     # consecutive steps above threshold before declaring rise

MAX_SAMPLES_PER_CLASS = 100
MAX_CTX_TOKENS        = 38000   # leave headroom below model's 40960 limit


# ---------------------------------------------------------------------------
# N-gram onset simulation (mirrors bench/loop_detect.py exactly)
# ---------------------------------------------------------------------------

_CHECK_EVERY = 8   # matches bench/generator_vllm.py _LOOP_CHECK_EVERY

def simulate_onset(token_ids: list[int], n: int, k: int, window: int) -> int | None:
    """Return the token index at which the streaming n-gram detector first fires,
    or None if it never fires on this sequence.

    Checks every _CHECK_EVERY tokens (matches the live vLLM throttle) — a loop
    needs at least n*k tokens to form so skipping steps is safe and gives ~8x speedup.
    """
    buf: list[int] = []
    for i, tid in enumerate(token_ids):
        buf.append(tid)
        if len(buf) < n * k:
            continue
        if i % _CHECK_EVERY != 0:
            continue
        t = buf[-window:]
        counts = Counter(tuple(t[j:j + n]) for j in range(len(t) - n + 1))
        if max(counts.values()) >= k:
            return i
    return None


# ---------------------------------------------------------------------------
# Sample classification
# ---------------------------------------------------------------------------

def classify_samples(
    jsonl_path: Path,
    tokenizer,
    loop_params: dict,
) -> dict[str, list[dict]]:
    """Return {'clean': [...], 'looping_completed': [...], 'looping_truncated': [...]}

    Each entry is a dict with: task_id, sample_idx, prompt_text,
    think_text, think_tokens (list[int]), onset_token (int|None).
    """
    n, k, window = loop_params["n"], loop_params["k"], loop_params["window"]
    buckets: dict[str, list[dict]] = {
        "clean": [], "looping_completed": [], "looping_truncated": []
    }

    with open(jsonl_path) as f:
        for line in f:
            record = json.loads(line)
            task_id = record["task_id"]
            prompt_text = record.get("prompt_text", "")

            for s_idx, raw in enumerate(record.get("samples_with_thinking", [])):
                think_start = raw.find("<think>")
                think_end   = raw.find("</think>")

                if think_start < 0:
                    continue  # no think block at all — skip

                is_complete = think_end > think_start

                if is_complete:
                    think_text = raw[think_start: think_end]
                else:
                    think_text = raw[think_start:]  # truncated at token limit

                think_tokens = tokenizer.encode(think_text, add_special_tokens=False)
                onset = simulate_onset(think_tokens, n, k, window)

                entry = dict(
                    task_id=task_id,
                    sample_idx=s_idx,
                    prompt_text=prompt_text,
                    think_text=think_text,
                    think_tokens=think_tokens,
                    onset_token=onset,
                )

                if not is_complete:
                    buckets["looping_truncated"].append(entry)
                elif onset is not None:
                    buckets["looping_completed"].append(entry)
                else:
                    buckets["clean"].append(entry)

    return buckets


# ---------------------------------------------------------------------------
# Teacher-forced signals (per-token Σpᵢ² and Shannon H)
# ---------------------------------------------------------------------------

def teacher_forced_signals(
    model,
    input_ids,       # (1, seq_len) torch tensor on model.device
    prompt_len: int,
) -> tuple[list[float], list[float]]:
    """Single forward pass → (shannon_H list, collision_prob Σpᵢ² list).

    Returns one value per think-block token (positions prompt_len … seq_len-1).
    """
    import torch
    with torch.no_grad():
        logits = model(input_ids).logits[0]  # (seq_len, vocab)

    n_think = input_ids.shape[1] - prompt_len
    if n_think <= 0:
        return [], []

    # logits[t] predicts token at position t+1
    # think-block token i is at input position prompt_len + i
    # so its predictive logit is at logits[prompt_len - 1 + i]
    think_logits = logits[prompt_len - 1: prompt_len - 1 + n_think].float()

    probs = torch.softmax(think_logits, dim=-1)           # (n_think, vocab)
    log_probs = torch.log(probs + 1e-12)
    H    = -(probs * log_probs).sum(dim=-1).cpu().tolist()  # Shannon entropy
    coll = (probs ** 2).sum(dim=-1).cpu().tolist()          # Σpᵢ²

    return H, coll


# ---------------------------------------------------------------------------
# Signal-rise detection
# ---------------------------------------------------------------------------

def find_signal_rise(
    coll: list[float],
    onset: int,
    baseline_frac: float = BASELINE_FRAC,
    rolling_win: int = ROLLING_WINDOW,
    rise_sigma: float = RISE_SIGMA,
    sustain: int = SUSTAIN_TOKENS,
) -> int | None:
    """Return first token index where rolling-mean Σpᵢ² sustains above
    baseline + rise_sigma*std for `sustain` consecutive tokens, or None.

    Baseline is estimated from the first baseline_frac of the trace, capped
    at the onset position to avoid contamination from the loop region.
    """
    baseline_end = min(int(len(coll) * baseline_frac), onset - 1) if onset else int(len(coll) * baseline_frac)
    if baseline_end < 10:
        return None

    base_vals = coll[:baseline_end]
    mu  = float(np.mean(base_vals))
    sig = float(np.std(base_vals))
    if sig < 1e-8:
        return None

    threshold = mu + rise_sigma * sig
    arr = np.array(coll, dtype=float)

    # rolling mean (causal: window of last `rolling_win` tokens)
    rm = np.convolve(arr, np.ones(rolling_win) / rolling_win, mode="full")[:len(arr)]

    # find first run of `sustain` consecutive steps above threshold
    above = (rm > threshold).astype(int)
    consec = 0
    for i, a in enumerate(above):
        if a:
            consec += 1
            if consec >= sustain:
                return i - sustain + 1   # first token of the sustained run
        else:
            consec = 0
    return None


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _rolling_mean(arr: np.ndarray, w: int) -> np.ndarray:
    return np.convolve(arr, np.ones(w) / w, mode="same")


def plot_mean_trajectory(
    looping_results: list[dict],
    clean_results:   list[dict],
    output_path: Path,
    rolling_w: int = ROLLING_WINDOW,
) -> None:
    """Mean Σpᵢ² aligned to onset position (x=0).

    For looping samples: align so onset is at x=0; show [-500, +500] tokens.
    For clean samples: align to the median onset position from looping set.
    """
    half = 500
    lo_traces, cl_traces = [], []

    for r in looping_results:
        coll = np.array(r["coll"])
        onset = r["onset_token"]
        if onset is None or onset < 10:
            continue
        rm = _rolling_mean(coll, rolling_w)
        start = max(0, onset - half)
        end   = min(len(rm), onset + half)
        pad_left  = half - (onset - start)
        pad_right = half - (end - onset)
        trace = np.pad(rm[start:end], (pad_left, pad_right), constant_values=np.nan)
        lo_traces.append(trace)

    for r in clean_results:
        coll = np.array(r["coll"])
        rm = _rolling_mean(coll, rolling_w)
        mid = len(rm) // 2
        start = max(0, mid - half)
        end   = min(len(rm), mid + half)
        pad_left  = half - (mid - start)
        pad_right = half - (end - mid)
        trace = np.pad(rm[start:end], (pad_left, pad_right), constant_values=np.nan)
        cl_traces.append(trace)

    xs = np.arange(-half, half)
    fig, ax = plt.subplots(figsize=(12, 5))

    def _plot_band(traces, color, label):
        if not traces:
            return
        mat = np.array(traces)
        mean = np.nanmean(mat, axis=0)
        std  = np.nanstd(mat, axis=0)
        ax.plot(xs, mean, color=color, linewidth=1.8, label=f"{label} (n={len(traces)})")
        ax.fill_between(xs, mean - std, mean + std, color=color, alpha=0.15)

    _plot_band(lo_traces, "#e74c3c", "looping_completed")
    _plot_band(cl_traces, "#2ecc71", "clean")

    ax.axvline(0, color="black", linestyle="--", linewidth=1.2, label="n-gram onset (x=0)")
    ax.set_xlabel("Token offset from n-gram onset", fontsize=12)
    ax.set_ylabel("Σpᵢ² (rolling mean, w=50)", fontsize=12)
    ax.set_title("Mean collision probability aligned to loop onset\n"
                 "(does Σpᵢ² rise BEFORE x=0?)", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {output_path}")


def plot_individual_traces(
    looping_results: list[dict],
    output_path: Path,
    n_traces: int = 20,
    rolling_w: int = ROLLING_WINDOW,
) -> None:
    """Grid of individual looping traces with onset and signal-rise marked."""
    candidates = [r for r in looping_results if r["onset_token"] is not None]
    random.seed(42)
    subset = random.sample(candidates, min(n_traces, len(candidates)))

    cols = 4
    rows = math.ceil(len(subset) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 3))
    axes = np.array(axes).flatten()

    for ax, r in zip(axes, subset):
        coll = np.array(r["coll"])
        rm   = _rolling_mean(coll, rolling_w)
        xs   = np.arange(len(rm))
        ax.plot(xs, rm, color="#3498db", linewidth=0.8, alpha=0.7)
        ax.axvline(r["onset_token"], color="#e74c3c", linestyle="--",
                   linewidth=1.2, label="n-gram onset")
        if r.get("signal_rise_token") is not None:
            ax.axvline(r["signal_rise_token"], color="#f39c12", linestyle=":",
                       linewidth=1.2, label="signal rise")
        ax.set_title(f"{r['task_id']} s{r['sample_idx']}\n"
                     f"onset={r['onset_token']}  "
                     f"rise={r.get('signal_rise_token', 'none')}  "
                     f"lead={r.get('lead_time', 'n/a')}",
                     fontsize=7)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.2)

    # hide unused axes
    for ax in axes[len(subset):]:
        ax.set_visible(False)

    handles = [
        plt.Line2D([0], [0], color="#e74c3c", linestyle="--", label="n-gram onset"),
        plt.Line2D([0], [0], color="#f39c12", linestyle=":", label="Σpᵢ² rise"),
    ]
    fig.legend(handles=handles, loc="lower right", fontsize=9)
    fig.suptitle("Individual Σpᵢ² traces — looping samples", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--jsonl", required=True, type=Path,
                   help="Path to the ATCODER JSONL results file")
    p.add_argument("--model", required=True,
                   help="HuggingFace model id (used to look up loop params and load weights)")
    p.add_argument("--output-dir", type=Path, default=None,
                   help="Output directory (default: results/signal_diagnostic/<model_slug>)")
    p.add_argument("--max-samples", type=int, default=MAX_SAMPLES_PER_CLASS,
                   help="Max samples per class (clean / looping_completed)")
    p.add_argument("--classify-only", action="store_true",
                   help="Only classify samples and print stats; skip model inference")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dtype", choices=["bfloat16", "float16"], default="bfloat16")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Resolve loop params
    lp = LOOP_PARAMS.get(args.model)
    if lp is None:
        sys.exit(f"[error] No validated loop params for model '{args.model}'. "
                 f"Known models: {list(LOOP_PARAMS)}")

    model_slug = args.model.replace("/", "--")
    out_dir = args.output_dir or Path("results/signal_diagnostic") / model_slug
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[diagnostic] model={args.model}  loop params: n={lp['n']} k={lp['k']} window={lp['window']}")
    print(f"[diagnostic] output_dir={out_dir}")

    # ---- Step 1: load tokenizer and classify ----
    print("\n[1/4] Loading tokenizer and classifying samples ...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    buckets = classify_samples(args.jsonl, tokenizer, lp)
    for cls, items in buckets.items():
        print(f"  {cls:25s}: {len(items):4d}")

    if args.classify_only:
        print("\n[classify-only] done.")
        return

    # ---- Step 2: subsample ----
    print(f"\n[2/4] Subsampling (max {args.max_samples} per class) ...")
    clean_pool    = random.sample(buckets["clean"],
                                  min(args.max_samples, len(buckets["clean"])))
    looping_pool  = random.sample(buckets["looping_completed"],
                                  min(args.max_samples, len(buckets["looping_completed"])))
    print(f"  clean:             {len(clean_pool)}")
    print(f"  looping_completed: {len(looping_pool)}")

    # ---- Step 3: teacher-forced inference ----
    per_sample_path = out_dir / "per_sample.jsonl"
    done_keys: set[tuple] = set()
    if per_sample_path.exists():
        with open(per_sample_path) as f:
            for line in f:
                r = json.loads(line)
                done_keys.add((r["task_id"], r["sample_idx"], r["cls"]))
        print(f"  [resume] {len(done_keys)} samples already done")

    to_process = [
        (entry, "looping_completed") for entry in looping_pool
        if (entry["task_id"], entry["sample_idx"], "looping_completed") not in done_keys
    ] + [
        (entry, "clean") for entry in clean_pool
        if (entry["task_id"], entry["sample_idx"], "clean") not in done_keys
    ]

    if to_process:
        print(f"\n[3/4] Loading model for teacher-forcing ({len(to_process)} samples) ...")
        import torch
        from bench.generator import load_model_and_tokenizer
        model, _ = load_model_and_tokenizer(args.model, dtype=args.dtype)
        model.eval()
        device = next(model.parameters()).device
        print(f"  model loaded on {device}")

        with open(per_sample_path, "a") as f_out:
            for i, (entry, cls) in enumerate(to_process):
                task_id   = entry["task_id"]
                s_idx     = entry["sample_idx"]
                onset     = entry["onset_token"]
                think_tok = entry["think_tokens"]

                # build full input: prompt + think tokens
                prompt_ids  = tokenizer.encode(entry["prompt_text"],
                                               add_special_tokens=False)
                total_len   = len(prompt_ids) + len(think_tok)
                if total_len > MAX_CTX_TOKENS:
                    # truncate think block to fit
                    think_tok = think_tok[: MAX_CTX_TOKENS - len(prompt_ids)]

                input_ids = torch.tensor(
                    [prompt_ids + think_tok], dtype=torch.long, device=device
                )
                prompt_len = len(prompt_ids)

                try:
                    H, coll = teacher_forced_signals(model, input_ids, prompt_len)
                except Exception as e:
                    print(f"  [warn] {task_id}[{s_idx}] failed: {e}", file=sys.stderr)
                    continue

                signal_rise = (
                    find_signal_rise(coll, onset) if (onset is not None and cls == "looping_completed")
                    else None
                )
                lead_time = (
                    (onset - signal_rise) if (signal_rise is not None and onset is not None)
                    else None
                )

                rec = dict(
                    task_id=task_id,
                    sample_idx=s_idx,
                    cls=cls,
                    onset_token=onset,
                    n_think_tokens=len(coll),
                    signal_rise_token=signal_rise,
                    lead_time=lead_time,
                    H=H,
                    coll=coll,
                )
                f_out.write(json.dumps(rec) + "\n")
                f_out.flush()

                marker = (f"onset={onset}  rise={signal_rise}  lead={lead_time}"
                          if cls == "looping_completed" else "clean")
                print(f"  [{i+1}/{len(to_process)}] {task_id}[{s_idx}] {cls}  {marker}")

        del model
    else:
        print("\n[3/4] All samples already processed — loading saved results.")

    # ---- Step 4: analyse + plot ----
    print("\n[4/4] Analysing and plotting ...")
    looping_results, clean_results = [], []
    with open(per_sample_path) as f:
        for line in f:
            r = json.loads(line)
            if r["cls"] == "looping_completed":
                looping_results.append(r)
            elif r["cls"] == "clean":
                clean_results.append(r)

    # Lead-time statistics
    lead_times = [r["lead_time"] for r in looping_results if r["lead_time"] is not None]
    detected   = [r for r in looping_results if r["signal_rise_token"] is not None]

    stats = {
        "model": args.model,
        "loop_params": lp,
        "n_looping": len(looping_results),
        "n_clean":   len(clean_results),
        "signal_rise_detection_rate": len(detected) / len(looping_results) if looping_results else 0,
        "lead_time_tokens": {
            "n":      len(lead_times),
            "median": float(np.median(lead_times)) if lead_times else None,
            "mean":   float(np.mean(lead_times))   if lead_times else None,
            "p10":    float(np.percentile(lead_times, 10)) if lead_times else None,
            "p25":    float(np.percentile(lead_times, 25)) if lead_times else None,
            "p75":    float(np.percentile(lead_times, 75)) if lead_times else None,
            "p90":    float(np.percentile(lead_times, 90)) if lead_times else None,
            "negative_pct": float(100 * sum(1 for x in lead_times if x < 0) / len(lead_times)) if lead_times else None,
        },
        "interpretation": (
            "STRONG SIGNAL — proceed to CUSUM" if lead_times and float(np.median(lead_times)) > 200
            else "MARGINAL — add SpecRA confirmation layer" if lead_times and float(np.median(lead_times)) > 50
            else "WEAK — logit signal lags; consider hidden-state probe"
        ),
    }

    stats_path = out_dir / "lead_time_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2))

    print("\n" + "=" * 60)
    print("SIGNAL DIAGNOSTIC RESULTS")
    print("=" * 60)
    print(f"  looping samples:        {stats['n_looping']}")
    print(f"  clean samples:          {stats['n_clean']}")
    print(f"  signal rise detected:   {len(detected)}/{len(looping_results)} "
          f"({100*stats['signal_rise_detection_rate']:.0f}%)")
    if lead_times:
        lt = stats["lead_time_tokens"]
        print(f"  lead time (tokens):")
        print(f"    median  {lt['median']:>8.0f}")
        print(f"    mean    {lt['mean']:>8.0f}")
        print(f"    p10     {lt['p10']:>8.0f}")
        print(f"    p90     {lt['p90']:>8.0f}")
        print(f"    negative (signal LAGS onset): {lt['negative_pct']:.0f}%")
    print(f"\n  VERDICT: {stats['interpretation']}")
    print("=" * 60)

    plot_mean_trajectory(looping_results, clean_results,
                         out_dir / "mean_trajectory.png")
    plot_individual_traces(looping_results, out_dir / "individual_traces.png")
    print(f"\n[diagnostic] done.  stats → {stats_path}")


if __name__ == "__main__":
    main()
