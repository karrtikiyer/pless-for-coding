"""Two-phase vLLM adaptive reconstruction: α=2 → chop at loop onset → continue at α=5.

Phase-1 is REUSED from an existing α=2 run (zero pipeline offset vs the α=2 baseline — the
non-fired samples ARE that baseline). We re-tokenize each α=2 sample with the safe tokenizer,
run the DEPLOYED detector (``scan()`` == ``RepeatDetector.update``) over its think phase, and
for fired samples chop to the loop onset. Phase-2 continues each chopped prefix at α=5 on
vLLM via ``TokensPrompt`` (partial-``<think>`` prefill), batched. Output is a base-schema
JSONL (method ``pless_adaptive``) scored later with ``bench.eval`` — apple-to-apple with the
vLLM α=2 (same phase-1) and vLLM α=5 (same α=5 sampler / backend / scorer).

Per-model config (match the HF adaptive detector so the port reproduces it):
  QWEN:     ALPHA2=results/pless_cot_efficiency_vllm/Qwen--Qwen3-8B/ATCODER_interview_all_252/pless_think_t1.0_t1.0.jsonl
            MODEL=Qwen/Qwen3-8B LOOP_N=30 LOOP_K=6 LOOP_WINDOW=1600
  DEEPSEEK: ALPHA2=results/_deepseek_fixed_full252/deepseek-ai--DeepSeek-R1-Distill-Llama-8B/ATCODER_interview/pless_think_t1.0_t1.0.jsonl
            MODEL=deepseek-ai/DeepSeek-R1-Distill-Llama-8B LOOP_N=30 LOOP_K=8 LOOP_WINDOW=3000

Modes:
  DETECT_ONLY=1  → phase-1 only (CPU, no vLLM): prints fired rate / onset stats. Run on Mac
                   to validate detection before spending GPU.
  (default)      → phase-1 + phase-2 (needs vLLM/GPU) → writes OUT jsonl.

Run (DeepSeek, pod):
  VLLM_VENV=/workspace/vllm_env/.venv MODEL=deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
  ALPHA2=results/_deepseek_fixed_full252/deepseek-ai--DeepSeek-R1-Distill-Llama-8B/ATCODER_interview/pless_think_t1.0_t1.0.jsonl \
  LOOP_N=30 LOOP_K=8 LOOP_WINDOW=3000 GPUS=0 \
  OUT=results/_deepseek_fixed_full252/deepseek-ai--DeepSeek-R1-Distill-Llama-8B/ATCODER_interview/pless_adaptive_recon.jsonl \
  uv run python scripts/vllm_adaptive_reconstruct.py
"""
import gzip
import json
import os
import sys
from datetime import datetime, timezone

# Make `import scripts.*` / `import bench.*` resolve when invoked as
# `python scripts/vllm_adaptive_reconstruct.py` without PYTHONPATH set (running a file
# puts its own dir on sys.path, not the repo root).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.repeat_detector import scan


def _open_text(path):
    """Open a JSONL, transparently handling gzip (.gz) — the α=2 traces are large
    (100-300 MB) so they're often transferred/kept compressed (~9x smaller)."""
    return gzip.open(path, "rt") if path.endswith(".gz") else open(path)

MODEL = os.environ["MODEL"]
ALPHA2 = os.environ["ALPHA2"]
SOURCE = os.environ.get("SOURCE", "ATCODER")
DIFFICULTY = os.environ.get("DIFFICULTY", "interview")
LOOP_N = int(os.environ.get("LOOP_N", "30"))
LOOP_K = int(os.environ.get("LOOP_K", "8"))
LOOP_WINDOW = int(os.environ.get("LOOP_WINDOW", "3000"))
ESC_ALPHA = float(os.environ.get("ESC_ALPHA", "5"))
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "32768"))
MIN_CONT = int(os.environ.get("MIN_CONT", "512"))     # floor on the α=5 continuation budget
PHASE2_BATCH = int(os.environ.get("PHASE2_BATCH", "64"))
DETECT_ONLY = os.environ.get("DETECT_ONLY", "") not in ("", "0", "false")
OUT = os.environ.get("OUT", "")
TASK_IDS = os.environ.get("TASK_IDS", "")
MAX_PROBLEMS = int(os.environ.get("MAX_PROBLEMS", "0"))  # 0 = all


def _think_phase_text(sample_text: str) -> tuple[str, bool]:
    """Return (think-phase text, closed_think). Detection runs only on the think phase."""
    if "</think>" in sample_text:
        return sample_text.split("</think>", 1)[0], True
    return sample_text, False


def main():
    # --- tokenizer: safe one if the model's default mangles whitespace (DeepSeek), else
    #     the model tokenizer (Qwen is fine). Loaded without vLLM so DETECT_ONLY runs on Mac.
    from transformers import AutoTokenizer, PreTrainedTokenizerFast
    at = AutoTokenizer.from_pretrained(MODEL)
    _t = "a b\nc"
    if at.decode(at.encode(_t, add_special_tokens=False), skip_special_tokens=True).strip() != _t.strip():
        tok = PreTrainedTokenizerFast.from_pretrained(MODEL)
        print(f"[recon] using safe PreTrainedTokenizerFast for {MODEL} (default mangles whitespace)")
    else:
        tok = at
        print(f"[recon] using {at.__class__.__name__} for {MODEL}")

    from bench.apps.dataset import load_apps
    from bench.apps.prompts import format_prompt_apps_instruct
    from bench.generator import _strip_think_content

    want = {int(x) for x in TASK_IDS.split()} if TASK_IDS else None
    problems = {p.problem_id: p for p in load_apps(source=SOURCE, difficulty=DIFFICULTY)}

    recs = [json.loads(l) for l in _open_text(ALPHA2)]
    if want is not None:
        recs = [r for r in recs if r["task_id"] in want]
    if MAX_PROBLEMS:
        recs = recs[:MAX_PROBLEMS]

    # ---- Phase 1 (CPU): detect + chop over the reused α=2 samples ----
    # Per task we keep: prompt_ids, and per-sample either a kept α=2 text (non-fired) or a
    # phase-2 work item (fired: combined prefix ids + continuation budget).
    per_task = {}          # task_id -> {"prompt_ids", "samples": [ {..} per sample ]}
    fired_items = []       # flat list of phase-2 work: (task_id, s_idx, combined_ids, budget, onset, chopped_ids)
    n_samp = None
    for r in recs:
        tid = r["task_id"]
        prob = problems.get(tid)
        if prob is None:
            print(f"[recon] WARN task {tid} not in APPS bucket; skipping")
            continue
        prompt_str, _ = format_prompt_apps_instruct(prob, tok, enable_thinking=True)
        prompt_ids = tok.encode(prompt_str)                 # add_special_tokens default (matches the α=2 run)
        sw = r["samples_with_thinking"]
        n_samp = len(sw)
        slots = []
        for si, text in enumerate(sw):
            think_text, closed = _think_phase_text(text)
            think_toks = tok.encode(think_text, add_special_tokens=False)
            fired, fire_pos, onset = scan(think_toks, LOOP_N, LOOP_K, LOOP_WINDOW)
            if fired:
                chopped = think_toks[:onset]
                budget = max(MIN_CONT, MAX_TOKENS - onset)
                fired_items.append((tid, si, prompt_ids + chopped, budget, onset, chopped))
                slots.append({"fired": True, "onset": onset, "closed_think_alpha2": closed,
                              "text": None})           # filled in phase-2
            else:
                slots.append({"fired": False, "onset": None, "closed_think_alpha2": closed,
                              "text": text})            # keep α=2 verbatim
        per_task[tid] = {"prompt_ids": prompt_ids, "slots": slots, "rec": r}

    n_fired = len(fired_items)
    n_total = sum(len(v["slots"]) for v in per_task.values())
    print(f"[recon] phase-1: {len(per_task)} tasks, {n_total} samples, "
          f"fired {n_fired} ({100*n_fired/max(1,n_total):.1f}%), detector={LOOP_N}/{LOOP_K}/{LOOP_WINDOW}")
    if DETECT_ONLY:
        import statistics as st
        onsets = [it[4] for it in fired_items]
        if onsets:
            print(f"[recon] onset tokens: min={min(onsets)} median={int(st.median(onsets))} max={max(onsets)}")
        print("[recon] DETECT_ONLY — skipping phase-2 (no vLLM).")
        return

    # ---- Phase 2 (GPU): continue each chopped prefix at α=5 on vLLM ----
    # GPU pinning + sampler-path parity (set before vLLM/CUDA init).
    if os.environ.get("GPUS") and not os.environ.get("CUDA_VISIBLE_DEVICES"):
        os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["GPUS"]
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    from bench.generator_vllm import load_engine, resolve_think_end_id
    from vllm import SamplingParams, TokensPrompt
    engine = load_engine(MODEL)
    safe = getattr(engine, "_safe_tokenizer", None) or engine.get_tokenizer()
    think_end_id = resolve_think_end_id(engine.get_tokenizer())

    def cfg_alpha5():
        return {"t_think": 1.0, "t_code": 1.0,
                "sampler_think": "pless_alpha", "sampler_code": "pless_alpha",
                "alpha_think": ESC_ALPHA, "alpha_code": ESC_ALPHA,
                "think_end_id": think_end_id}

    cont_text = {}     # (tid, si) -> full decoded generation (chopped + α=5 continuation)
    for start in range(0, n_fired, PHASE2_BATCH):
        chunk = fired_items[start:start + PHASE2_BATCH]
        prompts = [TokensPrompt(prompt_token_ids=it[2]) for it in chunk]
        sps = [SamplingParams(n=1, max_tokens=it[3], temperature=1.0, top_p=1.0, top_k=-1,
                              extra_args={"pless_split": cfg_alpha5()}) for it in chunk]
        outs = engine.generate(prompts, sps, use_tqdm=False)
        for it, out in zip(chunk, outs):
            tid, si, _combined, _budget, _onset, chopped = it
            cont_ids = list(out.outputs[0].token_ids)
            full = list(chopped) + cont_ids                  # generation = kept prefix + α=5 continuation
            cont_text[(tid, si)] = safe.decode(full, skip_special_tokens=True)
        print(f"[recon] phase-2 {min(start+PHASE2_BATCH, n_fired)}/{n_fired} continued", flush=True)

    # ---- Reassemble + write base-schema JSONL ----
    if not OUT:
        raise SystemExit("set OUT=<path.jsonl> to write results")
    os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
    now = datetime.now(timezone.utc).isoformat()
    with open(OUT, "w") as f:
        for tid, v in per_task.items():
            r = v["rec"]
            texts, adaptive = [], []
            for si, slot in enumerate(v["slots"]):
                if slot["fired"]:
                    txt = cont_text.get((tid, si), "")
                else:
                    txt = slot["text"]
                texts.append(txt)
                adaptive.append({"sample": si, "fired": slot["fired"], "chops": 1 if slot["fired"] else 0,
                                 "onset": slot["onset"], "closed_think": "</think>" in txt, "task_id": tid})
            out = {
                "model": MODEL, "backend": "vllm", "method": "pless_adaptive",
                "temperature": 1.0, "top_p": 1.0, "top_k": 0,
                "task_id": tid, "source": r.get("source", SOURCE), "difficulty": r.get("difficulty", DIFFICULTY),
                "prompt_text": r.get("prompt_text", ""),
                "samples": [_strip_think_content(t) for t in texts],
                "samples_with_thinking": texts,
                "timestamp": now, "base_alpha": 2.0, "esc_alpha": ESC_ALPHA,
                "detector": [LOOP_N, LOOP_K, LOOP_WINDOW], "adaptive": adaptive,
            }
            f.write(json.dumps(out) + "\n")
    print(f"[recon] wrote {OUT} ({len(per_task)} tasks). Score with: "
          f"python -m bench.eval --results-file {OUT} --dataset apps")


if __name__ == "__main__":
    main()
