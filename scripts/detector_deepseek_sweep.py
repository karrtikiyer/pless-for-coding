"""Offline detector window/k tuning for DeepSeek-R1-Distill-Llama-8B (NO GPU).

Sweeps window and k on real traces — FP on productive reasoning (closed </think>) vs catch
on looping traces (never closed </think>) — to pick the n-gram loop-detector operating point.

CONSISTENCY: this uses ``scripts.repeat_detector.scan`` — the SAME detection logic the live
detector deploys (``RepeatDetector.update``), proven bit-identical in
``tests/test_repeat_detector.py``. It does NOT re-implement a strided approximation (the old
``fires()`` did, which under/over-counted vs the deployed detector and drifted from it).

Also uses the SAFE ``PreTrainedTokenizerFast`` (not ``AutoTokenizer`` → the broken DeepSeek
LlamaTokenizer that mangles whitespace, #45488), so the token stream matches generation.

Run (post-fix traces): HF_HUB_OFFLINE=1 PYTHONPATH=. \\
    DS=results/_deepseek_fixed_full252/deepseek-ai--DeepSeek-R1-Distill-Llama-8B/ATCODER_interview \\
    uv run python scripts/detector_deepseek_sweep.py
"""
import json
import os

from transformers import PreTrainedTokenizerFast

from scripts.repeat_detector import scan

MODEL = os.environ.get("MODEL", "deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
# Default to the UN-MANGLED (post-fix) traces. Override DS for another config dir.
DS = os.environ.get(
    "DS",
    "results/_deepseek_fixed_full252/deepseek-ai--DeepSeek-R1-Distill-Llama-8B/ATCODER_interview",
)
ALPHA2_JSONL = os.environ.get("ALPHA2_JSONL", "pless_think_t1.0_t1.0.jsonl")
N = int(os.environ.get("N", "30"))
WINDOWS = [int(w) for w in os.environ.get("WINDOWS", "1200,1600,2000,3000,4000").split(",")]
KS = [int(x) for x in os.environ.get("KS", "6,8").split(",")]
N_SUCC = int(os.environ.get("N_SUCC", "400"))   # cap productive sample for runtime


def fired(toks, n, k, window):
    """True iff the deployed detector would fire anywhere in ``toks`` (via scan())."""
    return scan(toks, n, k, window)[0]


def main():
    tok = PreTrainedTokenizerFast.from_pretrained(MODEL)   # SAFE tokenizer

    loop_txt, prod_txt = [], []
    for line in open(f"{DS}/{ALPHA2_JSONL}"):
        r = json.loads(line)
        for sw in r.get("samples_with_thinking", []):
            if "</think>" not in sw:
                loop_txt.append(sw)                        # looping (catch target)
            else:
                prod_txt.append(sw.split("</think>")[0])   # productive think (FP target)
    prod_txt = prod_txt[:N_SUCC]

    print(f"tokenizing {len(loop_txt)} loopers + {len(prod_txt)} productive (safe tok)...",
          flush=True)
    loops = [tok.encode(s, add_special_tokens=False) for s in loop_txt]
    prods = [tok.encode(s, add_special_tokens=False) for s in prod_txt]

    print(f"detector sweep via scan() (== deployed RepeatDetector), n={N}", flush=True)
    print(f"  data: {DS}/{ALPHA2_JSONL}", flush=True)
    header = "  window | " + " | ".join(f"k={k} catch/FP" for k in KS)
    print(header, flush=True)
    for w in WINDOWS:
        cells = []
        for k in KS:
            ca = sum(fired(t, N, k, w) for t in loops) / len(loops) * 100
            fp = sum(fired(t, N, k, w) for t in prods) / len(prods) * 100
            cells.append(f"{ca:.1f}/{fp:.1f}")
        print(f"  {w:<6} | " + " | ".join(f"{c:<12}" for c in cells), flush=True)
    print("catch% = looping traces caught; FP% = productive reasoning wrongly cut.", flush=True)


if __name__ == "__main__":
    main()
