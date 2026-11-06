"""GPU regression check for bench/generator_vllm.py: old vs new code must produce byte-identical samples.

Runs the call patterns other experiments use (string prompts, the engine's own tokenizer, seeded):
  * generate_samples_vllm("pless", T=1.0), ("pless_norm", T=0.7), ("pless_renyi", renyi_k=0.1)
  * generate_samples_standard_vllm(T=0.7, top_p=0.95, top_k=20)
and, with --meta (new code only), the same calls with return_meta=True, whose texts must equal the defaults.

One process per code version (one vLLM engine per process). Usage on the GPU box:
    git worktree add /tmp/pless_old <old-commit>
    VLLM_USE_V2_MODEL_RUNNER=0 VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ENABLE_V1_MULTIPROCESSING=0 \\
      <venv>/bin/python scripts/check_generator_vllm_regression.py --code-root /tmp/pless_old --out old.json
    ... --code-root . --out new.json --meta
    <venv>/bin/python scripts/check_generator_vllm_regression.py --compare old.json new.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

MODEL = "Qwen/Qwen2.5-Coder-1.5B"
PROMPTS = [
    'def add(a: int, b: int) -> int:\n    """Return a + b."""\n',
    'from typing import List\n\n\ndef running_max(xs: List[int]) -> List[int]:\n    """Running maximum."""\n',
    'def is_palindrome(s: str) -> bool:\n    """True if s reads the same backwards."""\n',
]
CALLS = [
    ("pless_T1.0", "vllm", dict(sampler_name="pless", temperature=1.0)),
    ("pless_norm_T0.7", "vllm", dict(sampler_name="pless_norm", temperature=0.7)),
    ("pless_renyi_k0.1", "vllm", dict(sampler_name="pless_renyi", temperature=1.0, renyi_k=0.1)),
    ("standard_T0.7_p0.95_k20", "standard", dict(temperature=0.7, top_p=0.95, top_k=20)),
]


def generate(code_root: Path, meta: bool) -> dict:
    sys.path.insert(0, str(code_root.resolve()))
    import bench.generator_vllm as gv
    assert Path(gv.__file__).resolve().is_relative_to(code_root.resolve()), gv.__file__
    engine = gv.load_engine(MODEL, max_model_len=1024, gpu_memory_utilization=0.85)
    tok = engine.get_tokenizer()
    out = {"code_root": str(code_root), "module": gv.__file__, "results": {}}
    for name, kind, kw in CALLS:
        for i, prompt in enumerate(PROMPTS):
            common = dict(n_samples=4, max_new_tokens=96, stop_strings=None, seed=123)
            fn = gv.generate_samples_vllm if kind == "vllm" else gv.generate_samples_standard_vllm
            args = (engine, tok, prompt) + ((kw["sampler_name"],) if kind == "vllm" else ())
            rest = {k: v for k, v in kw.items() if k != "sampler_name"}
            out["results"][f"{name}|{i}"] = fn(*args, **common, **rest)
            if meta:
                texts, metas = fn(*args, **common, **rest, return_meta=True)
                out["results"][f"{name}|{i}|meta_texts"] = texts
                out["results"][f"{name}|{i}|meta"] = metas
    return out


def compare(old_path: Path, new_path: Path) -> int:
    old, new = json.loads(old_path.read_text()), json.loads(new_path.read_text())
    bad = 0
    for key, old_samples in old["results"].items():
        if key.endswith(("|meta", "|meta_texts")):
            continue
        if new["results"].get(key) != old_samples:
            bad += 1
            print("DIFFERENT (old vs new default call):", key)
        mt = new["results"].get(f"{key}|meta_texts")
        if mt is not None and mt != new["results"][key]:
            bad += 1
            print("DIFFERENT (new default vs new return_meta texts):", key)
    n = sum(1 for k in old["results"] if not k.endswith(("|meta", "|meta_texts")))
    print(f"compared {n} calls x 4 samples: {'IDENTICAL' if not bad else f'{bad} differences'}")
    return 1 if bad else 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--code-root", type=Path)
    p.add_argument("--out", type=Path)
    p.add_argument("--meta", action="store_true")
    p.add_argument("--compare", nargs=2, type=Path)
    a = p.parse_args()
    if a.compare:
        return compare(*a.compare)
    a.out.write_text(json.dumps(generate(a.code_root, a.meta), indent=1))
    print("wrote", a.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
