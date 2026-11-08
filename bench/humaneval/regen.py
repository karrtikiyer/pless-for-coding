"""HumanEval regeneration runner (2026-11).

Spec: docs/experiments/humaneval_regen_2026_11_plan.md. One generator, one prompt specification and
one output folder for the paper's eight models; grade the output with
``python -m bench.eval --dataset humaneval --humaneval-grader official``.

* vLLM 0.33.0 for greedy / temperature / top-p (native ``SamplingParams``) and p-less / p-less-norm
  (our logits processor); HF ``generate(num_beams=4)`` for beam, identical to MBPP's beam.
  One process per backend: run the vLLM configs and the HF beam config in separate invocations.
* Prompts are built once as token ids (a single BOS) and handed to both backends.
* Base models: raw HumanEval prompt, BigCode stop list (also re-applied afterwards); the stored sample
  is the completion only. Chat models: the model's chat template; the stored sample is the reply.

The older runner (bench/humaneval/runner.py) is left unchanged so historical files stay reproducible.

Usage (GPU box, repo root, the envs/vllm-033 environment):
    VLLM_USE_V2_MODEL_RUNNER=0 VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_ENABLE_V1_MULTIPROCESSING=0 \\
    .venv-vllm033/bin/python -m bench.humaneval.regen --model Qwen/Qwen2.5-Coder-1.5B \\
        --configs greedy,temp_t0.7,top_p0.9_t1.0,pless_t0.7,pless_norm_t0.7,top_p0.95_t0.2
    .venv-vllm033/bin/python -m bench.humaneval.regen --model Qwen/Qwen2.5-Coder-1.5B --configs beam4

Diagnostics (not part of the default config sets; see run_regen_humaneval_8models.sh diag):
    --configs greedy_hf                       HF greedy (num_beams=1), same HF path as beam4: backend effect
    --configs greedy --prompt-variant stripped  prompt.strip() (BigCode task `humaneval`), base models only:
                                              prompt effect; written to <config>__prompt_stripped.jsonl

Stripped-prompt protocol (user decision 2026-11-07, run log): ``PROMPT_VARIANT=stripped
./run_regen_humaneval_8models.sh <mode>`` passes ``--prompt-variant stripped`` to the raw-prompted models only
(``prompt_variant_args``) and writes to ``results/humaneval_regen_2026_11_stripped*``.

Extra comparison configs (user request 2026-11-08, NOT for the paper): ``EXTRA_CONFIGS`` (p-less / p-less-norm
at T=1.0 and T=0.6, otherwise identical to ``pless_t0.7`` / ``pless_norm_t0.7``). Never in a default set; only
accepted with an ``*_extra`` out-root. ``PROMPT_VARIANT=stripped ./run_regen_humaneval_8models.sh extra [IDS...]``
writes them to ``results/humaneval_regen_2026_11_stripped_extra``.

Analysis-only extra model (user request 2026-11-08, NOT for the paper): ``Qwen/Qwen2.5-Coder-7B-Instruct``
(``INSTRUCT7B_MODELS``), chat protocol, the six common configs, no Hasan config.
``./run_regen_humaneval_8models.sh instruct7b`` writes ``results/humaneval_regen_2026_11_instruct7b``; the model
is refused in every other root and other models are refused in that one (``check_model_for_root``).
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import platform
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from bench.generator import generate_samples_beam
from bench.generator_vllm import generate_samples_standard_vllm, generate_samples_vllm
from bench.humaneval.prompts import BIGCODE_HUMANEVAL_STOP_SEQUENCES, instruct_messages, truncate_at_stop

DEFAULT_OUT_ROOT = Path("results/humaneval_regen_2026_11")
_ALLOWED_ROOT_PREFIX = "humaneval_regen_2026_11"   # also allows e.g. results/humaneval_regen_2026_11_smoke
DEFAULT_SEED = 42
DEFAULT_MAX_NEW_TOKENS = 512
DATASET = ("openai/openai_humaneval", "test")


# ---------------------------------------------------------------------------
# Models and configs (plan §3-§4)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    prompt_style: str          # "base" (raw prompt) | "chat" (chat template)
    hasan: bool = False        # shared with Hasan et al. -> also runs their config


MODELS = [  # run order: small models first, 7B last
    ModelSpec("m-a-p/OpenCodeInterpreter-DS-1.3B", "base", hasan=True),   # instruction-tuned; completion prompt as in MBPP
    ModelSpec("Qwen/Qwen2.5-Coder-1.5B", "base", hasan=True),
    ModelSpec("Qwen/Qwen2.5-Coder-3B", "base", hasan=True),
    ModelSpec("codellama/CodeLlama-7b-hf", "base"),
    ModelSpec("codellama/CodeLlama-7b-Instruct-hf", "chat"),
    ModelSpec("meta-llama/Llama-2-7b-hf", "base"),
    ModelSpec("meta-llama/Llama-2-7b-chat-hf", "chat"),
    ModelSpec("Qwen/Qwen2.5-Coder-7B", "base", hasan=True),   # user decision 2026-11-07: base, replaces -Instruct
]
# Analysis-only extra model (user request 2026-11-08, NOT for the paper): Qwen2.5-Coder-7B-Instruct with our chat
# protocol, as CodeLlama-7b-Instruct / Llama-2-7b-chat. Not one of the eight; only the six common configs; written
# only to a ``humaneval_regen_2026_11_instruct7b*`` root (check_model_for_root), via
# ``run_regen_humaneval_8models.sh instruct7b``.
INSTRUCT7B_MODELS = [ModelSpec("Qwen/Qwen2.5-Coder-7B-Instruct", "chat")]
INSTRUCT7B_ROOT_PREFIX = "humaneval_regen_2026_11_instruct7b"
_MODEL_BY_ID = {m.model_id: m for m in MODELS + INSTRUCT7B_MODELS}
_INSTRUCT7B_IDS = {m.model_id for m in INSTRUCT7B_MODELS}


def get_model(model_id: str) -> ModelSpec:
    return _MODEL_BY_ID[model_id]   # KeyError for anything outside the plan


def check_model_for_root(model_id: str, out_root: Path) -> None:
    """The instruct7b model goes only to a ``humaneval_regen_2026_11_instruct7b*`` root, and only it goes there.
    SystemExit otherwise (keeps it out of the paper roots and out of the small / large / extra / fit runs)."""
    is_i7_root = Path(out_root).name.startswith(INSTRUCT7B_ROOT_PREFIX)
    is_i7_model = model_id in _INSTRUCT7B_IDS
    if is_i7_model and not is_i7_root:
        raise SystemExit(f"{model_id} is the analysis-only instruct7b model; write it to a {INSTRUCT7B_ROOT_PREFIX}* "
                         f"root (run_regen_humaneval_8models.sh instruct7b), not {out_root}")
    if is_i7_root and not is_i7_model:
        raise SystemExit(f"the instruct7b root {out_root} holds only {sorted(_INSTRUCT7B_IDS)}; got {model_id}")


@dataclass(frozen=True)
class Config:
    backend: str               # "vllm" | "hf"
    method: str                # greedy | beam | temp | top_p | pless | pless_norm
    n: int                     # samples per task
    temperature: float
    top_p: float = 1.0
    sampler: str | None = None     # p-less logits processor name (vLLM)
    num_beams: int | None = None   # HF beam width


CONFIGS = {
    "greedy":          Config("vllm", "greedy", 1, 0.0),
    "beam4":           Config("hf", "beam", 1, 1.0, num_beams=4),
    "temp_t0.7":       Config("vllm", "temp", 10, 0.7, 1.0),
    "top_p0.9_t1.0":   Config("vllm", "top_p", 10, 1.0, 0.9),
    "pless_t0.7":      Config("vllm", "pless", 10, 0.7, sampler="pless"),
    "pless_norm_t0.7": Config("vllm", "pless_norm", 10, 0.7, sampler="pless_norm"),
    "top_p0.95_t0.2":  Config("vllm", "top_p", 10, 0.2, 0.95),   # Hasan et al.'s configuration
    # Diagnostic only: HF greedy through the beam4 code path (generate(num_beams=1, do_sample=False)).
    "greedy_hf":       Config("hf", "greedy", 1, 0.0, num_beams=1),
    # Extra comparison configs (user request 2026-11-08, NOT for the paper): identical to pless_t0.7 /
    # pless_norm_t0.7 except temperature. Written only to an ``*_extra`` root (check_configs_for_root).
    "pless_t1.0":      Config("vllm", "pless", 10, 1.0, sampler="pless"),
    "pless_norm_t1.0": Config("vllm", "pless_norm", 10, 1.0, sampler="pless_norm"),
    "pless_t0.6":      Config("vllm", "pless", 10, 0.6, sampler="pless"),
    "pless_norm_t0.6": Config("vllm", "pless_norm", 10, 0.6, sampler="pless_norm"),
}
COMMON_CONFIGS = ["greedy", "temp_t0.7", "top_p0.9_t1.0", "pless_t0.7", "pless_norm_t0.7", "beam4"]
HASAN_CONFIGS = ["top_p0.95_t0.2"]
DIAG_CONFIGS = ["greedy_hf"]   # accepted via --configs, never in the default (small / large) sets
EXTRA_CONFIGS = ["pless_t1.0", "pless_norm_t1.0", "pless_t0.6", "pless_norm_t0.6"]   # `extra` mode only
EXTRA_ROOT_SUFFIX = "_extra"


def configs_for(model_id: str) -> list[str]:
    """The plan's configs for a model (what the default runs execute)."""
    return COMMON_CONFIGS + (HASAN_CONFIGS if get_model(model_id).hasan else [])


def allowed_configs(model_id: str) -> list[str]:
    """Configs accepted by ``--configs``: the plan's configs plus the diagnostic and extra ones
    (the instruct7b model: its six configs only)."""
    if model_id in _INSTRUCT7B_IDS:
        return configs_for(model_id)
    return configs_for(model_id) + DIAG_CONFIGS + EXTRA_CONFIGS


def check_configs_for_root(names: list[str], out_root: Path) -> None:
    """Keep the extra comparison configs out of the paper roots, and only them in an ``*_extra`` root.

    SystemExit if an extra config targets a root whose name does not end in ``_extra``, or if a non-extra
    config targets an ``*_extra`` root."""
    is_extra_root = Path(out_root).name.endswith(EXTRA_ROOT_SUFFIX)
    extra = [n for n in names if n in EXTRA_CONFIGS]
    other = [n for n in names if n not in EXTRA_CONFIGS]
    if extra and not is_extra_root:
        raise SystemExit(f"configs {extra} are extra comparison configs; write them to a *{EXTRA_ROOT_SUFFIX} root "
                         f"(run_regen_humaneval_8models.sh extra), not {out_root}")
    if other and is_extra_root:
        raise SystemExit(f"the *{EXTRA_ROOT_SUFFIX} root {out_root} holds only the extra configs {EXTRA_CONFIGS}; "
                         f"got {other}")


def resolve_configs(model_id: str, configs_arg: str | None) -> tuple[list[str], str]:
    """Config names to run and their single shared backend; SystemExit on unknown names or mixed backends."""
    names = configs_arg.split(",") if configs_arg else configs_for(model_id)
    unknown = [n for n in names if n not in allowed_configs(model_id)]
    if unknown:
        raise SystemExit(f"configs {unknown} are not in the plan for {model_id}")
    backends = {CONFIGS[n].backend for n in names}
    if len(backends) != 1:
        raise SystemExit(f"configs {names} mix backends {backends}; run vLLM and HF configs separately")
    return names, backends.pop()


# ---------------------------------------------------------------------------
# Prompts (plan §5-§6)
# ---------------------------------------------------------------------------

def load_tokenizer(model_id: str):
    """Tokenizer with the same whitespace round-trip guard as bench/generator.py (OCI / DeepSeek-Coder)."""
    from transformers import AutoTokenizer, PreTrainedTokenizerFast

    tok = AutoTokenizer.from_pretrained(model_id)
    probe = "a b\nc"
    if tok.decode(tok.encode(probe), skip_special_tokens=True).strip() != probe:
        tok = PreTrainedTokenizerFast.from_pretrained(model_id)
        if tok.decode(tok.encode(probe), skip_special_tokens=True).strip() != probe:
            raise RuntimeError(f"tokenizer for {model_id} cannot round-trip whitespace")
    return tok


PROMPT_VARIANTS = ("unstripped", "stripped")
DEFAULT_PROMPT_VARIANT = "unstripped"


def apply_prompt_variant(prompt: str, style: str, variant: str) -> str:
    """The task prompt actually given to the model (and stored as ``task_prompt`` for the grader).

    * ``unstripped`` (default, plan §6): the raw HumanEval prompt, ending in the docstring's closing quotes plus a newline (BigCode task
      ``humaneval-unstripped``); returned unchanged.
    * ``stripped``: ``prompt.strip()`` (BigCode task ``humaneval``). Base prompts only; refused for chat
      models, whose prompt is wrapped in a chat template.
    """
    if variant == "unstripped":
        return prompt
    if variant == "stripped":
        if style != "base":
            raise ValueError(f"--prompt-variant stripped applies to base-style prompts only (style {style!r})")
        return prompt.strip()
    raise ValueError(f"unknown prompt variant {variant!r}; expected one of {PROMPT_VARIANTS}")


def prompt_variant_args(model_id: str, variant: str) -> list[str]:
    """Runner arguments that select ``variant`` for one model (used by run_regen_humaneval_8models.sh).

    ``stripped`` applies to raw-prompted (base-style) models only; chat models get no flag (the runner refuses
    stripped for them) and keep their chat-template prompt. ``unstripped`` (the default) adds nothing.
    """
    style = get_model(model_id).prompt_style          # KeyError for models outside the plan
    if variant not in PROMPT_VARIANTS:
        raise ValueError(f"unknown prompt variant {variant!r}; expected one of {PROMPT_VARIANTS}")
    if variant == DEFAULT_PROMPT_VARIANT or style != "base":
        return []
    return ["--prompt-variant", variant]


def build_prompt_ids(tokenizer, task_prompt: str, style: str) -> list[int]:
    """Prompt as token ids with exactly one BOS (where the model has one)."""
    if style == "base":
        ids = tokenizer(task_prompt)["input_ids"]
    elif style == "chat":
        ids = tokenizer.apply_chat_template(
            instruct_messages(task_prompt), tokenize=True, add_generation_prompt=True,
            return_dict=False, enable_thinking=False,
        )
    else:
        raise ValueError(f"unknown prompt style {style!r}")
    if hasattr(ids, "input_ids"):
        ids = ids["input_ids"]
    return [int(i) for i in ids]


def count_leading_bos(ids: list[int], tokenizer) -> int:
    bos = tokenizer.bos_token_id
    if bos is None:
        return 0
    n = 0
    while n < len(ids) and ids[n] == bos:
        n += 1
    return n


def context_length(prompt_lens: list[int], max_new_tokens: int, override: int | None = None) -> int:
    """vLLM ``max_model_len``: longest prompt over ALL tasks + new tokens, rounded up to 64.
    Always computed on the full task list, so a 5-task smoke/fit run is sized like the full run."""
    if override:
        return override
    return -(-(max(prompt_lens) + max_new_tokens) // 64) * 64


def check_prompt_fits(prompt_len: int, max_new_tokens: int, max_model_len: int, task_id: str) -> None:
    if prompt_len + max_new_tokens > max_model_len:
        raise ValueError(f"{task_id}: prompt {prompt_len} + {max_new_tokens} new tokens > max_model_len {max_model_len}")


# ---------------------------------------------------------------------------
# Generation (plan §4-§5)
# ---------------------------------------------------------------------------

def generate(config: Config, backend_obj, tokenizer, prompt_ids: list[int], *, style: str,
             seed: int, max_new_tokens: int) -> tuple[list[str], list[dict], bool]:
    """Raw completion texts for one task, per-sample finish metadata, and whether the backend itself
    applied the stop strings. ``backend_obj`` is a vLLM engine or an HF model.

    When vLLM's own detokenizer is broken for the model (``engine._safe_tokenizer`` set, e.g. OCI /
    DeepSeek-Coder, whose vLLM text loses whitespace), stop strings matched on that text could fire early
    (``"\n    if"`` read as ``"\nif"``) or never. Generation then runs without them and the completion is
    cut afterwards on the correctly decoded text: the same sampling distribution up to the stop (sampling is
    seeded per request and autoregressive); tokens can differ only through batch-size numerics, since a
    sample that would have stopped keeps running. vLLM's text is also audited against the HF tokenizer
    per sample (``sample_meta[i]["text_matches_reference"]``).
    """
    stops = list(BIGCODE_HUMANEVAL_STOP_SEQUENCES) if style == "base" else None
    if config.backend == "hf":
        texts, metas = generate_samples_beam(backend_obj, tokenizer, prompt_ids, num_beams=config.num_beams,
                                             max_new_tokens=max_new_tokens, stop_strings=stops, return_meta=True)
        return texts, metas, stops is not None
    if getattr(backend_obj, "_safe_tokenizer", None) is not None:
        stops = None
    if config.sampler is not None:
        texts, metas = generate_samples_vllm(backend_obj, tokenizer, prompt_ids, config.sampler,
                                             n_samples=config.n, max_new_tokens=max_new_tokens,
                                             temperature=config.temperature, stop_strings=stops, seed=seed,
                                             return_meta=True)
    else:
        texts, metas = generate_samples_standard_vllm(backend_obj, tokenizer, prompt_ids,
                                                      n_samples=config.n, max_new_tokens=max_new_tokens,
                                                      temperature=config.temperature, stop_strings=stops,
                                                      top_p=config.top_p, top_k=0, seed=seed, return_meta=True)
    return texts, metas, stops is not None


def postprocess(raw: str, style: str) -> str:
    """Base: cut at the BigCode stop list (idempotent if the backend already stopped). Chat: unchanged."""
    return truncate_at_stop(raw) if style == "base" else raw


# ---------------------------------------------------------------------------
# Records, paths, resume, manifest
# ---------------------------------------------------------------------------

def make_record(task: dict, *, model_id: str, style: str, config: Config, config_name: str,
                prompt_ids: list[int], raw: list[str], seed: int, max_new_tokens: int,
                versions: dict, metas: list[dict] | None = None, backend_stops: bool | None = None,
                count_tokens=None, prompt_variant: str = DEFAULT_PROMPT_VARIANT) -> dict:
    """One JSONL record. ``task["prompt"]`` is the raw dataset prompt; ``task_prompt`` stores the prompt
    actually used (after ``prompt_variant``), which the official grader prepends to each completion."""
    rec = {
        "task_id": task["task_id"],
        "model": model_id,
        "config": config_name,
        "backend": config.backend,
        "method": config.method,
        "temperature": config.temperature,
        "top_p": config.top_p,
        "prompt_style": style,
        "prompt_variant": prompt_variant,
        "task_prompt": apply_prompt_variant(task["prompt"], style, prompt_variant),
        "prompt_token_ids_head": prompt_ids[:8],
        "num_prompt_tokens": len(prompt_ids),
        "samples": (samples := [postprocess(r, style) for r in raw]),
        # tokens in the stored (cut) completion; sample_meta.num_output_tokens counts the uncut generation
        "completion_num_tokens": [count_tokens(x) for x in samples] if count_tokens else None,
        "raw_samples": raw,
        "test": task["test"],
        "entry_point": task["entry_point"],
        "seed": seed,
        "n": config.n,
        "max_new_tokens": max_new_tokens,
        "stop": list(BIGCODE_HUMANEVAL_STOP_SEQUENCES) if style == "base" else None,
        "versions": versions,
        "sample_meta": metas,                  # finish_reason / stop_reason / num_output_tokens per sample
        "backend_stop_strings": backend_stops,  # False: backend ran without stops, cut afterwards
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    if config.num_beams:
        rec["num_beams"] = config.num_beams
    return rec


def output_path(out_root: Path, model_id: str, config_name: str,
                prompt_variant: str = DEFAULT_PROMPT_VARIANT) -> Path:
    """``<root>/<model>/<config>.jsonl``; a non-default prompt variant adds ``__prompt_<variant>`` so both
    variants of one config can live in the same folder."""
    if prompt_variant not in PROMPT_VARIANTS:
        raise ValueError(f"unknown prompt variant {prompt_variant!r}")
    suffix = "" if prompt_variant == DEFAULT_PROMPT_VARIANT else f"__prompt_{prompt_variant}"
    return Path(out_root) / model_id.replace("/", "--") / f"{config_name}{suffix}.jsonl"


RESUME_KEYS = ("model", "config", "seed", "n", "max_new_tokens", "temperature", "top_p", "prompt_style",
               "prompt_variant")
RESUME_DEFAULTS = {"prompt_variant": DEFAULT_PROMPT_VARIANT}   # files written before the field existed
RESUME_VERSION_KEYS = ("vllm", "transformers", "torch")


def completed_task_ids(path: Path, resume: bool, expected: dict | None = None) -> set[str]:
    """Task ids already written. An existing file is only reused with ``resume=True``, and only when its
    records were written with the same settings and library versions as ``expected``."""
    path = Path(path)
    if not path.exists():
        return set()
    if not resume:
        raise FileExistsError(f"{path} exists; pass --resume to continue it (never overwritten)")
    lines = [ln for ln in Path(path).read_text().split("\n") if ln.strip()]
    recs = []
    for i, line in enumerate(lines):
        try:
            recs.append(json.loads(line))
        except json.JSONDecodeError:
            where = "last" if i == len(lines) - 1 else f"line {i + 1}"
            raise ValueError(f"{path}: {where} record is truncated or corrupt (interrupted write?); "
                             "inspect and remove it by hand before resuming") from None
    if expected and recs:
        first = recs[0]
        def get(d, k):
            return d.get(k, RESUME_DEFAULTS.get(k))
        diff = {k: (get(first, k), get(expected, k)) for k in RESUME_KEYS if get(first, k) != get(expected, k)}
        fv, ev = first.get("versions") or {}, expected.get("versions") or {}
        diff.update({f"versions.{k}": (fv.get(k), ev.get(k)) for k in RESUME_VERSION_KEYS if fv.get(k) != ev.get(k)})
        if fv.get("env") != ev.get("env"):   # vLLM runtime switches (model runner, sampler, multiprocessing)
            diff["versions.env"] = (fv.get("env"), ev.get("env"))
        if diff:
            raise ValueError(f"cannot resume {path}: settings differ (file, now): {diff}")
    return {r["task_id"] for r in recs}


def _check_out_root(out_root: Path) -> None:
    if not Path(out_root).name.startswith(_ALLOWED_ROOT_PREFIX):
        raise ValueError(f"--out-root must be a folder named {_ALLOWED_ROOT_PREFIX}*; got {out_root}")


def collect_versions(device_name: str | None = None) -> dict:
    import importlib.metadata as md

    def ver(pkg):
        try:
            return md.version(pkg)
        except md.PackageNotFoundError:
            return None

    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                                    capture_output=True, text=True, check=True).stdout.strip())
    except Exception:
        commit, dirty = None, None
    try:
        import torch
        cuda = torch.version.cuda
        gpu = device_name or (torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
    except Exception:
        cuda, gpu = None, device_name
    env = {k: os.environ.get(k) for k in ("VLLM_USE_V2_MODEL_RUNNER", "VLLM_USE_FLASHINFER_SAMPLER",
                                           "VLLM_ENABLE_V1_MULTIPROCESSING")}
    return {"vllm": ver("vllm"), "torch": ver("torch"), "transformers": ver("transformers"),
            "datasets": ver("datasets"), "cuda": cuda, "gpu": gpu, "python": platform.python_version(),
            "git_commit": commit, "git_dirty": dirty, "env": env}


def dataset_digest(tasks: list[dict]) -> str:
    """SHA-256 over task_id/prompt/test/entry_point of every task (pins the dataset content)."""
    import hashlib
    h = hashlib.sha256()
    for t in tasks:
        for k in ("task_id", "prompt", "test", "entry_point"):
            h.update(t[k].encode())
            h.update(b"\0")
    return h.hexdigest()


def manifest_entry(*, model: str, config_name: str, config: Config, out: Path, style: str, prompt_variant: str,
                   seed: int, max_new_tokens: int, max_model_len: int, gpu_memory_utilization: float,
                   num_tasks: int, longest: int, digest: str, versions: dict) -> dict:
    return {
        "model": model, "config": config_name, "file": str(out), "started": datetime.now(timezone.utc).isoformat(),
        "config_spec": asdict(config), "prompt_style": style, "prompt_variant": prompt_variant, "seed": seed,
        "max_new_tokens": max_new_tokens, "max_model_len": max_model_len,
        "gpu_memory_utilization": gpu_memory_utilization, "num_tasks": num_tasks,
        "longest_prompt_tokens": longest, "dataset": list(DATASET), "dataset_digest": digest,
        "versions": versions,
    }


def _atomic_write_text(path: Path, text: str) -> None:
    """Write via a temp file in the same folder + ``os.replace``: readers never see a half-written file and a
    kill mid-write leaves the previous version intact."""
    path = Path(path)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise


def readme_text(variants: list[str]) -> str:
    return (
        "# HumanEval regeneration (2026-11)\n\n"
        "Generated by `bench/humaneval/regen.py` per `docs/experiments/humaneval_regen_2026_11_plan.md`.\n"
        "One folder per model, one JSONL per config. Grade with\n"
        "`python -m bench.eval --dataset humaneval --humaneval-grader official --results-file <file>`.\n"
        "Do not mix with older HumanEval folders (`pless_human_eval_results/`).\n\n"
        f"Prompt variant(s) in this root: {', '.join(variants)}.\n"
        "`unstripped` files are `<config>.jsonl`; `stripped` files (prompt.strip(), base models only) are\n"
        "`<config>__prompt_stripped.jsonl`. Each record's `prompt_variant` field is authoritative;\n"
        "see MANIFEST.json for every run.\n"
    )


def write_manifest(out_root: Path, entry: dict) -> None:
    """Append one run entry to <out_root>/MANIFEST.json and (re)write README.md with the prompt variants in the
    root. Safe for parallel runners: an exclusive flock on MANIFEST.json.lock covers the read-modify-write, and
    both files are replaced atomically."""
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    manifest = out_root / "MANIFEST.json"
    readme = out_root / "README.md"
    with open(out_root / "MANIFEST.json.lock", "a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)   # released when the file is closed
        data = json.loads(manifest.read_text()) if manifest.exists() else {
            "plan": "docs/experiments/humaneval_regen_2026_11_plan.md", "runs": []}
        data["runs"].append(entry)
        _atomic_write_text(manifest, json.dumps(data, indent=2) + "\n")
        variants = sorted({r.get("prompt_variant", DEFAULT_PROMPT_VARIANT) for r in data["runs"]})
        text = readme_text(variants)
        if not readme.exists() or readme.read_text() != text:
            _atomic_write_text(readme, text)


def acquire_output_lock(out: Path):
    """Exclusive, non-blocking flock on ``<out>.lock`` (e.g. ``greedy.jsonl.lock``; never matches ``*.jsonl``).
    Returns the open lock file; keep it open for as long as ``out`` is being generated. Exits with an error naming
    ``out`` if another process already holds it (the same model/config started in two queues)."""
    out = Path(out)
    lock_path = out.with_name(out.name + ".lock")
    f = open(lock_path, "a")
    try:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        f.close()
        raise SystemExit(f"{out} is already being generated by another process (lock {lock_path} is held); "
                         "refusing to start a second writer. Run each model in only one queue.") from None
    return f


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="HumanEval regeneration runner (2026-11)")
    p.add_argument("--model", required=True)
    p.add_argument("--configs", default=None,
                   help="comma-separated config names (default: all for the model); must share one backend")
    p.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    p.add_argument("--max-model-len", type=int, default=None,
                   help="vLLM context; default = longest prompt + max-new-tokens, rounded up to 64")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    p.add_argument("--max-problems", type=int, default=None, help="first N tasks only (smoke tests)")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--prompt-variant", choices=PROMPT_VARIANTS, default=DEFAULT_PROMPT_VARIANT,
                   help="unstripped (default): raw prompt ending in a newline (BigCode humaneval-unstripped); "
                        "stripped: prompt.strip() (BigCode humaneval), base models only; the 2026-11-07 protocol "
                        "(PROMPT_VARIANT=stripped in run_regen_humaneval_8models.sh)")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    _check_out_root(args.out_root)
    spec = get_model(args.model)
    names, backend = resolve_configs(args.model, args.configs)
    check_configs_for_root(names, args.out_root)
    check_model_for_root(args.model, args.out_root)
    variant = args.prompt_variant
    try:
        apply_prompt_variant("", spec.prompt_style, variant)   # refuse e.g. stripped + chat before loading anything
    except ValueError as e:
        raise SystemExit(str(e)) from None
    # One writer per output file, held for the whole process, taken before loading anything: a second queue
    # started on the same model/config would otherwise append duplicate task_ids to the same JSONL.
    output_locks = []
    for name in names:
        out = output_path(args.out_root, args.model, name, variant)
        out.parent.mkdir(parents=True, exist_ok=True)
        output_locks.append(acquire_output_lock(out))

    import torch
    from datasets import load_dataset
    from tqdm import tqdm

    torch.manual_seed(args.seed)
    all_tasks = list(load_dataset(*DATASET[:1], split=DATASET[1]))
    digest = dataset_digest(all_tasks)
    tokenizer = load_tokenizer(args.model)
    # Prompts and the context length come from ALL 164 tasks, so a smoke/fit slice is sized like the full run.
    prompt_ids = {t["task_id"]: build_prompt_ids(tokenizer, apply_prompt_variant(t["prompt"], spec.prompt_style, variant),
                                                 spec.prompt_style) for t in all_tasks}
    if tokenizer.bos_token_id is not None:
        bad = [tid for tid, ids in prompt_ids.items() if count_leading_bos(ids, tokenizer) != 1]
        if bad:
            raise SystemExit(f"prompts without exactly one BOS: {bad[:5]}")
    longest = max(len(ids) for ids in prompt_ids.values())
    max_model_len = context_length([len(ids) for ids in prompt_ids.values()], args.max_new_tokens, args.max_model_len)
    for tid, ids in prompt_ids.items():
        check_prompt_fits(len(ids), args.max_new_tokens, max_model_len, tid)
    tasks = all_tasks[: args.max_problems] if args.max_problems else all_tasks

    if backend == "vllm":
        from bench.generator_vllm import load_engine
        backend_obj = load_engine(args.model, max_model_len=max_model_len,
                                  gpu_memory_utilization=args.gpu_memory_utilization, seed=args.seed)
    else:
        from bench.generator import load_model_and_tokenizer
        backend_obj, _ = load_model_and_tokenizer(args.model)
        offloaded = {k: v for k, v in getattr(backend_obj, "hf_device_map", {}).items() if str(v) in ("cpu", "disk")}
        if offloaded:
            raise SystemExit(f"HF model does not fit on the GPU (offloaded: {list(offloaded)[:3]}...)")

    versions = collect_versions()
    for name in names:
        config = CONFIGS[name]
        out = output_path(args.out_root, args.model, name, variant)
        expected = {"model": args.model, "config": name, "seed": args.seed, "n": config.n,
                    "max_new_tokens": args.max_new_tokens, "temperature": config.temperature, "top_p": config.top_p,
                    "prompt_style": spec.prompt_style, "prompt_variant": variant, "versions": versions}
        done = completed_task_ids(out, args.resume, expected=expected)
        out.parent.mkdir(parents=True, exist_ok=True)
        write_manifest(args.out_root, manifest_entry(
            model=args.model, config_name=name, config=config, out=out, style=spec.prompt_style,
            prompt_variant=variant, seed=args.seed, max_new_tokens=args.max_new_tokens, max_model_len=max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization, num_tasks=len(tasks), longest=longest,
            digest=digest, versions=versions))
        todo = [t for t in tasks if t["task_id"] not in done]
        print(f"[{args.model} | {name} | prompt {variant}] {len(todo)}/{len(tasks)} tasks to run -> {out}", flush=True)
        with open(out, "a") as f:
            for task in tqdm(todo, desc=name):
                raw, metas, backend_stops = generate(config, backend_obj, tokenizer, prompt_ids[task["task_id"]],
                                                     style=spec.prompt_style, seed=args.seed,
                                                     max_new_tokens=args.max_new_tokens)
                rec = make_record(task, model_id=args.model, style=spec.prompt_style, config=config,
                                  config_name=name, prompt_ids=prompt_ids[task["task_id"]], raw=raw,
                                  seed=args.seed, max_new_tokens=args.max_new_tokens, versions=versions,
                                  metas=metas, backend_stops=backend_stops, prompt_variant=variant,
                                  count_tokens=lambda t: len(tokenizer(t, add_special_tokens=False)["input_ids"]))
                f.write(json.dumps(rec) + "\n")
                f.flush()


if __name__ == "__main__":
    main(sys.argv[1:])
