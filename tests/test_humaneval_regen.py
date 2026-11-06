"""Tests for the 2026-11 HumanEval regeneration runner (bench/humaneval/regen.py).

Spec: docs/experiments/humaneval_regen_2026_11_plan.md §3-§6. CPU only: generation backends are faked;
tokenizer tests download tokenizer files only and skip when offline.
"""
import json

import pytest

from bench.humaneval import regen
from bench.humaneval.prompts import BIGCODE_HUMANEVAL_STOP_SEQUENCES, format_prompt_instruct


# ---------- models and configs ----------

def test_models_are_the_papers_eight_small_first():
    ids = [m.model_id for m in regen.MODELS]
    assert ids == [
        "m-a-p/OpenCodeInterpreter-DS-1.3B",
        "Qwen/Qwen2.5-Coder-1.5B",
        "Qwen/Qwen2.5-Coder-3B",
        "codellama/CodeLlama-7b-hf",
        "codellama/CodeLlama-7b-Instruct-hf",
        "meta-llama/Llama-2-7b-hf",
        "meta-llama/Llama-2-7b-chat-hf",
        "Qwen/Qwen2.5-Coder-7B-Instruct",
    ]


def test_prompt_styles():
    style = {m.model_id: m.prompt_style for m in regen.MODELS}
    assert style["m-a-p/OpenCodeInterpreter-DS-1.3B"] == "base"  # completion prompt, as for MBPP
    assert [m for m, s in style.items() if s == "chat"] == [
        "codellama/CodeLlama-7b-Instruct-hf",
        "meta-llama/Llama-2-7b-chat-hf",
        "Qwen/Qwen2.5-Coder-7B-Instruct",
    ]


def test_unknown_model_is_rejected():
    with pytest.raises(KeyError):
        regen.get_model("some/other-model")


def test_config_table():
    c = regen.CONFIGS
    assert set(c) == {"greedy", "beam4", "temp_t0.7", "top_p0.9_t1.0", "pless_t0.7", "pless_norm_t0.7", "top_p0.95_t0.2"}
    assert (c["greedy"].backend, c["greedy"].n, c["greedy"].temperature) == ("vllm", 1, 0.0)
    assert (c["beam4"].backend, c["beam4"].n, c["beam4"].num_beams) == ("hf", 1, 4)
    assert (c["temp_t0.7"].n, c["temp_t0.7"].temperature, c["temp_t0.7"].top_p) == (10, 0.7, 1.0)
    assert (c["top_p0.9_t1.0"].temperature, c["top_p0.9_t1.0"].top_p) == (1.0, 0.9)
    assert (c["pless_t0.7"].sampler, c["pless_t0.7"].temperature, c["pless_t0.7"].n) == ("pless", 0.7, 10)
    assert c["pless_norm_t0.7"].sampler == "pless_norm"
    assert (c["top_p0.95_t0.2"].temperature, c["top_p0.95_t0.2"].top_p) == (0.2, 0.95)


def test_hasan_config_only_for_the_three_shared_models():
    with_hasan = [m.model_id for m in regen.MODELS if "top_p0.95_t0.2" in regen.configs_for(m.model_id)]
    assert with_hasan == ["m-a-p/OpenCodeInterpreter-DS-1.3B", "Qwen/Qwen2.5-Coder-1.5B", "Qwen/Qwen2.5-Coder-3B"]
    assert len(regen.configs_for("codellama/CodeLlama-7b-hf")) == 6


# ---------- generation dispatch (faked backends) ----------

class _Calls:
    def __init__(self):
        self.calls = []

    META = {"finish_reason": "stop", "stop_reason": "\ndef", "num_output_tokens": 7}

    def std(self, engine, tokenizer, prompt_text, **kw):
        self.calls.append(("standard", prompt_text, kw))
        return ["X"] * kw["n_samples"], [self.META] * kw["n_samples"]

    def pless(self, engine, tokenizer, prompt_text, sampler_name, **kw):
        self.calls.append(("pless", prompt_text, dict(kw, sampler_name=sampler_name)))
        return ["X"] * kw["n_samples"], [self.META] * kw["n_samples"]

    def beam(self, model, tokenizer, prompt_text, num_beams, max_new_tokens, stop_strings=None, return_meta=False):
        self.calls.append(("beam", prompt_text, dict(num_beams=num_beams, max_new_tokens=max_new_tokens,
                                                     stop_strings=stop_strings, return_meta=return_meta)))
        return ["X"], [self.META]


@pytest.fixture
def fake(monkeypatch):
    c = _Calls()
    monkeypatch.setattr(regen, "generate_samples_standard_vllm", c.std)
    monkeypatch.setattr(regen, "generate_samples_vllm", c.pless)
    monkeypatch.setattr(regen, "generate_samples_beam", c.beam)
    return c


@pytest.mark.parametrize("name,kind", [
    ("greedy", "standard"), ("temp_t0.7", "standard"), ("top_p0.9_t1.0", "standard"),
    ("top_p0.95_t0.2", "standard"), ("pless_t0.7", "pless"), ("pless_norm_t0.7", "pless"), ("beam4", "beam"),
])
def test_dispatch_routes_to_the_right_backend(fake, name, kind):
    texts, metas, backend_stops = regen.generate(regen.CONFIGS[name], "backend", "tok", [1, 2, 3], style="base",
                                                 seed=42, max_new_tokens=512)
    assert len(texts) == len(metas) == regen.CONFIGS[name].n
    (got_kind, ids, kw), = fake.calls
    assert got_kind == kind
    assert ids == [1, 2, 3]  # token ids, never a string (single-BOS guarantee)
    assert kw["return_meta"] is True and backend_stops is True


def test_standard_params_are_native_sampling_params(fake):
    regen.generate(regen.CONFIGS["greedy"], "e", "t", [1], style="base", seed=42, max_new_tokens=512)
    regen.generate(regen.CONFIGS["top_p0.9_t1.0"], "e", "t", [1], style="base", seed=42, max_new_tokens=512)
    g, p = fake.calls[0][2], fake.calls[1][2]
    assert (g["n_samples"], g["temperature"], g["seed"], g["max_new_tokens"]) == (1, 0.0, 42, 512)
    assert (p["n_samples"], p["temperature"], p["top_p"], p["top_k"], p["seed"]) == (10, 1.0, 0.9, 0, 42)


def test_pless_params(fake):
    regen.generate(regen.CONFIGS["pless_t0.7"], "e", "t", [1], style="chat", seed=42, max_new_tokens=512)
    kw = fake.calls[0][2]
    assert (kw["sampler_name"], kw["temperature"], kw["n_samples"], kw["seed"]) == ("pless", 0.7, 10, 42)


def test_stop_sequences_base_only(fake):
    regen.generate(regen.CONFIGS["temp_t0.7"], "e", "t", [1], style="base", seed=1, max_new_tokens=8)
    regen.generate(regen.CONFIGS["temp_t0.7"], "e", "t", [1], style="chat", seed=1, max_new_tokens=8)
    regen.generate(regen.CONFIGS["beam4"], "m", "t", [1], style="base", seed=1, max_new_tokens=8)
    assert fake.calls[0][2]["stop_strings"] == BIGCODE_HUMANEVAL_STOP_SEQUENCES
    assert fake.calls[1][2]["stop_strings"] is None
    assert fake.calls[2][2]["stop_strings"] == BIGCODE_HUMANEVAL_STOP_SEQUENCES


def test_broken_vllm_detokenizer_disables_backend_stops(fake):
    # OCI / DeepSeek-Coder: vLLM's own detokenizer mangles whitespace (engine._safe_tokenizer is set), so
    # stop strings matched on its text could fire wrongly or never; generate without them and cut afterwards.
    class Engine:
        _safe_tokenizer = object()
    texts, metas, backend_stops = regen.generate(regen.CONFIGS["temp_t0.7"], Engine(), "t", [1], style="base",
                                                 seed=1, max_new_tokens=8)
    assert fake.calls[0][2]["stop_strings"] is None and backend_stops is False


def test_postprocess_cuts_base_but_not_chat():
    raw = "    return 1\n\n\ndef check(candidate):\n    pass\n"
    assert regen.postprocess(raw, "base") == "    return 1\n\n"
    assert regen.postprocess(raw, "chat") == raw


# ---------- prompt encoding ----------

def _tok(name):
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.AutoTokenizer.from_pretrained(name)
    except Exception as e:
        pytest.skip(f"tokenizer unavailable: {e}")


TASK_PROMPT = 'from typing import List\n\n\ndef f(xs: List[int]) -> int:\n    """ Sum. """\n'


def test_codellama_instruct_chat_prompt_has_exactly_one_bos():
    tok = _tok("codellama/CodeLlama-7b-Instruct-hf")
    ids = regen.build_prompt_ids(tok, TASK_PROMPT, "chat")
    assert isinstance(ids, list) and all(isinstance(i, int) for i in ids)
    assert ids[0] == tok.bos_token_id and ids[1] != tok.bos_token_id
    assert regen.count_leading_bos(ids, tok) == 1


def test_codellama_base_prompt_has_exactly_one_bos():
    tok = _tok("codellama/CodeLlama-7b-hf")
    ids = regen.build_prompt_ids(tok, TASK_PROMPT, "base")
    assert regen.count_leading_bos(ids, tok) == 1
    assert tok.decode(ids, skip_special_tokens=True) == TASK_PROMPT


def test_chat_template_text_matches_the_old_runner():
    # Same messages as bench/humaneval/prompts.py:format_prompt_instruct (wording unchanged).
    tok = _tok("Qwen/Qwen2.5-Coder-7B-Instruct")
    ids = regen.build_prompt_ids(tok, TASK_PROMPT, "chat")
    old_text, _ = format_prompt_instruct({"prompt": TASK_PROMPT}, tok)
    assert tok.decode(ids) == old_text


def test_context_length_uses_all_tasks_even_when_sliced():
    # The 5-task fit test must size max_model_len like the full run (code review HIGH #3).
    assert regen.context_length([10, 900, 5], max_new_tokens=512) == 1472   # ceil((900+512)/64)*64
    assert regen.context_length([10, 900, 5], max_new_tokens=512, override=2048) == 2048


def test_prompt_length_check():
    with pytest.raises(ValueError):
        regen.check_prompt_fits(prompt_len=600, max_new_tokens=512, max_model_len=1024, task_id="t")
    regen.check_prompt_fits(prompt_len=500, max_new_tokens=512, max_model_len=1024, task_id="t")


# ---------- records, paths, resume ----------

def test_record_fields():
    task = {"task_id": "HumanEval/0", "prompt": TASK_PROMPT, "test": "T", "entry_point": "f"}
    rec = regen.make_record(task, model_id="m/x", style="base", config=regen.CONFIGS["temp_t0.7"],
                            config_name="temp_t0.7", prompt_ids=[1, 2], raw=["    return 1\n\ndef g"],
                            seed=42, max_new_tokens=512, versions={"vllm": "0.33.0"},
                            metas=[{"finish_reason": "length", "stop_reason": None, "num_output_tokens": 512}],
                            backend_stops=False)
    for k in ("task_id", "model", "config", "backend", "method", "temperature", "top_p", "prompt_style",
              "task_prompt", "prompt_token_ids_head", "num_prompt_tokens", "samples", "raw_samples", "test",
              "entry_point", "seed", "n", "max_new_tokens", "stop", "versions", "timestamp", "sample_meta",
              "backend_stop_strings"):
        assert k in rec, k
    assert rec["samples"] == ["    return 1\n"]
    assert rec["task_prompt"] == TASK_PROMPT
    assert rec["stop"] == BIGCODE_HUMANEVAL_STOP_SEQUENCES


def test_output_path_is_under_the_new_root(tmp_path):
    p = regen.output_path(tmp_path, "codellama/CodeLlama-7b-hf", "pless_t0.7")
    assert p == tmp_path / "codellama--CodeLlama-7b-hf" / "pless_t0.7.jsonl"


def test_default_root_is_the_dedicated_folder():
    assert str(regen.DEFAULT_OUT_ROOT) == "results/humaneval_regen_2026_11"


def test_existing_file_without_resume_is_refused(tmp_path):
    p = tmp_path / "x.jsonl"
    p.write_text(json.dumps({"task_id": "HumanEval/0"}) + "\n")
    with pytest.raises(FileExistsError):
        regen.completed_task_ids(p, resume=False)
    assert regen.completed_task_ids(p, resume=True) == {"HumanEval/0"}
    assert regen.completed_task_ids(tmp_path / "missing.jsonl", resume=False) == set()


def test_resume_refuses_a_truncated_last_line(tmp_path):
    p = tmp_path / "x.jsonl"
    p.write_text(json.dumps({"task_id": "HumanEval/0"}) + "\n" + '{"task_id": "HumanEval/1", "sam')
    with pytest.raises(ValueError, match="truncated"):
        regen.completed_task_ids(p, resume=True)


def test_record_counts_completion_tokens_after_the_cut():
    task = {"task_id": "HumanEval/0", "prompt": TASK_PROMPT, "test": "T", "entry_point": "f"}
    rec = regen.make_record(task, model_id="m/x", style="base", config=regen.CONFIGS["greedy"], config_name="greedy",
                            prompt_ids=[1], raw=["    return 1\n\ndef g():\n    pass\n"], seed=42,
                            max_new_tokens=512, versions={}, metas=[{"num_output_tokens": 30}], backend_stops=False,
                            count_tokens=lambda text: len(text.split()))
    assert rec["completion_num_tokens"] == [2]        # "return", "1" (the cut text)


def test_resume_refuses_a_file_written_with_different_settings(tmp_path):
    p = tmp_path / "x.jsonl"
    p.write_text(json.dumps({"task_id": "HumanEval/0", "model": "m/x", "config": "temp_t0.7", "seed": 42, "n": 10,
                             "max_new_tokens": 512, "temperature": 0.7, "top_p": 1.0, "prompt_style": "base",
                             "versions": {"vllm": "0.33.0", "transformers": "5.20.2", "torch": "2.14.0"}}) + "\n")
    same = {"model": "m/x", "config": "temp_t0.7", "seed": 42, "n": 10, "max_new_tokens": 512, "temperature": 0.7,
            "top_p": 1.0, "prompt_style": "base", "versions": {"vllm": "0.33.0", "transformers": "5.20.2", "torch": "2.14.0"}}
    assert regen.completed_task_ids(p, resume=True, expected=same) == {"HumanEval/0"}
    with pytest.raises(ValueError):
        regen.completed_task_ids(p, resume=True, expected=dict(same, seed=7))
    with pytest.raises(ValueError):
        regen.completed_task_ids(p, resume=True, expected=dict(same, versions=dict(same["versions"], vllm="0.32.0")))
    with pytest.raises(ValueError):   # vLLM runtime switches are part of the settings
        regen.completed_task_ids(p, resume=True, expected=dict(same, versions=dict(same["versions"],
                                                                                 env={"VLLM_USE_V2_MODEL_RUNNER": "1"})))


def test_vllm_completion_meta_is_extracted():
    from types import SimpleNamespace
    from bench.generator_vllm import _extract_completion_meta
    out = SimpleNamespace(outputs=[SimpleNamespace(finish_reason="stop", stop_reason="\ndef", token_ids=[1, 2, 3]),
                                   SimpleNamespace(finish_reason="length", stop_reason=None, token_ids=[4] * 512)])
    assert _extract_completion_meta(out) == [
        {"finish_reason": "stop", "stop_reason": "\ndef", "num_output_tokens": 3},
        {"finish_reason": "length", "stop_reason": None, "num_output_tokens": 512},
    ]


def test_vllm_text_is_audited_against_the_reference_tokenizer():
    # vLLM's own text must equal decoding prompt+completion with the HF tokenizer minus the decoded prompt
    # (so leading spaces of SentencePiece pieces are judged in context). A mismatch flags unreliable stops.
    from types import SimpleNamespace
    from bench.generator_vllm import _extract_completion_meta
    tok = _tok("codellama/CodeLlama-7b-hf")
    prompt = tok("def f():\n")["input_ids"]
    gen = tok("def f():\n    return 1\n", add_special_tokens=False)["input_ids"][len(prompt) - 1:]
    good = tok.decode(prompt + gen, skip_special_tokens=True)[len(tok.decode(prompt, skip_special_tokens=True)):]
    out = SimpleNamespace(prompt_token_ids=prompt, outputs=[
        SimpleNamespace(finish_reason="stop", stop_reason=None, token_ids=gen, text=good),
        SimpleNamespace(finish_reason="stop", stop_reason=None, token_ids=gen, text=good.replace("    ", "")),
    ])
    metas = _extract_completion_meta(out, reference_tokenizer=tok)
    assert [m["text_matches_reference"] for m in metas] == [True, False]


def test_text_audit_accounts_for_the_removed_stop_string():
    # vLLM drops the matched stop string from .text (include_stop_str_in_output=False) but keeps its tokens.
    from types import SimpleNamespace
    from bench.generator_vllm import _extract_completion_meta
    tok = _tok("codellama/CodeLlama-7b-hf")
    prompt = tok("def f():\n")["input_ids"]
    full = tok("def f():\n    return 1\n\ndef g():", add_special_tokens=False)["input_ids"]
    gen = full[len(prompt) - 1:]
    ref = tok.decode(prompt + gen, skip_special_tokens=True)[len(tok.decode(prompt, skip_special_tokens=True)):]
    text = ref[: ref.index("\ndef")]
    out = SimpleNamespace(prompt_token_ids=prompt, outputs=[
        SimpleNamespace(finish_reason="stop", stop_reason="\ndef", token_ids=gen, text=text),
        SimpleNamespace(finish_reason="stop", stop_reason="\ndef", token_ids=gen, text=text.replace("    ", "")),
    ])
    assert [m["text_matches_reference"] for m in _extract_completion_meta(out, reference_tokenizer=tok)] == [True, False]


def test_hf_beam_meta(monkeypatch):
    torch = pytest.importorskip("torch")
    tok = _tok("Qwen/Qwen2.5-Coder-1.5B")
    from bench import generator as gen
    prompt_ids = tok("def f():\n")["input_ids"]
    new = tok("    return 1\n", add_special_tokens=False)["input_ids"]

    class FakeModel:
        device = "cpu"
        def generate(self, input_ids, **kw):
            assert kw["num_beams"] == 4 and kw["do_sample"] is False
            return torch.cat([input_ids, torch.tensor([new])], dim=1)

    texts, metas = gen.generate_samples_beam(FakeModel(), tok, prompt_ids, num_beams=4, max_new_tokens=512,
                                             stop_strings=None, return_meta=True)
    assert texts == ["    return 1\n"]
    assert metas == [{"finish_reason": "stop", "stop_reason": None, "num_output_tokens": len(new)}]
    assert gen.generate_samples_beam(FakeModel(), tok, prompt_ids, num_beams=4, max_new_tokens=512) == ["    return 1\n"]


def test_runner_does_not_import_the_eval_package():
    # The GPU box's slim generation env (envs/vllm-033) lacks bench.eval's deps (zss, codebleu);
    # importing bench.eval.* there crashes. Generation must not depend on it.
    import subprocess
    import sys
    code = "import sys, bench.humaneval.regen; print(any(m == 'bench.eval' or m.startswith('bench.eval.') for m in sys.modules))"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.strip()
    assert out == "False"


def test_truncate_at_stop_lives_with_the_prompts():
    from bench.humaneval.prompts import truncate_at_stop
    assert truncate_at_stop("    return 1\n\ndef g():\n") == "    return 1\n"
