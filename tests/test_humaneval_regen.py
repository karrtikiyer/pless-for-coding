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
        "Qwen/Qwen2.5-Coder-7B",   # base model (user decision 2026-11-07), replaced Qwen2.5-Coder-7B-Instruct
    ]


def test_prompt_styles():
    style = {m.model_id: m.prompt_style for m in regen.MODELS}
    assert style["m-a-p/OpenCodeInterpreter-DS-1.3B"] == "base"  # completion prompt, as for MBPP
    assert [m for m, s in style.items() if s == "chat"] == [
        "codellama/CodeLlama-7b-Instruct-hf",
        "meta-llama/Llama-2-7b-chat-hf",
    ]
    assert style["Qwen/Qwen2.5-Coder-7B"] == "base"


def test_unknown_model_is_rejected():
    with pytest.raises(KeyError):
        regen.get_model("some/other-model")


def test_config_table():
    c = regen.CONFIGS
    assert set(c) == {"greedy", "beam4", "temp_t0.7", "top_p0.9_t1.0", "pless_t0.7", "pless_norm_t0.7", "top_p0.95_t0.2",
                      "greedy_hf"}
    assert (c["greedy"].backend, c["greedy"].n, c["greedy"].temperature) == ("vllm", 1, 0.0)
    g = c["greedy_hf"]
    assert (g.backend, g.method, g.n, g.temperature, g.num_beams, g.sampler) == ("hf", "greedy", 1, 0.0, 1, None)
    assert (c["beam4"].backend, c["beam4"].n, c["beam4"].num_beams) == ("hf", 1, 4)
    assert (c["temp_t0.7"].n, c["temp_t0.7"].temperature, c["temp_t0.7"].top_p) == (10, 0.7, 1.0)
    assert (c["top_p0.9_t1.0"].temperature, c["top_p0.9_t1.0"].top_p) == (1.0, 0.9)
    assert (c["pless_t0.7"].sampler, c["pless_t0.7"].temperature, c["pless_t0.7"].n) == ("pless", 0.7, 10)
    assert c["pless_norm_t0.7"].sampler == "pless_norm"
    assert (c["top_p0.95_t0.2"].temperature, c["top_p0.95_t0.2"].top_p) == (0.2, 0.95)


def test_hasan_config_only_for_the_three_shared_models():
    with_hasan = [m.model_id for m in regen.MODELS if "top_p0.95_t0.2" in regen.configs_for(m.model_id)]
    assert with_hasan == ["m-a-p/OpenCodeInterpreter-DS-1.3B", "Qwen/Qwen2.5-Coder-1.5B", "Qwen/Qwen2.5-Coder-3B",
                          "Qwen/Qwen2.5-Coder-7B"]
    assert len(regen.configs_for("codellama/CodeLlama-7b-hf")) == 6
    assert regen.configs_for("Qwen/Qwen2.5-Coder-7B") == regen.COMMON_CONFIGS + ["top_p0.95_t0.2"]


def test_greedy_hf_is_a_diagnostic_config_outside_the_default_sets():
    # Never run by the `small` / `large` modes (which use configs_for), but accepted via --configs.
    for m in regen.MODELS:
        assert "greedy_hf" not in regen.configs_for(m.model_id)
        assert "greedy_hf" in regen.allowed_configs(m.model_id)
    assert regen.resolve_configs("Qwen/Qwen2.5-Coder-1.5B", "greedy_hf") == (["greedy_hf"], "hf")
    assert regen.resolve_configs("Qwen/Qwen2.5-Coder-1.5B", "beam4,greedy_hf") == (["beam4", "greedy_hf"], "hf")
    with pytest.raises(SystemExit, match="mix backends"):   # unchanged: the default set mixes vLLM and HF
        regen.resolve_configs("codellama/CodeLlama-7b-hf", None)
    with pytest.raises(SystemExit, match="mix backends"):
        regen.resolve_configs("Qwen/Qwen2.5-Coder-1.5B", "greedy,greedy_hf")
    with pytest.raises(SystemExit, match="not in the plan"):
        regen.resolve_configs("codellama/CodeLlama-7b-hf", "top_p0.95_t0.2")


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
    ("greedy_hf", "beam"),
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


# ---------- diagnostics: HF greedy (backend effect) ----------

def test_greedy_hf_passes_num_beams_1_and_the_base_stops(fake):
    regen.generate(regen.CONFIGS["greedy_hf"], "m", "t", [1, 2], style="base", seed=42, max_new_tokens=512)
    (kind, ids, kw), = fake.calls
    assert (kind, ids, kw["num_beams"], kw["max_new_tokens"]) == ("beam", [1, 2], 1, 512)
    assert kw["stop_strings"] == BIGCODE_HUMANEVAL_STOP_SEQUENCES


def test_hf_num_beams_1_is_plain_greedy():
    # generate_samples_beam(num_beams=1) (do_sample=False) must equal a hand-written argmax loop.
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    tok = _tok("Qwen/Qwen2.5-Coder-1.5B")
    from bench import generator as gen
    torch.manual_seed(0)
    cfg = transformers.LlamaConfig(vocab_size=len(tok), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                   num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=128,
                                   bos_token_id=tok.bos_token_id, eos_token_id=tok.eos_token_id)
    model = transformers.LlamaForCausalLM(cfg).eval()
    prompt_ids = tok("def f():\n")["input_ids"]
    ids = torch.tensor([prompt_ids])
    with torch.no_grad():
        for _ in range(12):
            nxt = model(ids).logits[0, -1].argmax()
            ids = torch.cat([ids, nxt.view(1, 1)], dim=1)
            if int(nxt) == tok.eos_token_id:
                break
    expected = tok.decode(ids[0], skip_special_tokens=True)[len(tok.decode(prompt_ids, skip_special_tokens=True)):]
    texts, metas = gen.generate_samples_beam(model, tok, prompt_ids, num_beams=1, max_new_tokens=12,
                                             stop_strings=None, return_meta=True)
    assert texts == [expected]
    assert metas[0]["num_output_tokens"] == ids.shape[1] - len(prompt_ids)


# ---------- diagnostics: prompt variant (prompt effect) ----------

HE0_PROMPT = ('from typing import List\n\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n'
              '    """ Check if any two numbers are closer than threshold.\n'
              '    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    """\n')
HE0_TEST = ("\n\nMETADATA = {}\n\n\ndef check(candidate):\n"
            "    assert candidate([1.0, 2.0, 3.9, 4.0], 0.3) == True\n"
            "    assert candidate([1.0, 2.0, 3.0], 0.5) == False\n")
HE0_BODY = ("    for i, a in enumerate(numbers):\n        for j, b in enumerate(numbers):\n"
            "            if i != j and abs(a - b) < threshold:\n                return True\n    return False\n")


def test_prompt_variant_unstripped_is_the_raw_prompt_unchanged():
    assert regen.apply_prompt_variant(HE0_PROMPT, "base", "unstripped") is HE0_PROMPT
    assert regen.apply_prompt_variant(HE0_PROMPT, "chat", "unstripped") is HE0_PROMPT
    assert regen.DEFAULT_PROMPT_VARIANT == "unstripped"


def test_prompt_variant_stripped_is_the_harness_humaneval_task():
    s = regen.apply_prompt_variant("\n" + HE0_PROMPT, "base", "stripped")
    assert s == HE0_PROMPT.strip() and s.endswith('    """') and not s.startswith("\n")


def test_prompt_variant_stripped_is_rejected_for_chat_models():
    with pytest.raises(ValueError, match="base"):
        regen.apply_prompt_variant(HE0_PROMPT, "chat", "stripped")
    with pytest.raises(ValueError):
        regen.apply_prompt_variant(HE0_PROMPT, "base", "bogus")


def test_cli_prompt_variant_default_and_chat_rejection():
    assert regen.parse_args(["--model", "Qwen/Qwen2.5-Coder-1.5B"]).prompt_variant == "unstripped"
    assert regen.parse_args(["--model", "x", "--prompt-variant", "stripped"]).prompt_variant == "stripped"
    with pytest.raises(SystemExit):
        regen.parse_args(["--model", "x", "--prompt-variant", "bogus"])
    with pytest.raises(SystemExit, match="base"):   # refused before any model / dataset is loaded
        regen.main(["--model", "meta-llama/Llama-2-7b-chat-hf", "--configs", "greedy", "--prompt-variant", "stripped",
                    "--out-root", "results/humaneval_regen_2026_11_never_written"])


def _he0_record(variant="unstripped", raw=None):
    task = {"task_id": "HumanEval/0", "prompt": HE0_PROMPT, "test": HE0_TEST, "entry_point": "has_close_elements"}
    kw = {} if variant is None else {"prompt_variant": variant}
    return regen.make_record(task, model_id="Qwen/Qwen2.5-Coder-1.5B", style="base", config=regen.CONFIGS["greedy"],
                             config_name="greedy", prompt_ids=[1], raw=raw or ["\n" + HE0_BODY], seed=42,
                             max_new_tokens=512, versions={}, **kw)


def test_record_stores_the_prompt_variant_and_the_prompt_actually_used():
    default = _he0_record(variant=None)
    assert default["prompt_variant"] == "unstripped" and default["task_prompt"] == HE0_PROMPT
    stripped = _he0_record("stripped")
    assert stripped["prompt_variant"] == "stripped" and stripped["task_prompt"] == HE0_PROMPT.strip()
    assert stripped["stop"] == default["stop"] == BIGCODE_HUMANEVAL_STOP_SEQUENCES   # stop list unchanged


def test_stripped_record_composes_a_passing_official_program():
    # The official grader composes task_prompt + completion + "\n" + test + "\n" + check(entry_point);
    # with the stripped prompt the completion starts with "\n    ".
    from bench.eval.humaneval_official import build_program, run_program
    rec = _he0_record("stripped")
    program, mode = build_program(rec, rec["samples"][0])
    assert mode == "base"
    assert program == (HE0_PROMPT.strip() + "\n" + HE0_BODY + "\n" + HE0_TEST + "\n" + "check(has_close_elements)")
    compile(program, "<he0>", "exec")
    assert run_program(program) == (True, "passed")
    wrong = _he0_record("stripped", raw=["\n    return True\n"])
    assert run_program(build_program(wrong, wrong["samples"][0])[0])[0] is False


def test_stripped_completion_starting_with_newline_survives_the_post_cut():
    rec = _he0_record("stripped", raw=["\n" + HE0_BODY + "\n\ndef check(x):\n    pass\n"])
    assert rec["samples"] == ["\n" + HE0_BODY + "\n"]


def test_output_path_distinguishes_the_prompt_variant(tmp_path):
    m = "Qwen/Qwen2.5-Coder-1.5B"
    default = regen.output_path(tmp_path, m, "greedy")
    assert default == regen.output_path(tmp_path, m, "greedy", "unstripped") == tmp_path / "Qwen--Qwen2.5-Coder-1.5B" / "greedy.jsonl"
    stripped = regen.output_path(tmp_path, m, "greedy", "stripped")
    assert stripped == tmp_path / "Qwen--Qwen2.5-Coder-1.5B" / "greedy__prompt_stripped.jsonl"
    assert stripped != default
    with pytest.raises(ValueError):
        regen.output_path(tmp_path, m, "greedy", "bogus")


def test_resume_checks_the_prompt_variant_and_reads_old_files_as_unstripped(tmp_path):
    assert "prompt_variant" in regen.RESUME_KEYS
    base = {"task_id": "HumanEval/0", "model": "m/x", "config": "greedy", "seed": 42, "n": 1, "max_new_tokens": 512,
            "temperature": 0.0, "top_p": 1.0, "prompt_style": "base", "versions": {}}
    expected = {k: v for k, v in base.items() if k != "task_id"}
    old = tmp_path / "old.jsonl"     # written before prompt_variant existed
    old.write_text(json.dumps(base) + "\n")
    assert regen.completed_task_ids(old, True, expected=dict(expected, prompt_variant="unstripped")) == {"HumanEval/0"}
    with pytest.raises(ValueError, match="prompt_variant"):
        regen.completed_task_ids(old, True, expected=dict(expected, prompt_variant="stripped"))
    new = tmp_path / "new.jsonl"
    new.write_text(json.dumps(dict(base, prompt_variant="stripped")) + "\n")
    assert regen.completed_task_ids(new, True, expected=dict(expected, prompt_variant="stripped")) == {"HumanEval/0"}
    with pytest.raises(ValueError, match="prompt_variant"):
        regen.completed_task_ids(new, True, expected=dict(expected, prompt_variant="unstripped"))


def test_manifest_entry_records_the_prompt_variant(tmp_path):
    entry = regen.manifest_entry(model="m/x", config_name="greedy", config=regen.CONFIGS["greedy"],
                                 out=tmp_path / "f.jsonl", style="base", prompt_variant="stripped", seed=42,
                                 max_new_tokens=512, max_model_len=1024, gpu_memory_utilization=0.9, num_tasks=164,
                                 longest=300, digest="abc", versions={})
    assert entry["prompt_variant"] == "stripped" and entry["config"] == "greedy"
    regen.write_manifest(tmp_path, entry)
    assert json.loads((tmp_path / "MANIFEST.json").read_text())["runs"][0]["prompt_variant"] == "stripped"


def test_stripped_prompt_still_has_one_bos():
    tok = _tok("codellama/CodeLlama-7b-hf")
    ids = regen.build_prompt_ids(tok, regen.apply_prompt_variant(HE0_PROMPT, "base", "stripped"), "base")
    assert regen.count_leading_bos(ids, tok) == 1
    assert tok.decode(ids, skip_special_tokens=True) == HE0_PROMPT.strip()


# ---------- orchestration: diag mode ----------

def _diag_block():
    from pathlib import Path
    script = (Path(__file__).resolve().parents[1] / "run_regen_humaneval_8models.sh").read_text()
    lines = script.splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.strip().startswith("diag)"))
    end = next(i for i in range(start, len(lines)) if lines[i].strip().endswith(";;"))
    return "\n".join(lines[start:end + 1])


def test_diag_mode_runs_hf_greedy_and_stripped_vllm_greedy_on_the_small_models():
    block = _diag_block()
    assert 'for m in "${SMALL[@]}"' in block
    assert "results/humaneval_regen_2026_11_diag" in block
    runs = [ln.strip() for ln in block.splitlines() if ln.strip().startswith("run ")]
    assert len(runs) == 2
    hf, vllm = runs
    assert '"greedy_hf"' in hf and "--prompt-variant" not in hf
    assert '"greedy"' in vllm and "--prompt-variant stripped" in vllm
    assert "--max-problems" not in block   # full 164 tasks


def test_diag_outputs_do_not_collide(tmp_path):
    m = "Qwen/Qwen2.5-Coder-1.5B"
    paths = {regen.output_path(tmp_path, m, "greedy_hf"), regen.output_path(tmp_path, m, "greedy", "stripped"),
             regen.output_path(tmp_path, m, "greedy")}
    assert len(paths) == 3


# ---------- stripped-prompt protocol: per-model flag (single source of truth for the orchestration) ----------

BASE_IDS = ["m-a-p/OpenCodeInterpreter-DS-1.3B", "Qwen/Qwen2.5-Coder-1.5B", "Qwen/Qwen2.5-Coder-3B",
            "codellama/CodeLlama-7b-hf", "meta-llama/Llama-2-7b-hf", "Qwen/Qwen2.5-Coder-7B"]
LARGE_BASE_IDS = ["codellama/CodeLlama-7b-hf", "meta-llama/Llama-2-7b-hf", "Qwen/Qwen2.5-Coder-7B"]
CHAT_IDS = ["codellama/CodeLlama-7b-Instruct-hf", "meta-llama/Llama-2-7b-chat-hf"]


def test_prompt_variant_args_flag_raw_prompted_models_only():
    for m in BASE_IDS:
        assert regen.prompt_variant_args(m, "stripped") == ["--prompt-variant", "stripped"], m
        assert regen.prompt_variant_args(m, "unstripped") == [], m
    for m in CHAT_IDS:   # chat-template prompts: the runner refuses stripped, so no flag at all
        assert regen.prompt_variant_args(m, "stripped") == [], m
        assert regen.prompt_variant_args(m, "unstripped") == [], m
    with pytest.raises(ValueError):
        regen.prompt_variant_args(BASE_IDS[0], "bogus")
    with pytest.raises(KeyError):
        regen.prompt_variant_args("some/other-model", "stripped")


def test_prompt_variant_args_are_accepted_by_the_runner():
    # Whatever the helper emits for a model must pass main()'s chat refusal (checked via apply_prompt_variant).
    for m in BASE_IDS + CHAT_IDS:
        args = regen.parse_args(["--model", m] + regen.prompt_variant_args(m, "stripped"))
        regen.apply_prompt_variant("x", regen.get_model(m).prompt_style, args.prompt_variant)


# ---------- orchestration: commands actually issued (fake interpreter, nothing is generated) ----------

def _run_script(tmp_path, args, env_extra=None):
    """Run a copy of run_regen_humaneval_8models.sh with PY pointing to a fake interpreter that prints the
    runner invocations (`-m bench.humaneval.regen ...`) instead of executing them. Returns (rc, calls, stdout)."""
    import os
    import shutil
    import subprocess
    import sys
    from pathlib import Path
    repo = Path(__file__).resolve().parents[1]
    work = tmp_path / "work"
    work.mkdir(exist_ok=True)
    shutil.copy(repo / "run_regen_humaneval_8models.sh", work / "run.sh")
    fake = work / "fakepy"
    fake.write_text("#!/usr/bin/env bash\n"
                    'if [[ "$1" == "-m" ]]; then echo "CALL $*"; exit 0; fi\n'
                    'if [[ "$*" == *whoami* ]]; then exit 0; fi\n'
                    f'exec "{sys.executable}" "$@"\n')
    fake.chmod(0o755)
    env = {k: v for k, v in os.environ.items() if k not in ("PROMPT_VARIANT", "SMOKE_NAME")}
    env.update({"PY": str(fake), "PYTHONPATH": str(repo)}, **(env_extra or {}))
    p = subprocess.run(["bash", str(work / "run.sh"), *args], capture_output=True, text=True, env=env, cwd=work)
    calls = [ln[len("CALL -m bench.humaneval.regen "):] for ln in p.stdout.splitlines() if ln.startswith("CALL ")]
    return p.returncode, calls, p.stdout + p.stderr


Q15 = "Qwen/Qwen2.5-Coder-1.5B"
SMOKE_VLLM = "greedy,temp_t0.7,top_p0.9_t1.0,pless_t0.7,pless_norm_t0.7"


def test_default_smoke_commands_are_unchanged(tmp_path):
    rc, calls, out = _run_script(tmp_path, ["smoke"])
    assert rc == 0, out
    assert calls[:2] == [
        f"--model m-a-p/OpenCodeInterpreter-DS-1.3B --configs {SMOKE_VLLM} --out-root results/humaneval_regen_2026_11_smoke --max-problems 5",
        "--model m-a-p/OpenCodeInterpreter-DS-1.3B --configs beam4 --out-root results/humaneval_regen_2026_11_smoke --max-problems 5",
    ]
    assert len(calls) == 6 and not any("--prompt-variant" in c for c in calls)
    rc, calls, out = _run_script(tmp_path, ["smoke-rerun"], {"SMOKE_NAME": "smoke2"})
    assert rc == 0 and all("--out-root results/humaneval_regen_2026_11_smoke2_rerun " in c for c in calls), out


def test_default_fit_small_large_commands_are_unchanged(tmp_path):
    rc, calls, out = _run_script(tmp_path, ["fit", "meta-llama/Llama-2-7b-chat-hf"])
    assert rc == 0, out
    assert calls == [
        "--model meta-llama/Llama-2-7b-chat-hf --configs greedy --out-root results/humaneval_regen_2026_11_fit --max-problems 5",
        "--model meta-llama/Llama-2-7b-chat-hf --configs beam4 --out-root results/humaneval_regen_2026_11_fit --max-problems 5",
    ]
    rc, calls, out = _run_script(tmp_path, ["large", "codellama/CodeLlama-7b-hf"])
    assert rc == 0, out
    assert calls == [
        "--model codellama/CodeLlama-7b-hf --configs greedy,temp_t0.7,top_p0.9_t1.0,pless_t0.7,pless_norm_t0.7 "
        "--out-root results/humaneval_regen_2026_11 --resume",
        "--model codellama/CodeLlama-7b-hf --configs beam4 --out-root results/humaneval_regen_2026_11 --resume",
    ]
    rc, calls, out = _run_script(tmp_path, ["small"])
    assert rc == 0 and len(calls) == 6, out
    assert calls[2] == ("--model Qwen/Qwen2.5-Coder-1.5B --configs greedy,temp_t0.7,top_p0.9_t1.0,pless_t0.7,"
                        "pless_norm_t0.7,top_p0.95_t0.2 --out-root results/humaneval_regen_2026_11 --resume")


STRIPPED = {"PROMPT_VARIANT": "stripped"}


def test_stripped_smoke_and_rerun_flag_every_call_into_new_roots(tmp_path):
    rc, calls, out = _run_script(tmp_path, ["smoke"], STRIPPED)
    assert rc == 0, out
    assert calls[:2] == [
        f"--model m-a-p/OpenCodeInterpreter-DS-1.3B --configs {SMOKE_VLLM} --out-root "
        "results/humaneval_regen_2026_11_stripped_smoke --max-problems 5 --prompt-variant stripped",
        "--model m-a-p/OpenCodeInterpreter-DS-1.3B --configs beam4 --out-root "
        "results/humaneval_regen_2026_11_stripped_smoke --max-problems 5 --prompt-variant stripped",
    ]
    assert len(calls) == 6 and all(c.endswith("--prompt-variant stripped") for c in calls)   # beam (HF) too
    assert "prompt-variant stripped" in out   # the echo line shows the variant
    rc, calls, out = _run_script(tmp_path, ["smoke-rerun"], STRIPPED)
    assert rc == 0 and all("--out-root results/humaneval_regen_2026_11_stripped_smoke_rerun " in c for c in calls), out


def test_stripped_fit_flags_base_but_not_chat(tmp_path):
    rc, calls, out = _run_script(tmp_path, ["fit", "codellama/CodeLlama-7b-hf"], STRIPPED)
    assert rc == 0, out
    assert calls == [
        "--model codellama/CodeLlama-7b-hf --configs greedy --out-root results/humaneval_regen_2026_11_stripped_fit "
        "--max-problems 5 --prompt-variant stripped",
        "--model codellama/CodeLlama-7b-hf --configs beam4 --out-root results/humaneval_regen_2026_11_stripped_fit "
        "--max-problems 5 --prompt-variant stripped",
    ]
    rc, calls, out = _run_script(tmp_path, ["fit", "meta-llama/Llama-2-7b-chat-hf"], STRIPPED)
    assert rc == 0, out
    assert calls == [
        "--model meta-llama/Llama-2-7b-chat-hf --configs greedy --out-root results/humaneval_regen_2026_11_stripped_fit "
        "--max-problems 5",
        "--model meta-llama/Llama-2-7b-chat-hf --configs beam4 --out-root results/humaneval_regen_2026_11_stripped_fit "
        "--max-problems 5",
    ]
    assert "prompt-variant unstripped" in out


def test_stripped_small_and_large_share_one_new_root(tmp_path):
    rc, calls, out = _run_script(tmp_path, ["small"], STRIPPED)
    assert rc == 0 and len(calls) == 6, out
    assert all(c.endswith("--out-root results/humaneval_regen_2026_11_stripped --resume --prompt-variant stripped")
               for c in calls), calls
    rc, calls, out = _run_script(tmp_path, ["large"], STRIPPED)   # default: all five 7B models
    assert rc == 0 and len(calls) == 10, out
    by_model = {}
    for c in calls:
        by_model.setdefault(c.split()[1], []).append(c)
        assert "--out-root results/humaneval_regen_2026_11_stripped --resume" in c
    assert list(by_model) == ["codellama/CodeLlama-7b-hf", "codellama/CodeLlama-7b-Instruct-hf",
                              "meta-llama/Llama-2-7b-hf", "meta-llama/Llama-2-7b-chat-hf", "Qwen/Qwen2.5-Coder-7B"]
    for m in LARGE_BASE_IDS:
        assert all(c.endswith("--prompt-variant stripped") for c in by_model[m]), m
    for m in CHAT_IDS:
        assert not any("--prompt-variant" in c for c in by_model[m]), m
    assert any("--configs beam4" in c for c in by_model["codellama/CodeLlama-7b-hf"])
    assert by_model["Qwen/Qwen2.5-Coder-7B"][0].split()[3].endswith(",top_p0.95_t0.2")   # Hasan model


def test_stripped_roots_pass_the_guard_and_never_collide_with_existing_roots():
    from pathlib import Path
    new = ["results/humaneval_regen_2026_11_stripped", "results/humaneval_regen_2026_11_stripped_smoke",
           "results/humaneval_regen_2026_11_stripped_smoke_rerun", "results/humaneval_regen_2026_11_stripped_fit"]
    old = ["results/humaneval_regen_2026_11", "results/humaneval_regen_2026_11_smoke",
           "results/humaneval_regen_2026_11_smoke_rerun", "results/humaneval_regen_2026_11_fit",
           "results/humaneval_regen_2026_11_diag", "results/humaneval_regen_2026_11_smoke2"]
    for r in new:
        regen._check_out_root(Path(r))
    assert not set(new) & set(old)


def test_orchestration_rejects_bad_variant_and_colliding_smoke_names(tmp_path):
    rc, calls, _ = _run_script(tmp_path, ["smoke"], {"PROMPT_VARIANT": "bogus"})
    assert rc != 0 and calls == []
    rc, calls, _ = _run_script(tmp_path, ["diag"], STRIPPED)   # diag has its own fixed variants
    assert rc != 0 and calls == []
    rc, calls, _ = _run_script(tmp_path, ["smoke"], {"SMOKE_NAME": "stripped"})   # would write the stripped full root
    assert rc != 0 and calls == []


def test_default_diag_commands_are_unchanged(tmp_path):
    rc, calls, out = _run_script(tmp_path, ["diag"])
    assert rc == 0 and len(calls) == 6, out
    assert calls[:2] == [
        "--model m-a-p/OpenCodeInterpreter-DS-1.3B --configs greedy_hf --out-root results/humaneval_regen_2026_11_diag --resume",
        "--model m-a-p/OpenCodeInterpreter-DS-1.3B --configs greedy --out-root results/humaneval_regen_2026_11_diag "
        "--prompt-variant stripped --resume",
    ]
