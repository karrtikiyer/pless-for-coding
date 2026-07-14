"""TDD for the vLLM prompt-encoding fix (bench.generator_vllm.encode_prompt_for_vllm).

Root cause it guards: vLLM tokenizes a *string* prompt with its internal tokenizer,
which for DeepSeek-R1-Distill is transformers-v5's LlamaTokenizer — it overrides the
ByteLevel-BPE pre-tokenizer with Metaspace and silently eats whitespace (HF #45488), so a
Python prompt reaches the model as `deff(a,b):`. The HF backend dodges this by reloading a
PreTrainedTokenizerFast; the vLLM backend only used that safe tokenizer for *decoding*.
This fix pre-encodes with the safe tokenizer so vLLM gets ids (TokensPrompt), byte-identical
to the HF backend — and is a strict no-op when no safe tokenizer was installed (Qwen3).

generator_vllm.py defers all vLLM imports, so the module (and this helper) import fine on a
machine without vLLM/CUDA.
"""
import pytest

from bench.generator_vllm import encode_prompt_for_vllm


class _MockTok:
    """Records calls; returns a fixed id list so we can assert the encode path."""
    def __init__(self):
        self.calls = []

    def encode(self, text):
        self.calls.append(text)
        return [10, 20, 30]


def test_no_safe_tokenizer_returns_string_unchanged():
    # Well-behaved model (e.g. Qwen3) → no safe tokenizer installed → strict no-op.
    assert encode_prompt_for_vllm("def f(): pass", None) == "def f(): pass"


def test_safe_tokenizer_present_preencodes_string_to_ids():
    tok = _MockTok()
    out = encode_prompt_for_vllm("hello world", tok)
    assert out == [10, 20, 30]
    assert tok.calls == ["hello world"]        # encoded exactly the given prompt


def test_already_tokenized_prompt_passes_through_even_with_safe_tokenizer():
    # If a caller already supplied ids (old-Qwen direct-tokenize path), don't re-encode.
    tok = _MockTok()
    ids = [1, 2, 3, 4]
    assert encode_prompt_for_vllm(ids, tok) is ids
    assert tok.calls == []                      # not touched


def test_none_prompt_types_never_double_encoded():
    # list stays list regardless of safe tokenizer presence
    assert encode_prompt_for_vllm([5, 6], None) == [5, 6]


def test_real_deepseek_fix_preserves_whitespace_and_matches_hf():
    """With the real DeepSeek safe tokenizer, the fix (a) preserves whitespace and
    (b) reproduces the exact ids the HF backend feeds. Skips if the tokenizer isn't
    locally available."""
    transformers = pytest.importorskip("transformers")
    mid = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
    try:
        broken = transformers.AutoTokenizer.from_pretrained(mid)
        safe = transformers.PreTrainedTokenizerFast.from_pretrained(mid)
    except Exception as e:                       # not cached / offline
        pytest.skip(f"DeepSeek tokenizer unavailable: {e}")

    prompt = "Solve:\n\n    for i in range(n):\n        a[i] += 1"

    # HF backend: bench/generator.py encodes the (string) prompt with the safe tokenizer.
    hf_ids = safe.encode(prompt)

    # Fix path: encode_prompt_for_vllm with the safe tokenizer → must equal HF ids.
    fixed_ids = encode_prompt_for_vllm(prompt, safe)
    assert fixed_ids == hf_ids

    # And it must round-trip with whitespace intact (the whole point).
    assert "for i in range(n):" in safe.decode(fixed_ids, skip_special_tokens=True)

    # Sanity: the OLD broken path (string → vLLM's internal LlamaTokenizer) mangles it —
    # whitespace is stripped, so the collapsed signature appears and the correct one doesn't.
    broken_decoded = broken.decode(broken.encode(prompt), skip_special_tokens=True)
    assert "foriinrange(n):" in broken_decoded
    assert "for i in range(n):" not in broken_decoded
