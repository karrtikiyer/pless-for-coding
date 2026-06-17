"""Model-aware </think> token-id resolution for the vLLM split / loop-force decoder.

The logits processor forces a single </think> token id and detects the think->code
boundary by that id. The id MUST be derived from the model's own tokenizer, not
hardcoded to Qwen3's 151668 — which is out of range for DeepSeek-R1-Distill's
128000-token vocab and would IndexError at `row[151668]` during a loop-force run.

Tokenizers are loaded from the local HF cache; run with HF_HUB_OFFLINE=1.
"""
import pytest
from transformers import AutoTokenizer

from bench.generator_vllm import QWEN3_THINK_END_TOKEN_ID, resolve_think_end_id


def test_qwen3_constant_is_documented_value():
    # The constant is kept as Qwen3's known value (default + sanity anchor).
    assert QWEN3_THINK_END_TOKEN_ID == 151668


@pytest.mark.parametrize(
    "model_id, expected",
    [
        ("Qwen/Qwen3-8B", 151668),
        ("deepseek-ai/DeepSeek-R1-Distill-Llama-8B", 128014),
    ],
)
def test_resolve_matches_known_ids(model_id, expected):
    tok = AutoTokenizer.from_pretrained(model_id)
    assert resolve_think_end_id(tok) == expected


def test_qwen3_resolution_equals_hardcoded_constant():
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    assert resolve_think_end_id(tok) == QWEN3_THINK_END_TOKEN_ID


class _MultiTokenTokenizer:
    """Stub whose </think> encodes to more than one token."""

    def encode(self, text, add_special_tokens=False):
        return [1, 2, 3]


def test_multitoken_think_end_raises():
    # The processor can only force ONE token id; a multi-token </think> is a hard
    # error, not something to silently truncate.
    with pytest.raises(AssertionError):
        resolve_think_end_id(_MultiTokenTokenizer())
