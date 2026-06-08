import torch
from unittest.mock import MagicMock
from bench.generator import generate_samples_standard


def _make_mocks(n_samples=2):
    tokenizer = MagicMock()
    tokenizer.pad_token_id = 0
    tokenizer.eos_token_id = 2
    tokenizer.encode.return_value = torch.tensor([[1, 2, 3]])  # real tensor: .to() + .shape work
    tokenizer.decode.return_value = "decoded"
    model = MagicMock()
    model.device = torch.device("cpu")
    model.generate.return_value = torch.zeros(n_samples, 5, dtype=torch.long)
    return model, tokenizer


def test_generate_samples_standard_passes_top_p_to_model_generate():
    """top_p=0.9 must be forwarded to model.generate()."""
    model, tokenizer = _make_mocks()
    generate_samples_standard(model, tokenizer, "hello", 2, 10, 1.0, top_p=0.9)
    assert model.generate.call_args.kwargs["top_p"] == 0.9


def test_generate_samples_standard_default_top_p_is_1():
    """Without top_p arg, model.generate() receives top_p=1.0 (backward compat)."""
    model, tokenizer = _make_mocks()
    generate_samples_standard(model, tokenizer, "hello", 2, 10, 1.0)
    assert model.generate.call_args.kwargs["top_p"] == 1.0


def test_generate_samples_standard_passes_repetition_penalty():
    """repetition_penalty must be forwarded to model.generate() — needed for the
    provider-faithful baseline (Qwen2.5-Coder ships rep_penalty 1.1/1.05)."""
    model, tokenizer = _make_mocks()
    generate_samples_standard(model, tokenizer, "hello", 2, 10, 0.7,
                              repetition_penalty=1.1)
    assert model.generate.call_args.kwargs["repetition_penalty"] == 1.1


def test_generate_samples_standard_default_repetition_penalty_is_1():
    """Default rep_penalty=1.0 is a no-op (backward compat — pless and the
    other benchmarks must be unaffected)."""
    model, tokenizer = _make_mocks()
    generate_samples_standard(model, tokenizer, "hello", 2, 10, 1.0)
    assert model.generate.call_args.kwargs["repetition_penalty"] == 1.0


# ─── hf_batch_size: chunked generation to avoid OOM at large n_samples ─────
# Background: on a single H100 80GB, model.generate(num_return_sequences=100)
# for Deepseek-6.7B + 1024 max_new_tokens triggers CUDA OOM (~100 GiB KV cache
# vs 80 GiB capacity). Chunking n_samples into smaller batches and looping
# fixes it without changing the output distribution (each chunk uses the
# same sampler / dtype / temperature; sample independence is preserved).


def _make_variable_batch_mocks():
    """Mock whose model.generate returns a tensor matching the requested
    num_return_sequences each call. Lets us assert the chunk plan."""
    tokenizer = MagicMock()
    tokenizer.pad_token_id = 0
    tokenizer.eos_token_id = 2
    tokenizer.encode.return_value = torch.tensor([[1, 2, 3]])
    tokenizer.decode.return_value = "decoded"
    model = MagicMock()
    model.device = torch.device("cpu")

    def _generate(*args, **kwargs):
        b = kwargs["num_return_sequences"]
        return torch.zeros(b, 5, dtype=torch.long)

    model.generate.side_effect = _generate
    return model, tokenizer


def test_hf_batch_size_default_none_makes_single_call():
    """Backward compat: when hf_batch_size is omitted, n_samples is passed
    as num_return_sequences in ONE model.generate call (current behavior)."""
    model, tokenizer = _make_variable_batch_mocks()
    samples = generate_samples_standard(
        model, tokenizer, "hello", n_samples=10, max_new_tokens=8,
        temperature=1.0,
    )
    assert model.generate.call_count == 1
    assert model.generate.call_args.kwargs["num_return_sequences"] == 10
    assert len(samples) == 10


def test_hf_batch_size_chunks_when_smaller_than_n_samples():
    """n_samples=10 with hf_batch_size=3 must produce 4 generate() calls:
    chunks of (3, 3, 3, 1) summing to 10."""
    model, tokenizer = _make_variable_batch_mocks()
    samples = generate_samples_standard(
        model, tokenizer, "hello", n_samples=10, max_new_tokens=8,
        temperature=1.0, hf_batch_size=3,
    )
    assert model.generate.call_count == 4, (
        f"expected 4 calls (3+3+3+1=10), got {model.generate.call_count}"
    )
    chunk_sizes = [c.kwargs["num_return_sequences"]
                   for c in model.generate.call_args_list]
    assert chunk_sizes == [3, 3, 3, 1], (
        f"expected chunks [3,3,3,1], got {chunk_sizes}"
    )
    assert len(samples) == 10


def test_hf_batch_size_equal_to_n_samples_single_call():
    """hf_batch_size == n_samples is a single call (no overhead from chunking)."""
    model, tokenizer = _make_variable_batch_mocks()
    samples = generate_samples_standard(
        model, tokenizer, "hello", n_samples=10, max_new_tokens=8,
        temperature=1.0, hf_batch_size=10,
    )
    assert model.generate.call_count == 1
    assert model.generate.call_args.kwargs["num_return_sequences"] == 10
    assert len(samples) == 10


def test_hf_batch_size_larger_than_n_samples_clamps():
    """hf_batch_size > n_samples should clamp down (single call, batch=n_samples)
    — avoids requesting batch larger than samples requested."""
    model, tokenizer = _make_variable_batch_mocks()
    samples = generate_samples_standard(
        model, tokenizer, "hello", n_samples=5, max_new_tokens=8,
        temperature=1.0, hf_batch_size=100,
    )
    assert model.generate.call_count == 1
    assert model.generate.call_args.kwargs["num_return_sequences"] == 5
    assert len(samples) == 5


def test_hf_batch_size_100_into_chunks_of_10():
    """Phase A's target shape: N=100, hf_batch_size=10 → 10 calls × batch 10."""
    model, tokenizer = _make_variable_batch_mocks()
    samples = generate_samples_standard(
        model, tokenizer, "hello", n_samples=100, max_new_tokens=8,
        temperature=1.0, hf_batch_size=10,
    )
    assert model.generate.call_count == 10
    chunk_sizes = [c.kwargs["num_return_sequences"]
                   for c in model.generate.call_args_list]
    assert chunk_sizes == [10] * 10
    assert len(samples) == 100


def test_hf_batch_size_preserves_other_kwargs():
    """Chunking must forward all other generate() kwargs to every chunk —
    temperature, top_p, top_k, max_new_tokens stay the same per call."""
    model, tokenizer = _make_variable_batch_mocks()
    generate_samples_standard(
        model, tokenizer, "hello", n_samples=6, max_new_tokens=42,
        temperature=0.7, top_p=0.9, top_k=50, hf_batch_size=2,
    )
    assert model.generate.call_count == 3
    for call in model.generate.call_args_list:
        assert call.kwargs["temperature"] == 0.7
        assert call.kwargs["top_p"] == 0.9
        assert call.kwargs["top_k"] == 50
        assert call.kwargs["max_new_tokens"] == 42
