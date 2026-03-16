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
