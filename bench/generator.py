import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

# Monkey-patch transformers so that the outdated transformers-stream-generator
# package (required by old Qwen-7B remote code) can import without crashing
# on transformers 5.x where many classes were removed.
import types as _types
import sys as _sys

# Stub top-level names
for _cls in ("DisjunctiveConstraint", "BeamSearchScorer", "PhrasalConstraint",
             "ConstrainedBeamSearchScorer"):
    if not hasattr(transformers, _cls):
        setattr(transformers, _cls, type(_cls, (), {}))

# Stub names in transformers.generation.utils
_gen_utils = _sys.modules.get("transformers.generation.utils")
if _gen_utils is None:
    import transformers.generation.utils as _gen_utils
for _name in ("GenerateOutput", "SampleOutput"):
    if not hasattr(_gen_utils, _name):
        setattr(_gen_utils, _name, type(_name, (), {}))

# Restore DynamicCache subscript access removed in transformers 5.x.
# Qwen-7B's remote code does past_key_values[i] to access KV cache layers.
from transformers.cache_utils import DynamicCache

if not hasattr(DynamicCache, '__getitem__'):
    def _dynamic_cache_getitem(self, i):
        layer = self.layers[i]
        if not layer.is_initialized:
            return None
        return (layer.keys, layer.values)
    DynamicCache.__getitem__ = _dynamic_cache_getitem

# Restore get_head_mask removed in transformers 5.x.
# Qwen-7B's remote code calls self.get_head_mask(head_mask, num_layers).
from transformers import PreTrainedModel

if not hasattr(PreTrainedModel, 'get_head_mask'):
    def _get_head_mask(self, head_mask, num_hidden_layers, is_attention_chunked=False):
        if head_mask is not None:
            raise NotImplementedError("head_mask pruning is no longer supported")
        return [None] * num_hidden_layers
    PreTrainedModel.get_head_mask = _get_head_mask

# Fix transformers 5.x _initialize_weights bug for remote code models:
# Non-persistent buffers (e.g. QWenAttention.masked_bias) don't have
# _is_hf_initialized, causing _init_weights to be called on the parent
# module even though its child parameters ARE correctly loaded. Qwen's
# _init_weights then re-randomizes c_proj.weight via a named_parameters()
# loop, corrupting 32 attention output projection weights.
_TF5 = int(transformers.__version__.split(".")[0]) >= 5

if _TF5:
    _orig_initialize_weights = PreTrainedModel._initialize_weights

    def _fixed_initialize_weights(self, module, is_remote_code=False):
        if is_remote_code and not getattr(module, '_is_hf_initialized', False):
            for buf in module.buffers(recurse=False):
                if buf is not None and not getattr(buf, '_is_hf_initialized', False):
                    buf._is_hf_initialized = True
        return _orig_initialize_weights(self, module, is_remote_code)

    PreTrainedModel._initialize_weights = _fixed_initialize_weights


def load_model_and_tokenizer(model_id: str):
    """Load model in bfloat16 with SDPA attention and its tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    # Detect broken byte-level BPE decoder.  transformers 5.x + LlamaTokenizer
    # overrides the ByteLevel decoder from tokenizer.json for DeepSeek-Coder /
    # OCI tokenizers, destroying whitespace.  Reload as PreTrainedTokenizerFast
    # which respects the tokenizer.json ByteLevel decoder.
    _test = "a b\nc"
    if tokenizer.decode(tokenizer.encode(_test), skip_special_tokens=True).strip() != _test:
        from transformers import PreTrainedTokenizerFast
        tokenizer = PreTrainedTokenizerFast.from_pretrained(model_id, trust_remote_code=True)
        assert tokenizer.decode(tokenizer.encode(_test), skip_special_tokens=True).strip() == _test, \
            f"Tokenizer for {model_id} cannot round-trip whitespace"

    # Old Qwen-7B / Qwen-7B-Chat use custom attention that doesn't support SDPA.
    is_old_qwen = model_id.startswith("Qwen/Qwen-7B")
    attn_impl = "eager" if is_old_qwen else "sdpa"
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation=attn_impl,
        use_cache=True,
        trust_remote_code=True,
    )
    model.eval()
    # torch.compile disabled: reduce-overhead mode conflicts with transformers 5.x
    # DynamicCache (CUDAGraph overwrites KV cache tensors). Re-enable when fixed upstream.

    # Qwen-7B's remote code manages its own tuple-of-tuples KV cache and expects
    # past_key_values=None on the first forward pass.  transformers 5.x generate()
    # pre-creates a DynamicCache(config=...) with uninitialised layers (keys=None),
    # which Qwen misinterprets as a populated cache → 'NoneType' has no attr 'size'.
    # Opting out makes generate() skip DynamicCache creation entirely.
    if is_old_qwen:
        type(model)._supports_default_dynamic_cache = classmethod(lambda cls: False)

    # Old Qwen tokenizers lack chat_template; set Qwen's ChatML format so
    # tokenizer.apply_chat_template() works for instruct/chat models.
    # Also flag the tokenizer so format_prompt_instruct uses tokenize=True
    # (old Qwen encode() splits <|im_start|>/<|im_end|> into subwords).
    if is_old_qwen and tokenizer.chat_template is None:
        tokenizer.chat_template = (
            "{% for message in messages %}"
            "<|im_start|>{{ message['role'] }}\n{{ message['content'] }}<|im_end|>\n"
            "{% endfor %}"
            "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
        )
        tokenizer._qwen_direct_tokenize = True

    return model, tokenizer


def _truncate_at_stop(text: str, stop_strings: list[str]) -> str:
    """Truncate text at the first occurrence of any stop string."""
    earliest = len(text)
    for stop in stop_strings:
        idx = text.find(stop)
        if idx != -1 and idx < earliest:
            earliest = idx
    return text[:earliest]


def generate_samples_standard(
    model,
    tokenizer,
    prompt_text: str | list[int],
    n_samples: int,
    max_new_tokens: int,
    temperature: float,
    stop_strings: list[str] | None = None,
    top_p: float = 1.0,
) -> list[str]:
    """Generate samples using standard model.generate() with batched generation."""
    if isinstance(prompt_text, list):
        input_ids = torch.tensor([prompt_text], device=model.device)
    else:
        input_ids = tokenizer.encode(prompt_text, return_tensors="pt").to(model.device)
    attention_mask = torch.ones_like(input_ids)
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    prompt_len = input_ids.shape[1]

    # Try native stop_strings support (transformers 5.x+)
    kwargs = {}
    if stop_strings:
        kwargs["stop_strings"] = stop_strings
        kwargs["tokenizer"] = tokenizer

    with torch.no_grad():
        output = model.generate(
            input_ids,
            attention_mask=attention_mask,
            pad_token_id=pad_token_id,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_k=0,
            top_p=top_p,
            num_return_sequences=n_samples,
            **kwargs,
        )
    decoded_prompt = tokenizer.decode(output[0, :prompt_len], skip_special_tokens=True)
    samples = []
    for i in range(n_samples):
        full_text = tokenizer.decode(output[i], skip_special_tokens=True)
        text = full_text[len(decoded_prompt):]
        # Post-generation truncation as safety net
        if stop_strings:
            text = _truncate_at_stop(text, stop_strings)
        samples.append(text)
    return samples


def _resolve_pad_token_id(tokenizer):
    """Resolve pad_token_id with robust fallback for old Qwen tokenizers."""
    pad_id = tokenizer.pad_token_id
    if pad_id is not None:
        return pad_id
    pad_id = tokenizer.eos_token_id
    if pad_id is not None:
        return pad_id
    for candidate in ("<|endoftext|>", "<|im_end|>"):
        cid = tokenizer.convert_tokens_to_ids(candidate)
        if cid is not None and cid != tokenizer.unk_token_id:
            return cid
    return 0  # last resort


def generate_samples_greedy(
    model,
    tokenizer,
    prompt_text: str | list[int],
    max_new_tokens: int,
    stop_strings: list[str] | None = None,
) -> list[str]:
    """Generate a single greedy (argmax) completion."""
    if isinstance(prompt_text, list):
        input_ids = torch.tensor([prompt_text], device=model.device)
    else:
        input_ids = tokenizer.encode(prompt_text, return_tensors="pt").to(model.device)
    attention_mask = torch.ones_like(input_ids)
    pad_token_id = _resolve_pad_token_id(tokenizer)
    prompt_len = input_ids.shape[1]

    kwargs = {}
    if stop_strings:
        kwargs["stop_strings"] = stop_strings
        kwargs["tokenizer"] = tokenizer

    with torch.no_grad():
        output = model.generate(
            input_ids,
            attention_mask=attention_mask,
            pad_token_id=pad_token_id,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            **kwargs,
        )
    decoded_prompt = tokenizer.decode(output[0, :prompt_len], skip_special_tokens=True)
    full_text = tokenizer.decode(output[0], skip_special_tokens=True)
    text = full_text[len(decoded_prompt):]
    if stop_strings:
        text = _truncate_at_stop(text, stop_strings)
    return [text]


def generate_samples_beam(
    model,
    tokenizer,
    prompt_text: str | list[int],
    num_beams: int,
    max_new_tokens: int,
    stop_strings: list[str] | None = None,
) -> list[str]:
    """Generate the single best completion via beam search.

    Beam search explores ``num_beams`` paths but returns only the highest-
    scoring sequence.  Beam width is a search budget, not a sample count —
    the lower-ranked beams are correlated search byproducts, not independent
    samples.  This matches the standard evaluation protocol (arXiv:2402.06925).
    """
    if isinstance(prompt_text, list):
        input_ids = torch.tensor([prompt_text], device=model.device)
    else:
        input_ids = tokenizer.encode(prompt_text, return_tensors="pt").to(model.device)
    attention_mask = torch.ones_like(input_ids)
    pad_token_id = _resolve_pad_token_id(tokenizer)
    prompt_len = input_ids.shape[1]

    kwargs = {}
    if stop_strings:
        kwargs["stop_strings"] = stop_strings
        kwargs["tokenizer"] = tokenizer

    with torch.no_grad():
        output = model.generate(
            input_ids,
            attention_mask=attention_mask,
            pad_token_id=pad_token_id,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=num_beams,
            num_return_sequences=1,
            **kwargs,
        )
    decoded_prompt = tokenizer.decode(output[0, :prompt_len], skip_special_tokens=True)
    full_text = tokenizer.decode(output[0], skip_special_tokens=True)
    text = full_text[len(decoded_prompt):]
    if stop_strings:
        text = _truncate_at_stop(text, stop_strings)
    return [text]


def _expand_past_key_values(past_key_values, n: int):
    """Expand a KV cache from batch_size=1 to batch_size=n.

    Supports both DynamicCache objects and plain tuples of tensors.
    """
    from transformers.cache_utils import DynamicCache

    if isinstance(past_key_values, DynamicCache):
        expanded = DynamicCache()
        if hasattr(past_key_values, 'key_cache'):
            # transformers <5: list attributes
            for i in range(len(past_key_values)):
                expanded.update(
                    past_key_values.key_cache[i].expand(n, -1, -1, -1).contiguous(),
                    past_key_values.value_cache[i].expand(n, -1, -1, -1).contiguous(),
                    i,
                )
        else:
            # transformers 5.x: DynamicLayer objects with .keys/.values
            for i, layer in enumerate(past_key_values.layers):
                expanded.update(
                    layer.keys.expand(n, -1, -1, -1).contiguous(),
                    layer.values.expand(n, -1, -1, -1).contiguous(),
                    i,
                )
        return expanded

    # Plain tuple of (key, value) pairs per layer
    return tuple(
        (k.expand(n, -1, -1, -1).contiguous(), v.expand(n, -1, -1, -1).contiguous())
        for k, v in past_key_values
    )


# Uniform-mixing weight applied to probs before pless/pless_norm sampling.
# The samplers claim "at least one token always satisfies the threshold" — true
# mathematically (sum(p²) ≤ max(p) for any valid distribution) but false in float32:
# GPU parallel reduction across 32256 vocab elements accumulates rounding errors that
# can push the computed sum(p²) above max(p), zeroing all tokens → NaN → CUDA crash.
# At α=1e-3, max_p changes by <0.1%, sampled tokens are always identical in practice.
_PLESS_SMOOTH_ALPHA = 1e-3


def generate_samples(
    model,
    tokenizer,
    prompt_text: str | list[int],
    sampler_fn,
    n_samples: int,
    max_new_tokens: int,
    temperature: float,
    stop_strings: list[str] | None = None,
) -> list[str]:
    """Generate n_samples completions in parallel using batched decoding."""
    if isinstance(prompt_text, list):
        input_ids = torch.tensor([prompt_text], device=model.device)
    else:
        input_ids = tokenizer.encode(prompt_text, return_tensors="pt").to(model.device)
    eos_id = tokenizer.eos_token_id
    # Old Qwen tokenizers may not set eos_token_id; fall back to
    # <|endoftext|> (151643) or <|im_end|> which Qwen uses as EOS.
    if eos_id is None:
        for candidate in ("<|endoftext|>", "<|im_end|>"):
            cid = tokenizer.convert_tokens_to_ids(candidate)
            if cid is not None and cid != tokenizer.unk_token_id:
                eos_id = cid
                break
    if eos_id is None:
        eos_id = getattr(model.config, "eos_token_id", None)
    if eos_id is None:
        raise ValueError("Cannot determine eos_token_id for this tokenizer")
    N = n_samples

    # Prefill: run prompt through model once to get KV cache (batch_size=1)
    with torch.no_grad():
        prefill_output = model(input_ids=input_ids, return_dict=True)
    prefill_kv = prefill_output.past_key_values
    prefill_logits = prefill_output.logits  # (1, seq_len, vocab)

    # Expand KV cache to batch_size=N
    past_key_values = _expand_past_key_values(prefill_kv, N)

    # First token from prefill logits — broadcast to all N samples
    logits = prefill_logits[0, -1].float()  # (vocab,)
    if temperature != 1.0:
        logits = logits / temperature
    probs = torch.softmax(logits, dim=-1).unsqueeze(0).expand(N, -1).contiguous()  # (N, vocab)
    probs_s = probs * (1.0 - _PLESS_SMOOTH_ALPHA) + (_PLESS_SMOOTH_ALPHA / probs.shape[-1])
    next_tokens = sampler_fn(probs_s)  # (N, 1)
    next_tokens = next_tokens.view(N)  # (N,)

    # Storage for generated token ids — pad with eos_id
    all_ids = torch.full((N, max_new_tokens), eos_id, dtype=torch.long, device=model.device)
    all_ids[:, 0] = next_tokens

    # Track which sequences are still generating
    finished = next_tokens == eos_id  # (N,)

    # Rolling text buffers for stop sequence detection
    text_buffers = [""] * N if stop_strings else None

    if stop_strings and not finished.all():
        # Decode first token into buffers and check for stops
        for i in range(N):
            if not finished[i]:
                text_buffers[i] = tokenizer.decode(all_ids[i, :1], skip_special_tokens=True)
                for stop in stop_strings:
                    if stop in text_buffers[i]:
                        finished[i] = True
                        break

    if not finished.all():
        encodings = next_tokens.view(N, 1)  # (N, 1)
        with torch.no_grad():
            for step in range(1, max_new_tokens):
                output = model(
                    input_ids=encodings,
                    past_key_values=past_key_values,
                    return_dict=True,
                )
                logits = output.logits[:, -1].float()  # (N, vocab)
                if temperature != 1.0:
                    logits = logits / temperature
                probs = torch.softmax(logits, dim=-1)  # (N, vocab)
                probs_s = probs * (1.0 - _PLESS_SMOOTH_ALPHA) + (_PLESS_SMOOTH_ALPHA / probs.shape[-1])
                next_tokens = sampler_fn(probs_s).view(N)  # (N,)

                # Only write tokens for unfinished sequences
                next_tokens = torch.where(finished, torch.tensor(eos_id, device=model.device), next_tokens)
                all_ids[:, step] = next_tokens

                finished = finished | (next_tokens == eos_id)

                # Check stop sequences in rolling text buffers
                if stop_strings:
                    for i in range(N):
                        if not finished[i]:
                            # Decode accumulated tokens to get accurate text
                            text_buffers[i] = tokenizer.decode(all_ids[i, :step + 1], skip_special_tokens=True)
                            for stop in stop_strings:
                                if stop in text_buffers[i]:
                                    finished[i] = True
                                    break

                if finished.all():
                    break

                past_key_values = output.past_key_values
                encodings = next_tokens.view(N, 1)

    # Decode each sequence (strip trailing eos tokens)
    decoded_prompt = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    samples = []
    for i in range(N):
        ids = all_ids[i]
        # Find first eos token
        eos_positions = (ids == eos_id).nonzero(as_tuple=True)[0]
        if len(eos_positions) > 0:
            ids = ids[:eos_positions[0]]
        full_ids = torch.cat([input_ids[0], ids])
        full_text = tokenizer.decode(full_ids, skip_special_tokens=True)
        text = full_text[len(decoded_prompt):]
        # Post-generation truncation as safety net
        if stop_strings:
            text = _truncate_at_stop(text, stop_strings)
        samples.append(text)

    return samples
