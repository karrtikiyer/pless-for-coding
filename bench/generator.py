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


def load_model_and_tokenizer(model_id: str):
    """Load model in bfloat16 with SDPA attention and its tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    # Old Qwen-7B uses custom attention that doesn't support SDPA.
    attn_impl = "eager" if model_id == "Qwen/Qwen-7B" else "sdpa"
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
    prompt_text: str,
    n_samples: int,
    max_new_tokens: int,
    temperature: float,
    stop_strings: list[str] | None = None,
) -> list[str]:
    """Generate samples using standard model.generate() with batched generation."""
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
            top_p=1.0,
            num_return_sequences=n_samples,
            **kwargs,
        )
    decoded_prompt = tokenizer.decode(
        output[0, :prompt_len], skip_special_tokens=True
    )
    samples = []
    for i in range(n_samples):
        full_text = tokenizer.decode(output[i], skip_special_tokens=True)
        text = full_text[len(decoded_prompt):]
        # Post-generation truncation as safety net
        if stop_strings:
            text = _truncate_at_stop(text, stop_strings)
        samples.append(text)
    return samples


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


def generate_samples(
    model,
    tokenizer,
    prompt_text: str,
    sampler_fn,
    n_samples: int,
    max_new_tokens: int,
    temperature: float,
    stop_strings: list[str] | None = None,
) -> list[str]:
    """Generate n_samples completions in parallel using batched decoding."""
    input_ids = tokenizer.encode(prompt_text, return_tensors="pt").to(model.device)
    eos_id = tokenizer.eos_token_id
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
    next_tokens = sampler_fn(probs.clone())  # (N, 1)
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
                next_tokens = sampler_fn(probs.clone()).view(N)  # (N,)

                # Only write tokens for unfinished sequences
                next_tokens = torch.where(finished, torch.tensor(eos_id, device=model.device), next_tokens)
                all_ids[:, step] = next_tokens

                finished = finished | (next_tokens == eos_id)

                # Check stop sequences in rolling text buffers
                if stop_strings:
                    for i in range(N):
                        if not finished[i]:
                            # Decode accumulated tokens to get accurate text
                            text_buffers[i] = tokenizer.decode(
                                all_ids[i, :step + 1], skip_special_tokens=True
                            )
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
