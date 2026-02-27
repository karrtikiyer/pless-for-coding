import copy

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
        dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation=attn_impl,
        use_cache=True,
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


def generate_samples_standard(
    model,
    tokenizer,
    prompt_text: str,
    n_samples: int,
    max_new_tokens: int,
    temperature: float,
) -> list[str]:
    """Generate samples using standard model.generate() with temperature sampling."""
    input_ids = tokenizer.encode(prompt_text, return_tensors="pt").to(model.device)
    attention_mask = torch.ones_like(input_ids)
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    prompt_len = input_ids.shape[1]
    samples = []
    for _ in range(n_samples):
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
            )
        generated = output[0, prompt_len:]
        samples.append(tokenizer.decode(generated, skip_special_tokens=True))
    return samples


def _clone_past_key_values(past_key_values):
    """Deep-clone a KV cache (DynamicCache or tuple of tensors)."""
    return copy.deepcopy(past_key_values)


def generate_samples(
    model,
    tokenizer,
    prompt_text: str,
    sampler_fn,
    n_samples: int,
    max_new_tokens: int,
    temperature: float,
) -> list[str]:
    """Generate n_samples completions for a given prompt using the provided sampler."""
    input_ids = tokenizer.encode(prompt_text, return_tensors="pt").to(model.device)

    # Prefill: run prompt through model once to get KV cache
    with torch.no_grad():
        prefill_output = model(input_ids=input_ids, return_dict=True)
    prefill_kv = prefill_output.past_key_values
    prefill_logits = prefill_output.logits

    samples = []
    for _ in range(n_samples):
        past_key_values = _clone_past_key_values(prefill_kv)
        generated_ids = []

        # First token: use prefill logits
        logits = prefill_logits[0, -1].float()
        if temperature != 1.0:
            logits = logits / temperature
        probs = torch.softmax(logits, dim=-1).unsqueeze(0)
        next_token_id = sampler_fn(probs.clone())
        token_id = next_token_id.item()
        generated_ids.append(token_id)

        if token_id == tokenizer.eos_token_id:
            samples.append(tokenizer.decode(generated_ids, skip_special_tokens=True))
            continue

        # Subsequent tokens
        encodings = next_token_id.reshape(1, 1)
        with torch.no_grad():
            for _ in range(max_new_tokens - 1):
                output = model(
                    input_ids=encodings,
                    past_key_values=past_key_values,
                    return_dict=True,
                )
                logits = output.logits[0, -1].float()
                if temperature != 1.0:
                    logits = logits / temperature
                probs = torch.softmax(logits, dim=-1).unsqueeze(0)
                next_token_id = sampler_fn(probs.clone())
                token_id = next_token_id.item()
                generated_ids.append(token_id)

                if token_id == tokenizer.eos_token_id:
                    break

                past_key_values = output.past_key_values
                encodings = next_token_id.view(1, 1)

        samples.append(tokenizer.decode(generated_ids, skip_special_tokens=True))

    return samples
