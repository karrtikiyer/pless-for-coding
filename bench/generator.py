import copy

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model_and_tokenizer(model_id: str):
    """Load model in bfloat16 with SDPA attention and its tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
        use_cache=True,
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
    prompt_len = input_ids.shape[1]
    samples = []
    for _ in range(n_samples):
        with torch.no_grad():
            output = model.generate(
                input_ids,
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
