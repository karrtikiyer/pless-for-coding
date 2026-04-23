"""Canonical model registry for benchmarking.

Shell scripts can query this via:
    uv run python -c "from bench.models import MBPP_MODELS; ..."
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    model_id: str          # HuggingFace ID e.g. "Qwen/Qwen-7B"
    prompt_style: str      # "paper" | "bigcode" (base models only; instruct auto-detects)
    legacy: bool = False   # Needs transformers<5


MBPP_MODELS: list[ModelConfig] = [
    ModelConfig("codellama/CodeLlama-7b-hf",          prompt_style="paper"),
    ModelConfig("codellama/CodeLlama-7b-Instruct-hf", prompt_style="paper"),
    ModelConfig("meta-llama/Llama-2-7b-hf",           prompt_style="paper"),
    ModelConfig("meta-llama/Llama-2-7b-chat-hf",      prompt_style="paper"),
    ModelConfig("Qwen/Qwen-7B",                       prompt_style="paper", legacy=True),
    ModelConfig("Qwen/Qwen-7B-Chat",                  prompt_style="paper", legacy=True),
    ModelConfig("Qwen/Qwen2.5-Coder-1.5B",           prompt_style="bigcode"),
    ModelConfig("Qwen/Qwen2.5-Coder-3B",             prompt_style="bigcode"),
    ModelConfig("Qwen/Qwen2.5-Coder-3B-Instruct",  prompt_style="bigcode"),
    ModelConfig("m-a-p/OpenCodeInterpreter-DS-1.3B",  prompt_style="bigcode"),
    ModelConfig("Qwen/Qwen3-8B",                      prompt_style="paper"),
]

HUMANEVAL_MODELS: list[ModelConfig] = [
    ModelConfig("codellama/CodeLlama-7b-hf",              prompt_style="paper"),
    ModelConfig("codellama/CodeLlama-7b-Instruct-hf",     prompt_style="paper"),
    ModelConfig("mistralai/Codestral-22B-v0.1",           prompt_style="paper"),
    ModelConfig("Qwen/Qwen2.5-Coder-7B",                 prompt_style="paper"),
    ModelConfig("Qwen/Qwen2.5-Coder-7B-Instruct",        prompt_style="paper"),
    ModelConfig("Qwen/Qwen3-Coder-30B-A3B-Instruct",     prompt_style="paper"),
    ModelConfig("Qwen/Qwen3-8B",                          prompt_style="paper"),
]
