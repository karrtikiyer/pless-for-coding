"""Smoke tests for bench.gsm8k.

No model loading, no HF dataset download. We mock the model/tokenizer and
hand-craft sample completions to exercise the answer extractor, the
pairwise-BLEU diversity, and the CLI plumbing.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest


# ─── prompts ────────────────────────────────────────────────────────────────

def test_wei2022_prompt_has_8_exemplars_and_question_slot():
    """The Wei 2022 8-shot CoT prompt must have exactly 8 'Q:'/'A:' pairs
    plus a final 'Q: {question}\\nA:' slot."""
    from bench.gsm8k.prompts import WEI_2022_GSM8K_8SHOT, format_prompt
    n_q = WEI_2022_GSM8K_8SHOT.count("Q: ")
    n_a = WEI_2022_GSM8K_8SHOT.count("A: ")
    # 8 exemplars + 1 templated final Q + 0 templated A (model generates it)
    assert n_q == 9, f"expected 8 exemplars + 1 templated question, got {n_q}"
    assert n_a == 8, f"expected 8 exemplar answers, got {n_a}"
    assert "{question}" in WEI_2022_GSM8K_8SHOT
    # Plugging in a question must work
    out = format_prompt("How many apples?")
    assert "How many apples?" in out
    assert "{question}" not in out  # the slot got filled


def test_wei2022_prompt_ends_with_canonical_phrase_each_exemplar():
    """All 8 exemplar answers must end with 'The answer is N.' so the model
    is conditioned to emit that phrase."""
    from bench.gsm8k.prompts import WEI_2022_GSM8K_8SHOT
    # Count the verified-canonical answer-anchor occurrences
    n_anchors = WEI_2022_GSM8K_8SHOT.count("The answer is")
    assert n_anchors == 8, f"expected 8 'The answer is' anchors, got {n_anchors}"


# ─── evaluator: answer extraction ───────────────────────────────────────────

@pytest.mark.parametrize("completion,expected", [
    ("After thinking, 5 + 3 = 8. The answer is 8.", "8"),
    ("After thinking, 5 + 3 = 8. The answer is $8.", "8"),
    ("Step 1: x = 100. Step 2: y = 1,234. The answer is 1,234.", "1234"),
    ("She has 0.5 cookies. The answer is 0.5.", "0.5"),
    ("Negative: The answer is -3.", "-3"),
    ("First attempt: The answer is 42. Wait — The answer is 7.", "7"),  # last wins
    ("No anchor here, just text.", None),
    ("", None),
])
def test_extract_predicted_answer(completion, expected):
    from bench.gsm8k.evaluator import extract_predicted_answer
    assert extract_predicted_answer(completion) == expected


@pytest.mark.parametrize("predicted,gold,expected", [
    ("8", "8", True),
    ("8", "8.0", True),
    ("0.5", ".5", True),
    ("8", "9", False),
    (None, "8", False),
    ("8", "", False),
    ("8", "eight", False),  # falls through to string equality
])
def test_numeric_equals(predicted, gold, expected):
    from bench.gsm8k.evaluator import numeric_equals
    assert numeric_equals(predicted, gold) is expected


# ─── diversity: pairwise BLEU on reasoning ─────────────────────────────────

def test_reasoning_extracts_text_before_answer_anchor():
    from bench.gsm8k.diversity import extract_reasoning
    completion = "First we compute 2+2 = 4. Then we have 4 apples. The answer is 4."
    assert "compute 2+2" in extract_reasoning(completion)
    assert "The answer is" not in extract_reasoning(completion)


def test_reasoning_returns_full_text_when_no_anchor():
    from bench.gsm8k.diversity import extract_reasoning
    assert extract_reasoning("Just reasoning, no anchor") == "Just reasoning, no anchor"


def test_pairwise_bleu_diversity_identical_samples_returns_zero():
    """Two identical completions ⇒ zero diversity (deduplication collapses
    them into a single unique reasoning; the function returns 0.0 to signal
    'no diversity' rather than None)."""
    from bench.gsm8k.diversity import pairwise_bleu4_diversity
    same = "We compute 2+2 = 4. The answer is 4."
    result = pairwise_bleu4_diversity([same, same, same])
    assert result == 0.0, f"identical samples should give 0.0 diversity, got {result}"


def test_pairwise_bleu_diversity_distinct_samples_returns_positive():
    """Visibly different reasoning paths should give nonzero diversity > 0.
    We don't assert a precise value because BLEU smoothing + tokenization
    can drift across NLTK versions; we just assert the qualitative property."""
    from bench.gsm8k.diversity import pairwise_bleu4_diversity
    samples = [
        "We compute 2 plus 2. That gives 4. The answer is 4.",
        "Adding two and two together yields four. The answer is 4.",
        "Two plus two equals four obviously. The answer is 4.",
    ]
    div = pairwise_bleu4_diversity(samples)
    assert div is not None
    assert 0.0 < div < 1.0


def test_pairwise_bleu_diversity_under_2_samples_returns_none():
    """With <2 samples diversity is undefined."""
    from bench.gsm8k.diversity import pairwise_bleu4_diversity
    assert pairwise_bleu4_diversity([]) is None
    assert pairwise_bleu4_diversity(["only one sample"]) is None


def test_compute_aggregate_diversity_handles_missing():
    """Per-task records with self_bleu_diversity=None should be skipped."""
    from bench.gsm8k.diversity import compute_aggregate_diversity
    recs = [
        {"task_id": "a", "self_bleu_diversity": 0.5},
        {"task_id": "b", "self_bleu_diversity": None},
        {"task_id": "c", "self_bleu_diversity": 0.3},
    ]
    out = compute_aggregate_diversity(recs)
    assert out["n_tasks_with_diversity"] == 2
    assert abs(out["self_bleu_diversity"] - 0.4) < 1e-6


# ─── dataset loader signature (no actual HF download) ──────────────────────

def test_dataset_loader_signature():
    """load_gsm8k_subset must accept n_problems and seed."""
    import inspect
    from bench.gsm8k.dataset import load_gsm8k_subset, Gsm8kProblem
    sig = inspect.signature(load_gsm8k_subset)
    assert "n_problems" in sig.parameters
    assert "seed" in sig.parameters
    # dataclass fields present
    p = Gsm8kProblem(task_id="x", question="q", gold_answer="0",
                     gold_solution="0", raw_index=0)
    assert p.task_id == "x"


def test_gold_extraction_from_gsm8k_format():
    """The '#### N' suffix is the gold-answer convention in openai/gsm8k."""
    from bench.gsm8k.dataset import _gold_from_answer_field
    # Real-shaped GSM8K answer
    ans = "First step: x = 2.\nSecond step: y = 3.\n#### 5"
    assert _gold_from_answer_field(ans) == "5"
    # With commas
    assert _gold_from_answer_field("blah\n#### 1,234") == "1234"
    # Missing anchor
    assert _gold_from_answer_field("no anchor here") == ""


# ─── runner / eval CLI plumbing ────────────────────────────────────────────

def test_runner_parse_args_requires_model_and_method():
    from bench.gsm8k.runner import parse_args
    with pytest.raises(SystemExit):
        parse_args([])
    with pytest.raises(SystemExit):
        parse_args(["--model", "x"])
    ns = parse_args(["--model", "x", "--method", "temp"])
    assert ns.model == "x"
    assert ns.method == "temp"


def test_runner_parse_args_requires_alpha_for_pless_alpha():
    from bench.gsm8k.runner import parse_args
    with pytest.raises(SystemExit):
        parse_args(["--model", "x", "--method", "pless_alpha"])
    ns = parse_args(["--model", "x", "--method", "pless_alpha", "--alpha", "5.0"])
    assert ns.alpha == 5.0


def test_runner_alpha_only_with_pless_alpha():
    from bench.gsm8k.runner import parse_args
    with pytest.raises(SystemExit):
        parse_args(["--model", "x", "--method", "temp", "--alpha", "2.0"])


def test_runner_log_entropy_accepted_with_pless():
    """--log-entropy is the new flag for the central-figure entropy probe.
    Should be accepted with --method pless (the GSM8K analog of MBPP's
    pless@T=1.0 entropy recording)."""
    from bench.gsm8k.runner import parse_args
    ns = parse_args(["--model", "x", "--method", "pless", "--log-entropy"])
    assert ns.log_entropy is True


def test_runner_log_entropy_accepted_with_pless_alpha():
    from bench.gsm8k.runner import parse_args
    ns = parse_args([
        "--model", "x", "--method", "pless_alpha",
        "--alpha", "2.0", "--log-entropy",
    ])
    assert ns.log_entropy is True


def test_runner_log_entropy_rejected_with_temp():
    """--log-entropy is incompatible with --method temp (the
    generate_samples_standard path doesn't expose entropy_log)."""
    from bench.gsm8k.runner import parse_args
    with pytest.raises(SystemExit):
        parse_args(["--model", "x", "--method", "temp", "--log-entropy"])


def test_runner_log_entropy_default_false():
    """Without --log-entropy, the flag defaults to False (no sidecar produced)."""
    from bench.gsm8k.runner import parse_args
    ns = parse_args(["--model", "x", "--method", "pless"])
    assert ns.log_entropy is False


def test_eval_runner_parse_args_requires_results_file():
    from bench.gsm8k.eval_runner import parse_args
    with pytest.raises(SystemExit):
        parse_args([])


def test_eval_runner_end_to_end_on_fake_records(tmp_path):
    """Write a tiny synthetic JSONL, invoke the evaluator, check the
    metrics JSON has the expected shape."""
    import json
    from bench.gsm8k.eval_runner import main as eval_main
    records = [
        {
            "model": "fake/model",
            "method": "pless_alpha",
            "alpha": 2.0,
            "temperature": 1.0,
            "task_id": "gsm8k_0001",
            "raw_index": 1,
            "question": "What is 2 + 2?",
            "prompt_text": "...",
            "gold_answer": "4",
            "samples": [
                "We compute 2 + 2 = 4. The answer is 4.",
                "Two plus two equals four. The answer is 4.",
                "I am unsure. The answer is 5.",  # wrong
            ],
        },
        {
            "model": "fake/model",
            "method": "pless_alpha",
            "alpha": 2.0,
            "temperature": 1.0,
            "task_id": "gsm8k_0002",
            "raw_index": 2,
            "question": "What is 10 - 3?",
            "prompt_text": "...",
            "gold_answer": "7",
            "samples": [
                "10 - 3 = 7. The answer is 7.",
                "Subtracting 3 from 10 gives 7. The answer is 7.",
            ],
        },
    ]
    jsonl_path = tmp_path / "fake.jsonl"
    jsonl_path.write_text("\n".join(json.dumps(r) for r in records))

    out_path = tmp_path / "fake_metrics.json"
    eval_main(["--results-file", str(jsonl_path), "--output", str(out_path)])
    assert out_path.exists()
    metrics = json.loads(out_path.read_text())
    # pass@1 = avg per-task pass rate = (2/3 + 2/2) / 2 = (0.667 + 1.0) / 2 ≈ 0.833
    assert "1" in metrics["pass_at_k"]
    assert abs(metrics["pass_at_k"]["1"] - 0.8333333) < 1e-3
    assert metrics["n_tasks"] == 2
    # Both tasks have 2 unique correct → both contribute to diversity
    assert metrics["n_tasks_with_diversity"] == 2
    # Diversity must be a float in [0, 1]
    assert 0.0 <= metrics["self_bleu_diversity"] <= 1.0
