"""Tests for bench.eval.compare_with_paper."""

import pytest

from bench.eval.compare_with_paper import (
    PAPER_RESULTS,
    _MODEL_SHORT,
    build_comparison_rows,
    format_comparison_table,
    format_extended_metrics_table,
    generate_report,
    rank_of,
)


# ── Fixtures ──────────────────────────────────────────────────────────────

def _make_metrics(method: str, pass_at_1: float, model: str = "meta-llama/Llama-2-7b-hf") -> dict:
    """Create a minimal metrics dict for testing."""
    return {
        "model": model,
        "method": method,
        "temperature": 1.0 if method != "temp" else 0.7,
        "dataset": "mbpp",
        "num_tasks": 257,
        "num_samples_per_task": 10,
        "pass_at_k": {
            "1": pass_at_1,
            "3": pass_at_1 * 1.5,
            "5": pass_at_1 * 1.8,
            "10": pass_at_1 * 2.0,
        },
        "cover_at_t": {"0.1": 50.0, "0.3": 35.0, "0.5": 20.0, "0.7": 10.0},
        "cover_at_t_distinct": {"0.1": 48.0, "0.3": 30.0, "0.5": 15.0, "0.7": 5.0},
    }


# ── Paper results validation ─────────────────────────────────────────────

class TestPaperResultsDict:
    def test_has_both_models(self):
        assert "meta-llama/Llama-2-7b-hf" in PAPER_RESULTS
        assert "meta-llama/Llama-2-7b-chat-hf" in PAPER_RESULTS

    def test_each_model_has_14_methods(self):
        for model_key, methods in PAPER_RESULTS.items():
            assert len(methods) == 14, f"{model_key} has {len(methods)} methods, expected 14"

    def test_all_values_are_positive_floats(self):
        for model_key, methods in PAPER_RESULTS.items():
            for method, score in methods.items():
                assert isinstance(score, (int, float)), f"{model_key}/{method}: not a number"
                assert 0 < score <= 100, f"{model_key}/{method}: {score} not in (0, 100]"

    def test_expected_methods_present(self):
        expected = {"Greedy", "Beam Search", "Temperature", "Top-p", "Top-k", "FSD-d"}
        for model_key, methods in PAPER_RESULTS.items():
            assert expected.issubset(methods.keys()), (
                f"{model_key} missing methods: {expected - methods.keys()}"
            )

    def test_all_models_have_short_names(self):
        for model_key in PAPER_RESULTS:
            assert model_key in _MODEL_SHORT


# ── Comparison table building ─────────────────────────────────────────────

class TestBuildComparisonTable:
    def test_rows_include_paper_and_ours(self):
        our_metrics = [_make_metrics("pless", 0.20)]
        rows = build_comparison_rows(PAPER_RESULTS["meta-llama/Llama-2-7b-hf"], our_metrics)
        sources = {r["source"] for r in rows}
        assert sources == {"Paper", "Ours"}

    def test_rows_sorted_descending(self):
        our_metrics = [_make_metrics("pless", 0.20)]
        rows = build_comparison_rows(PAPER_RESULTS["meta-llama/Llama-2-7b-hf"], our_metrics)
        scores = [r["pass_at_1"] for r in rows]
        assert scores == sorted(scores, reverse=True)

    def test_total_count(self):
        our_metrics = [
            _make_metrics("pless", 0.20),
            _make_metrics("pless_norm", 0.19),
            _make_metrics("temp", 0.17),
        ]
        rows = build_comparison_rows(PAPER_RESULTS["meta-llama/Llama-2-7b-hf"], our_metrics)
        assert len(rows) == 14 + 3  # 14 paper + 3 ours

    def test_our_scores_converted_to_pct(self):
        our_metrics = [_make_metrics("pless", 0.2345)]
        rows = build_comparison_rows(PAPER_RESULTS["meta-llama/Llama-2-7b-hf"], our_metrics)
        our_row = next(r for r in rows if r["source"] == "Ours")
        assert abs(our_row["pass_at_1"] - 23.45) < 0.01


# ── Ranking ───────────────────────────────────────────────────────────────

class TestRanking:
    def test_rank_of_top_method(self):
        rows = [
            {"method": "A", "source": "Paper", "pass_at_1": 30.0},
            {"method": "B", "source": "Ours", "pass_at_1": 20.0},
            {"method": "C", "source": "Paper", "pass_at_1": 10.0},
        ]
        assert rank_of(rows, "A") == 1
        assert rank_of(rows, "B") == 2
        assert rank_of(rows, "C") == 3

    def test_rank_of_missing(self):
        rows = [{"method": "A", "source": "Paper", "pass_at_1": 10.0}]
        assert rank_of(rows, "missing") is None

    def test_pless_ranked_among_paper_methods(self):
        # If pless gets 23.5%, it should beat FSD-d (21.2%) and rank #1 for base model
        our_metrics = [_make_metrics("pless", 0.235)]
        rows = build_comparison_rows(PAPER_RESULTS["meta-llama/Llama-2-7b-hf"], our_metrics)
        r = rank_of(rows, "P-Less (t=1.0)")
        assert r is not None
        assert r == 1  # 23.5% beats FSD-d's 21.2%


# ── Table formatting ─────────────────────────────────────────────────────

class TestFormatting:
    def test_comparison_table_has_markdown_structure(self):
        our_metrics = [_make_metrics("pless", 0.20)]
        rows = build_comparison_rows(PAPER_RESULTS["meta-llama/Llama-2-7b-hf"], our_metrics)
        table = format_comparison_table(rows, "Test Model")
        assert "| Rank |" in table
        assert "| ---:" in table
        assert "**←**" in table  # our methods are marked

    def test_extended_metrics_table_includes_all_k(self):
        our_metrics = [_make_metrics("pless", 0.20)]
        table = format_extended_metrics_table(our_metrics, "Test Model")
        assert "pass@1" in table
        assert "pass@10" in table
        assert "cover@0.1" in table
        assert "cover@0.1 (dist)" in table

    def test_extended_metrics_empty_input(self):
        assert format_extended_metrics_table([], "X") == ""


# ── Full report ───────────────────────────────────────────────────────────

class TestGenerateReport:
    def test_report_contains_key_sections(self):
        our_metrics = {
            "meta-llama/Llama-2-7b-hf": [_make_metrics("pless", 0.20)],
        }
        report = generate_report(our_metrics)
        assert "# MBPP" in report
        assert "## pass@1 Comparison" in report
        assert "## Extended Metrics" in report
        assert "## Analysis" in report
        assert "### Limitations" in report
