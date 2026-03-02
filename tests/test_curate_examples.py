"""Tests for bench.eval.curate_examples."""

import pytest

from bench.eval.curate_examples import (
    ALL_METHODS,
    BASELINE_METHODS,
    NUM_SAMPLES,
    PLESS_METHODS,
    analyze_code_length,
    analyze_diversity,
    build_task_matrix,
    select_examples,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_metrics(method: str, model: str, per_task: list[dict]) -> dict:
    """Build a minimal metrics dict."""
    return {
        "model": model,
        "method": method,
        "dataset": "humaneval",
        "num_tasks": len(per_task),
        "num_samples_per_task": 10,
        "per_task": per_task,
    }


def _make_per_task(task_id: str, num_correct: int, num_distinct_correct: int = None) -> dict:
    if num_distinct_correct is None:
        num_distinct_correct = min(num_correct, 3)
    return {
        "task_id": task_id,
        "num_correct": num_correct,
        "num_distinct_correct": num_distinct_correct,
        "pass_results": [i < num_correct for i in range(10)],
    }


def _make_detailed_entry(task_id: str, n_samples: int = 10, tokens: int = 50, code: str = "x = 1") -> dict:
    return {
        "task_id": task_id,
        "n_samples": n_samples,
        "n_correct": 5,
        "samples": [
            {
                "code": code,
                "passed": i < 5,
                "tokens_generated": tokens,
                "generation_time_ms": 100.0,
            }
            for i in range(n_samples)
        ],
    }


def _make_all_data_2models_2tasks():
    """Create synthetic all_data with 2 models, 6 methods, 2 tasks.

    Task HumanEval/0: p-less wins (p_less=8, baselines=4)
    Task HumanEval/1: p-less loses (p_less=2, baselines=7)
    """
    models = ["ModelA", "ModelB"]
    task_ids = ["HumanEval/0", "HumanEval/1"]

    # Scores: {task_idx: {method: num_correct}}
    task_scores = {
        0: {"greedy": 4, "temp_0.2": 3, "temp_0.7": 4, "top_p_0.95": 3, "p_less": 8, "p_less_norm": 7},
        1: {"greedy": 7, "temp_0.2": 6, "temp_0.7": 5, "top_p_0.95": 6, "p_less": 2, "p_less_norm": 3},
    }

    all_data = {}
    for model in models:
        metrics = {}
        for method in ALL_METHODS:
            per_task = []
            for tidx, tid in enumerate(task_ids):
                nc = task_scores[tidx][method]
                per_task.append(_make_per_task(tid, nc))
            metrics[method] = _make_metrics(method, model, per_task)

        detailed = {}
        for method in ALL_METHODS:
            detailed[method] = [
                _make_detailed_entry(tid, tokens=50 + tidx * 10)
                for tidx, tid in enumerate(task_ids)
            ]

        all_data[model] = {"detailed": detailed, "metrics": metrics}

    return all_data


# ---------------------------------------------------------------------------
# Tests: build_task_matrix
# ---------------------------------------------------------------------------


class TestBuildTaskMatrix:
    def test_basic_structure(self):
        all_data = _make_all_data_2models_2tasks()
        matrix = build_task_matrix(all_data)

        assert len(matrix) == 2
        assert matrix[0]["task_id"] == "HumanEval/0"
        assert matrix[1]["task_id"] == "HumanEval/1"

    def test_scores_populated(self):
        all_data = _make_all_data_2models_2tasks()
        matrix = build_task_matrix(all_data)

        row0 = matrix[0]
        assert row0["scores"]["ModelA"]["p_less"] == 8
        assert row0["scores"]["ModelA"]["greedy"] == 4
        assert row0["scores"]["ModelB"]["p_less"] == 8

    def test_pless_advantage_positive_for_win(self):
        all_data = _make_all_data_2models_2tasks()
        matrix = build_task_matrix(all_data)

        # Task 0: p-less best = 8, baseline best = 4, advantage = +4 for both models
        assert matrix[0]["pless_advantage"] > 0
        assert matrix[0]["pless_advantage"] == pytest.approx(4.0)

    def test_pless_advantage_negative_for_loss(self):
        all_data = _make_all_data_2models_2tasks()
        matrix = build_task_matrix(all_data)

        # Task 1: p-less best = 3, baseline best = 7, advantage = -4 for both models
        assert matrix[1]["pless_advantage"] < 0
        assert matrix[1]["pless_advantage"] == pytest.approx(-4.0)

    def test_difficulty_assignment(self):
        all_data = _make_all_data_2models_2tasks()
        matrix = build_task_matrix(all_data)

        # Task 0: baseline best = 4 → medium
        assert matrix[0]["difficulty"] == "medium"
        # Task 1: baseline best = 7 → medium
        assert matrix[1]["difficulty"] == "medium"

    def test_difficulty_easy(self):
        """A task with baseline best >= 8 should be 'easy'."""
        all_data = _make_all_data_2models_2tasks()
        # Override: make baselines score 9 on task 0
        for model in all_data:
            for method in BASELINE_METHODS:
                all_data[model]["metrics"][method]["per_task"][0]["num_correct"] = 9
        matrix = build_task_matrix(all_data)
        assert matrix[0]["difficulty"] == "easy"

    def test_difficulty_hard(self):
        """A task with baseline best <= 2 should be 'hard'."""
        all_data = _make_all_data_2models_2tasks()
        for model in all_data:
            for method in BASELINE_METHODS:
                all_data[model]["metrics"][method]["per_task"][0]["num_correct"] = 1
        matrix = build_task_matrix(all_data)
        assert matrix[0]["difficulty"] == "hard"


# ---------------------------------------------------------------------------
# Tests: select_examples
# ---------------------------------------------------------------------------


class TestSelectExamples:
    def _make_matrix(self, n_tasks=20) -> list[dict]:
        """Create a task matrix with varied advantages."""
        rows = []
        for i in range(n_tasks):
            advantage = (i - n_tasks // 2) * 0.5  # range: negative to positive
            scores = {
                "ModelA": {m: 5 for m in ALL_METHODS},
            }
            rows.append({
                "task_id": f"HumanEval/{i}",
                "scores": scores,
                "distinct": {"ModelA": {m: 3 for m in ALL_METHODS}},
                "pless_advantage": advantage,
                "difficulty": "medium",
            })
        return rows

    def test_returns_correct_count(self):
        matrix = self._make_matrix(n_tasks=20)
        result = select_examples(matrix, num_examples=3)

        assert len(result["wins"]) == 3
        assert len(result["losses"]) == 3

    def test_wins_have_positive_advantage(self):
        matrix = self._make_matrix(n_tasks=20)
        result = select_examples(matrix, num_examples=3)

        for row in result["wins"]:
            assert row["pless_advantage"] > 0

    def test_losses_have_negative_advantage(self):
        matrix = self._make_matrix(n_tasks=20)
        result = select_examples(matrix, num_examples=3)

        for row in result["losses"]:
            assert row["pless_advantage"] < 0

    def test_ceiling_filtered(self):
        """Tasks where all methods score 10/10 should be excluded."""
        matrix = self._make_matrix(n_tasks=10)
        # Make the top-advantage task a ceiling task
        matrix[-1]["scores"]["ModelA"] = {m: NUM_SAMPLES for m in ALL_METHODS}
        matrix[-1]["pless_advantage"] = 100  # huge advantage but ceiling

        result = select_examples(matrix, num_examples=3)

        win_ids = {r["task_id"] for r in result["wins"]}
        assert matrix[-1]["task_id"] not in win_ids

    def test_floor_filtered(self):
        """Tasks where all methods score 0/10 should be excluded."""
        matrix = self._make_matrix(n_tasks=10)
        # Make the bottom-advantage task a floor task
        matrix[0]["scores"]["ModelA"] = {m: 0 for m in ALL_METHODS}
        matrix[0]["pless_advantage"] = -100

        result = select_examples(matrix, num_examples=3)

        loss_ids = {r["task_id"] for r in result["losses"]}
        assert matrix[0]["task_id"] not in loss_ids

    def test_ranking_order(self):
        """Wins should be sorted by most positive advantage first."""
        matrix = self._make_matrix(n_tasks=20)
        result = select_examples(matrix, num_examples=5)

        win_advs = [r["pless_advantage"] for r in result["wins"]]
        assert win_advs == sorted(win_advs, reverse=True)

        loss_advs = [r["pless_advantage"] for r in result["losses"]]
        assert loss_advs == sorted(loss_advs)  # most negative first


# ---------------------------------------------------------------------------
# Tests: analyze_code_length (compute_code_stats)
# ---------------------------------------------------------------------------


class TestAnalyzeCodeLength:
    def test_basic_computation(self):
        all_data = _make_all_data_2models_2tasks()
        matrix = build_task_matrix(all_data)
        result = analyze_code_length(matrix, all_data)

        # Should have partition keys
        for label in result:
            assert label in ("win", "loss", "tie")

        # Check that values are populated
        for label, methods in result.items():
            for method, stats in methods.items():
                assert "mean_tokens" in stats
                assert "mean_code_len" in stats
                assert isinstance(stats["mean_tokens"], (int, float))
                assert isinstance(stats["mean_code_len"], (int, float))

    def test_tokens_from_detailed(self):
        """Verify token counts come from the detailed data."""
        all_data = _make_all_data_2models_2tasks()
        # Task 0 has tokens=50, Task 1 has tokens=60
        matrix = build_task_matrix(all_data)
        result = analyze_code_length(matrix, all_data)

        # All partitions should have non-zero mean tokens
        for label, methods in result.items():
            for method, stats in methods.items():
                if stats["mean_tokens"] > 0:
                    assert stats["mean_tokens"] >= 50  # minimum token count in our fixture


# ---------------------------------------------------------------------------
# Tests: analyze_diversity
# ---------------------------------------------------------------------------


class TestAnalyzeDiversity:
    def test_basic_computation(self):
        all_data = _make_all_data_2models_2tasks()
        matrix = build_task_matrix(all_data)
        result = analyze_diversity(matrix)

        for method in ALL_METHODS:
            assert method in result
            assert "mean_diversity_ratio" in result[method]
            assert "task_count" in result[method]

    def test_diversity_ratio_range(self):
        """Diversity ratio should be between 0 and 1."""
        all_data = _make_all_data_2models_2tasks()
        matrix = build_task_matrix(all_data)
        result = analyze_diversity(matrix)

        for method in ALL_METHODS:
            ratio = result[method]["mean_diversity_ratio"]
            assert 0 <= ratio <= 1.0, f"{method} ratio {ratio} out of [0,1]"

    def test_zero_correct_excluded(self):
        """Tasks with 0 correct should not contribute to diversity ratio."""
        # Create data where one method has 0 correct
        all_data = _make_all_data_2models_2tasks()
        # Override: p_less gets 0 correct on task 1 for both models
        for model in all_data:
            all_data[model]["metrics"]["p_less"]["per_task"][1]["num_correct"] = 0
            all_data[model]["metrics"]["p_less"]["per_task"][1]["num_distinct_correct"] = 0

        matrix = build_task_matrix(all_data)
        result = analyze_diversity(matrix)

        # p_less should have fewer contributing tasks than greedy
        assert result["p_less"]["task_count"] < result["greedy"]["task_count"]

    def test_perfect_diversity(self):
        """When num_distinct_correct == num_correct, ratio should be 1.0."""
        all_data = _make_all_data_2models_2tasks()
        # Set num_distinct_correct = num_correct for greedy across all tasks/models
        for model in all_data:
            for task_entry in all_data[model]["metrics"]["greedy"]["per_task"]:
                task_entry["num_distinct_correct"] = task_entry["num_correct"]

        matrix = build_task_matrix(all_data)
        result = analyze_diversity(matrix)

        assert result["greedy"]["mean_diversity_ratio"] == pytest.approx(1.0)
