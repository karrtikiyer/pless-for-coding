import pytest

from bench.eval.metrics import add_distinct_counts, compute_cover_at_t, compute_pass_at_k


# --- compute_pass_at_k ---

def test_pass_at_k_perfect_score():
    """All samples correct -> pass@k should be 1.0 for all k."""
    results = [
        {"task_id": 1, "num_correct": 10, "pass_results": [True] * 10},
        {"task_id": 2, "num_correct": 10, "pass_results": [True] * 10},
    ]
    pak = compute_pass_at_k(results, [1, 5, 10])
    assert pak["1"] == pytest.approx(1.0)
    assert pak["5"] == pytest.approx(1.0)
    assert pak["10"] == pytest.approx(1.0)


def test_pass_at_k_zero_score():
    """No samples correct -> pass@k should be 0.0 for all k."""
    results = [
        {"task_id": 1, "num_correct": 0, "pass_results": [False] * 10},
        {"task_id": 2, "num_correct": 0, "pass_results": [False] * 10},
    ]
    pak = compute_pass_at_k(results, [1, 5, 10])
    assert pak["1"] == pytest.approx(0.0)
    assert pak["5"] == pytest.approx(0.0)
    assert pak["10"] == pytest.approx(0.0)


def test_pass_at_k_monotonic():
    """pass@k should increase (or stay same) as k increases."""
    results = [
        {"task_id": i, "num_correct": 3, "pass_results": [True]*3 + [False]*7}
        for i in range(20)
    ]
    pak = compute_pass_at_k(results, [1, 5, 10])
    assert pak["1"] <= pak["5"] <= pak["10"]


def test_pass_at_k_skips_too_large_k():
    """If k > n_samples, that k should be skipped."""
    results = [
        {"task_id": 1, "num_correct": 2, "pass_results": [True, True, False]},
    ]
    pak = compute_pass_at_k(results, [1, 3, 5])
    assert "1" in pak
    assert "3" in pak
    assert "5" not in pak  # only 3 samples, can't compute pass@5


# --- compute_cover_at_t ---

def test_cover_at_t_basic():
    """10 samples per task, fractional thresholds."""
    results = [
        {"task_id": 1, "num_correct": 5, "num_distinct_correct": 3},
        {"task_id": 2, "num_correct": 2, "num_distinct_correct": 2},
        {"task_id": 3, "num_correct": 0, "num_distinct_correct": 0},
        {"task_id": 4, "num_correct": 10, "num_distinct_correct": 7},
    ]
    cover, cover_distinct = compute_cover_at_t(results, [0.1, 0.3, 0.5, 1.0], num_samples_per_task=10)

    # Non-distinct: % of tasks where num_correct >= t * 10
    # t=0.1 -> need >=1: tasks 1,2,4 -> 75%
    assert cover["0.1"] == pytest.approx(75.0)
    # t=0.3 -> need >=3: tasks 1,4 -> 50%
    assert cover["0.3"] == pytest.approx(50.0)
    # t=0.5 -> need >=5: tasks 1,4 -> 50%
    assert cover["0.5"] == pytest.approx(50.0)
    # t=1.0 -> need >=10: task 4 -> 25%
    assert cover["1.0"] == pytest.approx(25.0)

    # Distinct: % of tasks where num_distinct_correct >= t * 10
    # t=0.1 -> need >=1: tasks 1,2,4 -> 75%
    assert cover_distinct["0.1"] == pytest.approx(75.0)
    # t=0.3 -> need >=3: tasks 1,4 -> 50%
    assert cover_distinct["0.3"] == pytest.approx(50.0)
    # t=0.5 -> need >=5: task 4 -> 25%
    assert cover_distinct["0.5"] == pytest.approx(25.0)
    # t=1.0 -> need >=10: none -> 0%
    assert cover_distinct["1.0"] == pytest.approx(0.0)


def test_cover_at_t_decreasing():
    """cover@t should decrease (or stay same) as t increases."""
    results = [
        {"task_id": i, "num_correct": i, "num_distinct_correct": i}
        for i in range(11)
    ]
    t_vals = [0.1, 0.3, 0.5, 1.0]
    cover, cover_distinct = compute_cover_at_t(results, t_vals, num_samples_per_task=10)
    vals = [cover[str(t)] for t in t_vals]
    assert vals == sorted(vals, reverse=True)


def test_cover_at_t_distinct_leq_nondistinct():
    """Distinct percentages should be <= non-distinct percentages."""
    results = [
        {"task_id": 1, "num_correct": 5, "num_distinct_correct": 3},
        {"task_id": 2, "num_correct": 8, "num_distinct_correct": 4},
    ]
    t_vals = [0.1, 0.3, 0.5, 0.8]
    cover, cover_distinct = compute_cover_at_t(results, t_vals, num_samples_per_task=10)
    for t in t_vals:
        assert cover_distinct[str(t)] <= cover[str(t)]


# --- add_distinct_counts ---

def test_add_distinct_counts_basic():
    records = [
        {
            "task_id": 1,
            "samples": [
                "def f(x): return x + 1",      # correct, unique structure A
                "def g(y): return y + 1",        # correct, same structure as A
                "def h(z): return z * 2",        # correct, unique structure B
                "def broken(: return 0",          # incorrect (syntax error)
            ],
        }
    ]
    task_results = [
        {"task_id": 1, "num_correct": 3, "pass_results": [True, True, True, False]},
    ]
    add_distinct_counts(task_results, records)
    # Samples 0 and 1 have same structure, sample 2 is different
    assert task_results[0]["num_distinct_correct"] == 2


def test_add_distinct_counts_all_same():
    records = [
        {
            "task_id": 1,
            "samples": [
                "def a(x): return x",
                "def b(y): return y",
                "def c(z): return z",
            ],
        }
    ]
    task_results = [
        {"task_id": 1, "num_correct": 3, "pass_results": [True, True, True]},
    ]
    add_distinct_counts(task_results, records)
    assert task_results[0]["num_distinct_correct"] == 1


def test_add_distinct_counts_none_correct():
    records = [
        {
            "task_id": 1,
            "samples": ["def f(): return 1", "def g(): return 2"],
        }
    ]
    task_results = [
        {"task_id": 1, "num_correct": 0, "pass_results": [False, False]},
    ]
    add_distinct_counts(task_results, records)
    assert task_results[0]["num_distinct_correct"] == 0


def test_add_distinct_counts_syntax_error_in_correct():
    """A sample that passes tests but has a syntax error when parsed standalone
    (unlikely but possible if the test builder wraps it differently).
    The fingerprint should be None and not counted as distinct."""
    records = [
        {
            "task_id": 1,
            "samples": [
                "def f(x): return x + 1",
                "def f(: broken",  # won't parse
            ],
        }
    ]
    task_results = [
        {"task_id": 1, "num_correct": 2, "pass_results": [True, True]},
    ]
    add_distinct_counts(task_results, records)
    # Only the parseable one counts as distinct
    assert task_results[0]["num_distinct_correct"] == 1
