import sys

import pytest


def test_mbpp_runner_top_p_requires_top_p_flag(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["bench", "--model", "x", "--method", "top_p"])
    from bench.runner import parse_args
    with pytest.raises(SystemExit) as exc:
        parse_args()
    assert exc.value.code == 2


def test_humaneval_runner_top_p_requires_top_p_flag(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["bench.humaneval", "--model", "x", "--method", "top_p"])
    from bench.humaneval.runner import parse_args
    with pytest.raises(SystemExit) as exc:
        parse_args()
    assert exc.value.code == 2


def test_mbpp_runner_accepts_top_p_method(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["bench", "--model", "x", "--method", "top_p", "--top-p", "0.9"])
    from bench.runner import parse_args
    args = parse_args()
    assert args.method == "top_p"
    assert args.top_p == 0.9


def test_humaneval_runner_accepts_top_p_method(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["bench.humaneval", "--model", "x", "--method", "top_p", "--top-p", "0.9"])
    from bench.humaneval.runner import parse_args
    args = parse_args()
    assert args.method == "top_p"
    assert args.top_p == 0.9
