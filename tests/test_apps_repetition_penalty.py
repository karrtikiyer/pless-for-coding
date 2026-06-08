"""APPS runner: --repetition-penalty flag (provider-faithful standard-decoder
baselines — Qwen2.5-Coder ships rep_penalty 1.1/1.05). pless stays knob-free."""

from bench.apps.runner import _build_argparser


def _parse(argv):
    return _build_argparser().parse_args(argv)


_BASE = ["--model", "Qwen/Qwen2.5-Coder-7B-Instruct",
         "--source", "ATCODER", "--difficulty", "interview", "--method", "temp"]


def test_repetition_penalty_default_is_1():
    args = _parse(_BASE)
    assert args.repetition_penalty == 1.0


def test_repetition_penalty_parses():
    args = _parse(_BASE + ["--repetition-penalty", "1.1"])
    assert args.repetition_penalty == 1.1
