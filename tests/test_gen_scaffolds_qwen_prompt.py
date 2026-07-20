"""The Qwen self-scaffold prompt must ask for an algorithm with thinking ON."""
from __future__ import annotations


def _make_problem():
    from bench.apps.dataset import AppsProblem
    return AppsProblem(problem_id=1, source="ATCODER", difficulty="interview",
                       question="Compute the running maximum of an array.",
                       starter_code="", fn_name=None)


class _StubTokenizer:
    def __init__(self):
        self.last_kwargs = None

    def apply_chat_template(self, messages, tokenize=False,
                            add_generation_prompt=True, **kw):
        self.last_kwargs = kw
        self._roles = {m["role"]: m["content"] for m in messages}
        return "\n".join(f"<|{m['role']}|>{m['content']}" for m in messages)


def test_prompt_uses_scaffold_system_and_thinking_on():
    from bench.apps.gen_scaffolds import SYSTEM_PROMPT
    from bench.apps.gen_scaffolds_qwen import _build_scaffold_prompt
    tok = _StubTokenizer()
    out = _build_scaffold_prompt(tok, _make_problem())
    assert tok.last_kwargs.get("enable_thinking") is True          # thinking ON
    assert tok._roles["system"].startswith(SYSTEM_PROMPT[:40])     # same base prompt
    assert "FORBIDDEN" in tok._roles["system"]                     # no-code instruction
    assert "running maximum" in tok._roles["user"]                 # the problem


def test_extra_system_is_appended_not_replacing():
    from bench.apps.gen_scaffolds import SYSTEM_PROMPT
    from bench.apps.gen_scaffolds_qwen import _build_scaffold_prompt
    tok = _StubTokenizer()
    _build_scaffold_prompt(tok, _make_problem(), extra_system="Be extra concise.")
    sysmsg = tok._roles["system"]
    assert SYSTEM_PROMPT[:40] in sysmsg and "Be extra concise." in sysmsg
