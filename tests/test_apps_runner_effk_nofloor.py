"""bench.apps runner: `--method pless_effk_nofloor` (vLLM-only p-less variant) for the CF-748 pipeline.

Contract (Option A, notes/effk_nofloor_cf748_plan.md):
  * `pless_effk_nofloor` is an accepted --method; the pre-existing choices are unchanged (same order).
  * For --backend vllm the variant is spliced into bench.generator_vllm._SAMPLER_LOGIT_FN via
    bench.pless_variants_vllm.register() BEFORE load_engine builds the engine (under fork the engine
    core inherits the parent's dict as it is at engine construction time).
  * register() is NOT called for any pre-existing method.
  * --backend hf with the variant is rejected at parse time with a clear message.
  * --seed reaches every task's SamplingParams on the variant path (real generator_vllm code, fake vllm).

No vLLM / GPU needed: a fake `vllm` package is installed in sys.modules for the end-to-end test.
"""

from __future__ import annotations

import enum
import json
import sys
import types

import pytest

VARIANT = "pless_effk_nofloor"
BASE = ["bench.apps", "--model", "Qwen/Qwen3-8B", "--source", "CODEFORCES", "--difficulty", "interview",
        "--temperature", "1.0", "--enable-thinking"]
OLD_CHOICES_TAIL = ["temp", "split", "pless_alpha", "pless_renyi"]


class _Stop(Exception):
    pass


def _method_choices():
    from bench.apps.runner import _build_argparser
    p = _build_argparser()
    act = next(a for a in p._actions if a.dest == "method")
    return list(act.choices)


# ---------------------------------------------------------------- parser
def test_old_choices_unchanged_and_variant_appended():
    from bench.sampler_bridge import SAMPLERS
    choices = _method_choices()
    old = list(SAMPLERS.keys()) + OLD_CHOICES_TAIL
    assert choices[:len(old)] == old
    assert choices[len(old):] == [VARIANT]


def test_only_the_needed_variant_is_exposed():
    # bench.pless_variants_vllm imports torch at module import time, so the runner does not derive
    # its choices from VARIANTS_VLLM; only the one name needed for this experiment is added.
    choices = _method_choices()
    for other in ("pless_effk", "pless_min2", "pless_topmass", "pless_gini_nofloor"):
        assert other not in choices


def test_parser_accepts_variant_with_vllm(monkeypatch):
    monkeypatch.setattr(sys, "argv", BASE + ["--backend", "vllm", "--method", VARIANT, "--seed", "42"])
    from bench.apps.runner import parse_args
    a = parse_args()
    assert a.method == VARIANT and a.seed == 42 and a.backend == "vllm"


def test_hf_backend_with_variant_rejected_clearly(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", BASE + ["--backend", "hf", "--method", VARIANT])
    from bench.apps.runner import parse_args
    with pytest.raises(SystemExit) as ei:
        parse_args()
    assert ei.value.code == 2
    err = capsys.readouterr().err
    assert VARIANT in err and "--backend vllm" in err


def test_hf_is_default_backend_so_variant_without_backend_rejected(monkeypatch):
    monkeypatch.setattr(sys, "argv", BASE + ["--method", VARIANT])
    from bench.apps.runner import parse_args
    with pytest.raises(SystemExit):
        parse_args()


def test_method_key_for_variant(monkeypatch):
    monkeypatch.setattr(sys, "argv", BASE + ["--backend", "vllm", "--method", VARIANT])
    from bench.apps.runner import _method_key, parse_args
    assert _method_key(parse_args()) == f"{VARIANT}_think_t1.0"


# ---------------------------------------------------------------- register() ordering (fake modules)
def _fake_effk_fn(logits):
    return logits


def _install_fakes(monkeypatch, events, register_adds=True):
    table: dict = {}

    def fake_load_engine(model, **kw):
        events.append("load_engine")
        raise _Stop

    def fake_register():
        events.append("register")
        if register_adds:
            table[VARIANT] = _fake_effk_fn

    monkeypatch.setitem(sys.modules, "bench.generator_vllm",
                        types.SimpleNamespace(load_engine=fake_load_engine, _SAMPLER_LOGIT_FN=table))
    monkeypatch.setitem(sys.modules, "bench.pless_variants_vllm",
                        types.SimpleNamespace(register=fake_register))


def _run_main(monkeypatch, tmp_path, argv_extra):
    monkeypatch.setattr(sys, "argv", BASE + ["--results-dir", str(tmp_path)] + argv_extra)
    from bench.apps import runner
    with pytest.raises(_Stop):
        runner.main()


def test_register_called_before_load_engine_for_variant(monkeypatch, tmp_path):
    events: list[str] = []
    _install_fakes(monkeypatch, events)
    _run_main(monkeypatch, tmp_path, ["--backend", "vllm", "--method", VARIANT])
    assert events == ["register", "load_engine"]


def test_variant_not_registered_fails_loud_before_engine(monkeypatch, tmp_path):
    """Guard against a silent fallback: if register() did not install the variant, stop before load_engine."""
    events: list[str] = []
    _install_fakes(monkeypatch, events, register_adds=False)
    monkeypatch.setattr(sys, "argv", BASE + ["--results-dir", str(tmp_path), "--backend", "vllm", "--method", VARIANT])
    from bench.apps import runner
    with pytest.raises(SystemExit) as ei:
        runner.main()
    assert VARIANT in str(ei.value) and "not registered" in str(ei.value)
    assert events == ["register"]


@pytest.mark.parametrize("extra", [
    ["--method", "pless"],
    ["--method", "pless_norm"],
    ["--method", "pless_renyi", "--renyi-k", "0.25"],
    ["--method", "pless_alpha", "--alpha", "3"],
    ["--method", "temp", "--top-p", "0.9", "--top-k", "0"],
    ["--method", "split", "--sampler-think", "pless", "--temp-think", "1.0",
     "--sampler-code", "pless", "--temp-code", "1.0"],
])
def test_register_not_called_for_existing_methods(monkeypatch, tmp_path, extra):
    events: list[str] = []
    _install_fakes(monkeypatch, events)
    _run_main(monkeypatch, tmp_path, ["--backend", "vllm"] + extra)
    assert events == ["load_engine"]


# ---------------------------------------------------------------- end to end with real generator_vllm + fake vllm
class _FakeTok:
    def encode(self, s, add_special_tokens=False):
        return [151668] if s == "</think>" else [ord(c) for c in s]

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(i) for i in ids)


def _fake_vllm(monkeypatch, state):
    vllm = types.ModuleType("vllm")
    sp_mod = types.ModuleType("vllm.sampling_params")
    v1 = types.ModuleType("vllm.v1")
    v1s = types.ModuleType("vllm.v1.sample")
    lp = types.ModuleType("vllm.v1.sample.logits_processor")

    class SamplingParams:
        def __init__(self, **kw):
            self.__dict__.update(kw)
            self.kw = kw

    class LogitsProcessor:
        pass

    class MoveDirectionality(enum.Enum):
        UNIDIRECTIONAL = 0
        SWAP = 1

    class _Out:
        def __init__(self, text):
            self.text = text
            self.token_ids = [ord(c) for c in text]

    class LLM:
        def __init__(self, **kw):
            from bench import generator_vllm as gv
            # Snapshot of the dispatch table at engine construction (what a forked engine core inherits).
            state["dispatch_at_engine_init"] = set(gv._SAMPLER_LOGIT_FN)
            state["llm_kwargs"] = kw

        def get_tokenizer(self):
            return _FakeTok()

        def generate(self, prompts, sp, use_tqdm=False):
            # Emulate vLLM's per-request validation of the custom processor.
            for cls in state["llm_kwargs"]["logits_processors"]:
                cls.validate_params(sp)
            state["sps"].append(sp)
            if prompts[0] in {f"P{pid}" for pid in state.get("fail_pids", ())}:
                # Emulates the engine core lacking the variant (e.g. spawned child): KeyError per request.
                raise KeyError("pless_effk_nofloor")
            return [types.SimpleNamespace(outputs=[_Out("t</think>\nprint(1)") for _ in range(sp.n)])]

    vllm.SamplingParams, vllm.LLM = SamplingParams, LLM
    vllm.TokensPrompt = lambda prompt_token_ids: {"prompt_token_ids": prompt_token_ids}
    sp_mod.SamplingParams = SamplingParams
    lp.LogitsProcessor, lp.MoveDirectionality = LogitsProcessor, MoveDirectionality
    for name, mod in (("vllm", vllm), ("vllm.sampling_params", sp_mod), ("vllm.v1", v1),
                      ("vllm.v1.sample", v1s), ("vllm.v1.sample.logits_processor", lp)):
        monkeypatch.setitem(sys.modules, name, mod)


@pytest.fixture
def e2e(monkeypatch, tmp_path):
    from bench import generator_vllm as gv
    from bench.apps import runner
    state: dict = {"sps": [], "pids": (11, 22, 33)}
    # Isolate module-level state that the fake run mutates.
    monkeypatch.setattr(gv, "_SAMPLER_LOGIT_FN", dict(gv._SAMPLER_LOGIT_FN))
    monkeypatch.setattr(gv._build_pless_split_logits_processor_class, "_cached", None)
    monkeypatch.setattr(gv, "PlessSplitLogitsProcessor", None, raising=False)
    _fake_vllm(monkeypatch, state)
    def problems(**kw):
        return iter([types.SimpleNamespace(problem_id=pid, source="CODEFORCES", difficulty="interview",
                                           question=f"q{pid}") for pid in state["pids"]])
    monkeypatch.setattr(runner, "load_apps", problems)
    monkeypatch.setattr(runner, "format_prompt_apps_instruct",
                        lambda problem, tok, enable_thinking=False: (f"P{problem.problem_id}", ""))

    def run_raw(extra):
        monkeypatch.setattr(sys, "argv", BASE + ["--backend", "vllm", "--results-dir", str(tmp_path),
                                                 "--n-samples", "10"] + extra)
        runner.main()

    state["run_raw"] = run_raw
    state["out_dir"] = tmp_path / "Qwen--Qwen3-8B" / "CODEFORCES_interview"

    def run(extra):
        run_raw(extra)
        out = tmp_path / "Qwen--Qwen3-8B" / "CODEFORCES_interview"
        files = sorted(out.glob("*.jsonl"))
        assert len(files) == 1, files
        recs = [json.loads(l) for l in files[0].read_text().splitlines() if l.strip()]
        return files[0], recs

    return state, run, gv


def test_e2e_seed_reaches_every_task_sampling_params(e2e, capsys):
    state, run, gv = e2e
    path, recs = run(["--method", VARIANT, "--seed", "42"])
    out = capsys.readouterr().out
    # Startup log line naming the logits function the vLLM dispatch table resolves for this method.
    assert ("[sampler] method=pless_effk_nofloor -> logits fn "
            "bench.pless_variants_vllm.pless_effk_nofloor_mask_logits") in out
    assert out.index("[sampler]") < out.index("Completed problem_id=")
    assert path.name == f"{VARIANT}_think_t1.0_t1.0.jsonl"
    assert VARIANT in state["dispatch_at_engine_init"]          # registered before engine init
    assert len(state["sps"]) == 3                               # one request per task
    for sp in state["sps"]:
        assert sp.seed == 42
        assert sp.n == 10 and sp.max_tokens == 8192
        assert (sp.temperature, sp.top_p, sp.top_k) == (1.0, 1.0, -1)
        cfg = sp.extra_args["pless_split"]
        assert cfg["sampler_think"] == cfg["sampler_code"] == VARIANT
        assert cfg["t_think"] == cfg["t_code"] == 1.0
    assert [r["task_id"] for r in recs] == [11, 22, 33]
    for r in recs:
        assert r["method"] == VARIANT and r["backend"] == "vllm" and r["temperature"] == 1.0
        assert len(r["samples"]) == 10 and len(r["samples_with_thinking"]) == 10
        # Documented: the output record does NOT store the seed (schema unchanged) -> launcher manifest logs it.
        assert "seed" not in r


def test_e2e_unseeded_variant_has_no_seed(e2e):
    state, run, _ = e2e
    run(["--method", VARIANT])
    assert len(state["sps"]) == 3
    assert all("seed" not in sp.kw for sp in state["sps"])


def test_e2e_existing_method_does_not_register_variants(e2e, capsys):
    state, run, _ = e2e
    path, recs = run(["--method", "pless_renyi", "--renyi-k", "0.25"])
    assert "[sampler]" not in capsys.readouterr().out   # existing methods' stdout unchanged
    assert path.name == "pless_renyi_think_t1.0_k0.25_t1.0.jsonl"
    assert VARIANT not in state["dispatch_at_engine_init"]
    assert all("seed" not in sp.kw for sp in state["sps"])
    assert all(r["method"] == "pless_renyi" for r in recs)


def test_validate_params_rejects_variant_without_register(e2e):
    """Why register() must precede the engine: the processor validates sampler names against the table."""
    state, _, gv = e2e
    cls = gv._build_pless_split_logits_processor_class()
    sp = types.SimpleNamespace(extra_args={"pless_split": {
        "t_think": 1.0, "t_code": 1.0, "sampler_think": VARIANT, "sampler_code": VARIANT}})
    with pytest.raises(ValueError):
        cls.validate_params(sp)
    from bench.pless_variants_vllm import register
    register()
    cls.validate_params(sp)  # no raise


# ---------------------------------------------------------------- fail-loud on task exceptions (variant only)
def _records(state):
    files = sorted(state["out_dir"].glob("*.jsonl"))
    return [json.loads(l) for f in files for l in f.read_text().splitlines() if l.strip()]


def test_variant_aborts_after_three_consecutive_task_errors(e2e, capsys):
    state, _, _ = e2e
    state["pids"] = (11, 22, 33, 44, 55)
    state["fail_pids"] = {11, 22, 33, 44, 55}
    with pytest.raises(SystemExit) as ei:
        state["run_raw"](["--method", VARIANT, "--seed", "42"])
    assert ei.value.code not in (0, None)
    assert "3 consecutive" in str(ei.value)
    assert len(state["sps"]) == 3          # stopped at the 3rd failure, tasks 44/55 never attempted
    assert _records(state) == []


def test_variant_consecutive_counter_resets_on_success(e2e):
    state, _, _ = e2e
    state["pids"] = (11, 22, 33, 44, 55, 66)
    state["fail_pids"] = {11, 22, 44, 55}   # never 3 in a row
    state["run_raw"](["--method", VARIANT])
    assert [r["task_id"] for r in _records(state)] == [33, 66]


def test_variant_zero_records_written_exits_nonzero(e2e):
    state, _, _ = e2e
    state["pids"] = (11, 22)
    state["fail_pids"] = {11, 22}           # 2 failures < 3, but nothing written this session
    with pytest.raises(SystemExit) as ei:
        state["run_raw"](["--method", VARIANT])
    assert ei.value.code not in (0, None)
    assert "0 records" in str(ei.value)


def test_variant_fully_resumed_session_exits_zero(e2e):
    state, run, _ = e2e
    run(["--method", VARIANT])
    n = len(state["sps"])
    state["run_raw"](["--method", VARIANT])  # nothing remaining -> no attempt, no error
    assert len(state["sps"]) == n and len(_records(state)) == 3


@pytest.mark.parametrize("extra", [["--method", "pless_renyi", "--renyi-k", "0.25"], ["--method", "pless"]])
def test_existing_methods_keep_swallowing_task_errors(e2e, extra):
    state, _, _ = e2e
    state["pids"] = (11, 22, 33, 44, 55)
    state["fail_pids"] = {11, 22, 33, 44, 55}
    state["run_raw"](extra)                  # unchanged behaviour: returns normally
    assert len(state["sps"]) == 5 and _records(state) == []
