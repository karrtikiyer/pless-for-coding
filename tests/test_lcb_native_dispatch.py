"""Regression tests for the native-vs-custom-processor sampling dispatch in
the ``run_pless_variants_*_lcb_cuda.py`` family of scripts.

Covers two things found/fixed 2026-10-26 (see CLAUDE.md Known Gotchas #8 and
docs/theory/todos.md E21):

1. ``_NATIVE_TOP_P_TOP_K`` must stay in sync with each script's ``CONFIGS``
   list -- every non-pless/non-top_nsigma sampler_name must be routed through
   vLLM's native sampling (``generate_samples_standard_vllm``), not the
   custom ``PlessSplitLogitsProcessor``. A silent coverage gap here is
   exactly how the original nucleus-sampling bug shipped undetected.

2. ``_all_configs_native`` must correctly decide whether to skip registering
   the custom logits processor at engine construction. vllm-metal (Apple
   Silicon) refuses ANY custom logits processor outright at ``LLM()``
   construction -- before any per-task code runs -- regardless of whether
   the processor would actually be invoked for the selected configs. A
   wrong answer here either (a) still crashes on vllm-metal for a pure
   native-sampling run, or (b) silently skips the processor for a run that
   genuinely needs it (pless/top_nsigma), which would be a correctness bug,
   not just a compatibility one.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent

SCRIPTS = [
    "run_pless_variants_qwen3_8b_lcb_cuda.py",
    "run_pless_variants_ministral_3b_lcb_cuda.py",
    "run_pless_variants_deepseek_8b_lcb_cuda.py",
]

# sampler_name prefixes/names with no vLLM-native equivalent -- these must
# NEVER appear in _NATIVE_TOP_P_TOP_K. pless* is this project's own method;
# top_nsigma has no native vLLM sampler (independently verified against the
# paper + two reference implementations, see todos.md E21).
_CUSTOM_ONLY_PREFIXES = ("pless",)
_CUSTOM_ONLY_EXACT = ("top_nsigma", "top_nsigma_1_5")


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / name)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(params=SCRIPTS)
def script(request):
    return _load_script(request.param)


def test_native_dict_covers_every_non_custom_sampler_name(script):
    """Every CONFIGS sampler_name is EITHER in _NATIVE_TOP_P_TOP_K OR is a
    known custom-only method (pless*/top_nsigma*) -- no sampler_name should
    silently fall through uncategorized (which would mean it's routed
    through the custom processor without anyone having decided that's
    correct for it)."""
    config_by_name = {c[0]: c for c in script.CONFIGS}
    for config_name, (name, sampler_name, _temp, _renyi_k) in [
        (c[0], c) for c in script.CONFIGS
    ]:
        is_native = sampler_name in script._NATIVE_TOP_P_TOP_K
        is_known_custom = sampler_name in _CUSTOM_ONLY_EXACT or any(
            sampler_name.startswith(p) for p in _CUSTOM_ONLY_PREFIXES
        )
        assert is_native or is_known_custom, (
            f"{config_name!r} (sampler_name={sampler_name!r}) is neither in "
            f"_NATIVE_TOP_P_TOP_K nor a recognized custom-only method -- "
            f"decide which it should be and update this test's allowlist "
            f"or the script's _NATIVE_TOP_P_TOP_K dict."
        )
    assert config_by_name  # sanity: CONFIGS isn't empty


def test_native_dict_never_contains_custom_only_sampler_names(script):
    """The reverse mistake: pless*/top_nsigma* must never end up in
    _NATIVE_TOP_P_TOP_K (that would silently route a genuinely custom
    method through vLLM's native sampler, producing wrong results)."""
    for sampler_name in script._NATIVE_TOP_P_TOP_K:
        assert sampler_name not in _CUSTOM_ONLY_EXACT
        assert not any(sampler_name.startswith(p) for p in _CUSTOM_ONLY_PREFIXES)


def test_all_configs_native_true_for_pure_native_selection(script):
    """A --configs selection containing ONLY native-eligible sampler_names
    must report True (so engine construction skips the custom processor --
    required for vllm-metal to work at all for such a run)."""
    config_by_name = {c[0]: c for c in script.CONFIGS}
    native_config_names = [
        c[0] for c in script.CONFIGS if c[1] in script._NATIVE_TOP_P_TOP_K
    ]
    assert native_config_names, "expected at least one native config to exist"
    assert script._all_configs_native(native_config_names, config_by_name) is True


def test_all_configs_native_false_for_pure_custom_selection(script):
    """A --configs selection containing ONLY pless/top_nsigma configs must
    report False (the custom processor must stay registered)."""
    config_by_name = {c[0]: c for c in script.CONFIGS}
    custom_config_names = [
        c[0] for c in script.CONFIGS if c[1] not in script._NATIVE_TOP_P_TOP_K
    ]
    assert custom_config_names, "expected at least one custom config to exist"
    assert script._all_configs_native(custom_config_names, config_by_name) is False


def test_all_configs_native_false_for_mixed_selection(script):
    """A --configs selection mixing one native and one custom config must
    report False -- the custom processor is needed for the custom config,
    so it must stay registered even though a native config is also present
    (the native config's own generation is unaffected either way, since it
    never consults the processor regardless of whether it's registered)."""
    config_by_name = {c[0]: c for c in script.CONFIGS}
    native_name = next(c[0] for c in script.CONFIGS if c[1] in script._NATIVE_TOP_P_TOP_K)
    custom_name = next(c[0] for c in script.CONFIGS if c[1] not in script._NATIVE_TOP_P_TOP_K)
    assert script._all_configs_native([native_name, custom_name], config_by_name) is False


def test_all_configs_native_true_for_single_native_config(script):
    """Exactly the scenario this was built for: a single-config smoke test
    like --configs top_p_0_9 on vllm-metal must not register the custom
    processor."""
    config_by_name = {c[0]: c for c in script.CONFIGS}
    for config_name, (_, sampler_name, _temp, _renyi_k) in [
        (c[0], c) for c in script.CONFIGS
    ]:
        if sampler_name in script._NATIVE_TOP_P_TOP_K:
            assert script._all_configs_native([config_name], config_by_name) is True
            return
    pytest.fail("no native config found to test")


# ---------------------------------------------------------------------------
# Ministral-only pure top-k baselines (top_k in {25, 20, 15}, T=1.0, top_p=1.0)
# -- must route through vLLM's native SamplingParams (CLAUDE.md Known
# Gotchas #8), never the custom PlessSplitLogitsProcessor.
# ---------------------------------------------------------------------------
_MINISTRAL = "run_pless_variants_ministral_3b_lcb_cuda.py"
_TOP_K_CONFIGS = {"top_k_25": 25, "top_k_20": 20, "top_k_15": 15}

# Snapshot of the Ministral CONFIGS names BEFORE the top-k addition: every one
# of these had a hard-coded record "top_k": 0, so _record_top_k must return 0
# for all of them (=> byte-identical records for existing configs).
_PRE_TOPK_MINISTRAL_CONFIGS = (
    "top_nsigma", "top_nsigma_1_5", "gk_0_1", "gk_0_35", "gk_0_025", "gk_0_0325",
    "gk_0_05", "gk_0_075", "gk_0_2", "gk_0_4", "gk_0_8", "gk_1_6",
    "ministral_recipe", "pless_default", "pless_effk", "pless_effk_nofloor",
    "pless_min2", "pless_topmass", "pless_topmass_nofloor", "pless_gini",
    "pless_gini_nofloor", "pless_top1half", "pless_default_t1_5",
    "pless_default_t2_0", "pless_default_t2_5", "pless_default_t3_0",
    "top_p_1_0", "top_p_0_95", "top_p_0_9", "top_p_0_85", "top_p_0_8",
    "top_p_0_75", "top_p_0_7",
)


@pytest.fixture(scope="module")
def ministral():
    return _load_script(_MINISTRAL)


@pytest.mark.parametrize("name,k", sorted(_TOP_K_CONFIGS.items()))
def test_ministral_top_k_config_row(ministral, name, k):
    rows = {c[0]: c for c in ministral.CONFIGS}
    assert rows[name] == (name, name, 1.0, None)
    assert name in ministral.CONFIG_NAMES


@pytest.mark.parametrize("name,k", sorted(_TOP_K_CONFIGS.items()))
def test_ministral_top_k_is_native_exact_tuple(ministral, name, k):
    assert ministral._NATIVE_TOP_P_TOP_K[name] == (1.0, k)


def test_ministral_top_k_not_custom_processor(ministral):
    config_by_name = {c[0]: c for c in ministral.CONFIGS}
    names = list(_TOP_K_CONFIGS)
    assert ministral._all_configs_native(names, config_by_name) is True
    for n in names:
        assert ministral._all_configs_native([n], config_by_name) is True


def test_ministral_pre_topk_configs_unchanged(ministral):
    """Existing configs are all still present, in the same order, unchanged
    count (only the 3 top-k rows + the gk_0_45, gk_0_25, gk_0_5 rows added)."""
    added = set(_TOP_K_CONFIGS) | {"gk_0_45", "gk_0_25", "gk_0_5"}
    assert tuple(n for n in ministral.CONFIG_NAMES if n not in added) \
        == _PRE_TOPK_MINISTRAL_CONFIGS
    assert len(ministral.CONFIG_NAMES) == len(_PRE_TOPK_MINISTRAL_CONFIGS) + 6
    assert len(set(ministral.CONFIG_NAMES)) == len(ministral.CONFIG_NAMES)


@pytest.mark.parametrize("name,k", sorted(_TOP_K_CONFIGS.items()))
def test_ministral_record_top_k_new_configs(ministral, name, k):
    assert ministral._record_top_k(name) == k


@pytest.mark.parametrize("name", _PRE_TOPK_MINISTRAL_CONFIGS)
def test_ministral_record_top_k_existing_configs_is_zero(ministral, name):
    """Existing configs keep the old hard-coded record value top_k=0."""
    sampler_name = {c[0]: c[1] for c in ministral.CONFIGS}[name]
    assert ministral._record_top_k(sampler_name) == 0


def test_ministral_top_k_output_path_no_collision(ministral, tmp_path):
    p = ministral._output_path(tmp_path, "top_k_25", "atcoder", 1.0)
    assert p.name == "top_k_25_atcoder_think_t1.0.jsonl"
    names = {ministral._output_path(tmp_path, c[0], plat, c[2]).name
             for c in ministral.CONFIGS for plat in ("atcoder", "leetcode")}
    assert len(names) == 2 * len(ministral.CONFIGS)


@pytest.mark.parametrize("name,k", sorted(_TOP_K_CONFIGS.items()))
def test_ministral_top_k_run_takes_native_branch(ministral, monkeypatch, tmp_path, name, k):
    """End-to-end (no vLLM): _run_config_on_platform must call
    generate_samples_standard_vllm(top_p=1.0, top_k=k, temperature=1.0) and
    never generate_samples_vllm; the record carries top_k=k, top_p=1.0."""
    import json
    import time
    from types import SimpleNamespace

    calls = {"standard": [], "custom": []}

    def fake_standard(**kw):
        calls["standard"].append(kw)
        return ["print(1)"] * kw["n_samples"]

    def fake_custom(**kw):
        calls["custom"].append(kw)
        return ["print(1)"] * kw["n_samples"]

    monkeypatch.setattr(ministral, "generate_samples_standard_vllm", fake_standard)
    monkeypatch.setattr(ministral, "generate_samples_vllm", fake_custom)
    monkeypatch.setattr(ministral, "format_prompt_lcb_instruct",
                        lambda problem, tok, **kw: ("PROMPT", ""))
    monkeypatch.setattr(ministral, "encode_prompt_for_vllm", lambda text, tok: text)

    row = {c[0]: c for c in ministral.CONFIGS}[name]
    problem = SimpleNamespace(task_id="t1", platform="atcoder", difficulty="easy",
                              contest_date="2024-01-01", question="q")
    far = time.time() + 3600
    out = ministral._run_config_on_platform(
        config_name=name, sampler_name=row[1], temperature=row[2], renyi_k=row[3],
        platform="atcoder", problems=[problem], engine=SimpleNamespace(),
        tokenizer=SimpleNamespace(), n_samples=2, max_new_tokens=8,
        out_root=tmp_path, model_id="m", seed=42, per_config_deadline=far,
        global_deadline=far, think_end_marker="[/THINK]",
    )
    assert calls["custom"] == []
    assert len(calls["standard"]) == 1
    kw = calls["standard"][0]
    assert (kw["top_p"], kw["top_k"], kw["temperature"]) == (1.0, k, 1.0)
    rec = json.loads(out.read_text().splitlines()[0])
    assert rec["top_k"] == k and rec["top_p"] == 1.0 and rec["method"] == name
    assert out.name == f"{name}_atcoder_think_t1.0.jsonl"


# ---------------------------------------------------------------------------
# Ministral G_k k=0.45 (custom pless_renyi path, full1046 batches 1-2 on GPU2,
# run alongside gk_0_35). Must stay on the custom PlessSplitLogitsProcessor --
# there is no vLLM-native Renyi threshold.
# ---------------------------------------------------------------------------
def test_ministral_gk_0_45_config_row(ministral):
    rows = {c[0]: c for c in ministral.CONFIGS}
    assert rows["gk_0_45"] == ("gk_0_45", "pless_renyi", 1.0, 0.45)
    assert rows["gk_0_35"] == ("gk_0_35", "pless_renyi", 1.0, 0.35)
    names = ministral.CONFIG_NAMES
    # placed with the Ministral-only G_k block, right after gk_0_4
    assert names.index("gk_0_45") == names.index("gk_0_4") + 1


def test_ministral_gk_0_45_not_native(ministral):
    assert "gk_0_45" not in ministral._NATIVE_TOP_P_TOP_K
    assert "pless_renyi" not in ministral._NATIVE_TOP_P_TOP_K
    config_by_name = {c[0]: c for c in ministral.CONFIGS}
    assert ministral._all_configs_native(["gk_0_45", "gk_0_35"], config_by_name) is False
    assert ministral._all_configs_native(["gk_0_45"], config_by_name) is False
    assert ministral._record_top_k("pless_renyi") == 0


def test_ministral_gk_0_45_output_path(ministral, tmp_path):
    for plat in ("atcoder", "leetcode"):
        assert ministral._output_path(tmp_path, "gk_0_45", plat, 1.0).name \
            == f"gk_0_45_{plat}_think_t1.0.jsonl"
        assert ministral._output_path(tmp_path, "gk_0_35", plat, 1.0).name \
            == f"gk_0_35_{plat}_think_t1.0.jsonl"
    assert ministral._output_path(tmp_path, "gk_0_45", "atcoder", 1.0) \
        != ministral._output_path(tmp_path, "gk_0_4", "atcoder", 1.0)


@pytest.mark.parametrize("name,k", [("gk_0_45", 0.45), ("gk_0_35", 0.35),
                                    ("gk_0_25", 0.25), ("gk_0_5", 0.5)])
def test_ministral_gk_run_takes_custom_branch(ministral, monkeypatch, tmp_path, name, k):
    """End-to-end (no vLLM): G_k configs call generate_samples_vllm with
    sampler_name='pless_renyi', renyi_k=k, T=1.0 and never the native path;
    the record carries renyi_k=k, top_k=0, top_p=1.0."""
    import json
    import time
    from types import SimpleNamespace

    calls = {"standard": [], "custom": []}

    def fake_standard(**kw):
        calls["standard"].append(kw)
        return ["print(1)"] * kw["n_samples"]

    def fake_custom(**kw):
        calls["custom"].append(kw)
        return ["print(1)"] * kw["n_samples"]

    monkeypatch.setattr(ministral, "generate_samples_standard_vllm", fake_standard)
    monkeypatch.setattr(ministral, "generate_samples_vllm", fake_custom)
    monkeypatch.setattr(ministral, "format_prompt_lcb_instruct",
                        lambda problem, tok, **kw: ("PROMPT", ""))
    monkeypatch.setattr(ministral, "encode_prompt_for_vllm", lambda text, tok: text)

    row = {c[0]: c for c in ministral.CONFIGS}[name]
    problem = SimpleNamespace(task_id="t1", platform="leetcode", difficulty="easy",
                              contest_date="2024-01-01", question="q")
    far = time.time() + 3600
    out = ministral._run_config_on_platform(
        config_name=name, sampler_name=row[1], temperature=row[2], renyi_k=row[3],
        platform="leetcode", problems=[problem], engine=SimpleNamespace(),
        tokenizer=SimpleNamespace(), n_samples=2, max_new_tokens=8,
        out_root=tmp_path, model_id="m", seed=42, per_config_deadline=far,
        global_deadline=far, think_end_marker="[/THINK]",
    )
    assert calls["standard"] == []
    assert len(calls["custom"]) == 1
    kw = calls["custom"][0]
    assert (kw["sampler_name"], kw["renyi_k"], kw["temperature"]) == ("pless_renyi", k, 1.0)
    rec = json.loads(out.read_text().splitlines()[0])
    assert rec["renyi_k"] == k and rec["top_k"] == 0 and rec["top_p"] == 1.0
    assert rec["method"] == name and rec["sampler_name"] == "pless_renyi"
    assert out.name == f"{name}_leetcode_think_t1.0.jsonl"


def test_pless_renyi_mask_accepts_k_0_45():
    """bench.generator_vllm._pless_renyi_mask_logits has no k whitelist: k=0.45
    takes the generic (sum p^k)^(1/(k-1)) branch and matches it exactly."""
    torch = pytest.importorskip("torch")
    from bench.generator_vllm import _pless_renyi_mask_logits

    g = torch.Generator().manual_seed(0)
    logits = torch.randn(3, 50, generator=g) * 3
    probs = torch.softmax(logits.float(), dim=-1)
    k = 0.45
    thr = probs.pow(k).sum(dim=-1, keepdim=True).pow(1.0 / (k - 1.0))
    out = _pless_renyi_mask_logits(logits.clone(), k=k)
    expected_keep = probs >= thr
    expected_keep[torch.arange(3), probs.argmax(-1)] = True
    assert torch.equal(torch.isfinite(out), expected_keep)


# ---------------------------------------------------------------------------
# Ministral G_k k=0.25 (custom pless_renyi path). NAMING TRAP: gk_0_025 is
# k=0.025, gk_0_25 is k=0.25 -- guard both mappings and filename uniqueness.
# ---------------------------------------------------------------------------
def test_ministral_gk_0_25_config_row(ministral):
    rows = {c[0]: c for c in ministral.CONFIGS}
    assert rows["gk_0_25"] == ("gk_0_25", "pless_renyi", 1.0, 0.25)
    names = ministral.CONFIG_NAMES
    # placed with the Ministral-only G_k block, right after gk_0_2
    assert names.index("gk_0_25") == names.index("gk_0_2") + 1


def test_ministral_gk_0_25_vs_gk_0_025_naming_trap(ministral):
    rows = {c[0]: c for c in ministral.CONFIGS}
    assert rows["gk_0_025"][3] == 0.025
    assert rows["gk_0_25"][3] == 0.25
    assert rows["gk_0_025"][1] == rows["gk_0_25"][1] == "pless_renyi"


def test_ministral_config_names_unique_and_filenames_distinct(ministral, tmp_path):
    names = ministral.CONFIG_NAMES
    assert len(set(names)) == len(names)
    for plat in ("atcoder", "leetcode"):
        paths = [ministral._output_path(tmp_path, c[0], plat, c[2]).name
                 for c in ministral.CONFIGS]
        assert len(set(paths)) == len(paths)
    # every G_k row has a distinct k
    ks = [c[3] for c in ministral.CONFIGS if c[1] == "pless_renyi"]
    assert len(set(ks)) == len(ks)


def test_ministral_gk_0_25_not_native(ministral):
    assert "gk_0_25" not in ministral._NATIVE_TOP_P_TOP_K
    config_by_name = {c[0]: c for c in ministral.CONFIGS}
    assert ministral._all_configs_native(["gk_0_25"], config_by_name) is False
    assert ministral._record_top_k("pless_renyi") == 0


def test_ministral_gk_0_25_output_path(ministral, tmp_path):
    for plat in ("atcoder", "leetcode"):
        p = ministral._output_path(tmp_path, "gk_0_25", plat, 1.0)
        assert p.name == f"gk_0_25_{plat}_think_t1.0.jsonl"
        for other in ("gk_0_025", "gk_0_2", "gk_0_4", "gk_0_35", "gk_0_45"):
            assert p != ministral._output_path(tmp_path, other, plat, 1.0)


def test_pless_renyi_mask_accepts_k_0_25():
    """k=0.25 is not special-cased (only 2, 1, 0 are): generic branch."""
    torch = pytest.importorskip("torch")
    from bench.generator_vllm import _pless_renyi_mask_logits

    g = torch.Generator().manual_seed(1)
    logits = torch.randn(3, 50, generator=g) * 3
    probs = torch.softmax(logits.float(), dim=-1)
    k = 0.25
    thr = probs.pow(k).sum(dim=-1, keepdim=True).pow(1.0 / (k - 1.0))
    out = _pless_renyi_mask_logits(logits.clone(), k=k)
    expected_keep = probs >= thr
    expected_keep[torch.arange(3), probs.argmax(-1)] = True
    assert torch.equal(torch.isfinite(out), expected_keep)


# ---------------------------------------------------------------------------
# Ministral G_k k=0.5 (custom pless_renyi path; first in the GPU2 queue).
# NAMING TRAPS: gk_0_05 is k=0.05, gk_0_5 is k=0.5 (and gk_0_025 vs gk_0_25).
# ---------------------------------------------------------------------------
def test_ministral_gk_0_5_config_row(ministral):
    rows = {c[0]: c for c in ministral.CONFIGS}
    assert rows["gk_0_5"] == ("gk_0_5", "pless_renyi", 1.0, 0.5)
    names = ministral.CONFIG_NAMES
    assert names.index("gk_0_5") == names.index("gk_0_45") + 1


@pytest.mark.parametrize("name,k", [("gk_0_05", 0.05), ("gk_0_5", 0.5),
                                    ("gk_0_025", 0.025), ("gk_0_25", 0.25)])
def test_ministral_gk_naming_traps(ministral, name, k):
    row = {c[0]: c for c in ministral.CONFIGS}[name]
    assert row[1] == "pless_renyi" and row[3] == k


def test_ministral_gk_0_5_not_native(ministral):
    assert "gk_0_5" not in ministral._NATIVE_TOP_P_TOP_K
    config_by_name = {c[0]: c for c in ministral.CONFIGS}
    assert ministral._all_configs_native(["gk_0_5"], config_by_name) is False
    assert ministral._record_top_k("pless_renyi") == 0


def test_ministral_gk_0_5_output_path(ministral, tmp_path):
    for plat in ("atcoder", "leetcode"):
        p = ministral._output_path(tmp_path, "gk_0_5", plat, 1.0)
        assert p.name == f"gk_0_5_{plat}_think_t1.0.jsonl"
        assert p != ministral._output_path(tmp_path, "gk_0_05", plat, 1.0)
        assert ministral._output_path(tmp_path, "gk_0_05", plat, 1.0).name \
            == f"gk_0_05_{plat}_think_t1.0.jsonl"


def test_pless_renyi_mask_accepts_k_0_5():
    """k=0.5 (exponent 1/(k-1) = -2) takes the generic branch."""
    torch = pytest.importorskip("torch")
    from bench.generator_vllm import _pless_renyi_mask_logits

    g = torch.Generator().manual_seed(2)
    logits = torch.randn(3, 50, generator=g) * 3
    probs = torch.softmax(logits.float(), dim=-1)
    k = 0.5
    thr = probs.sqrt().sum(dim=-1, keepdim=True).pow(-2.0)
    out = _pless_renyi_mask_logits(logits.clone(), k=k)
    expected_keep = probs >= thr
    expected_keep[torch.arange(3), probs.argmax(-1)] = True
    assert torch.equal(torch.isfinite(out), expected_keep)


def test_pless_renyi_k_0_5_large_vocab_finite():
    """Large vocab, float32: threshold finite and in (1/v, 1]; no overflow."""
    torch = pytest.importorskip("torch")
    from bench.generator_vllm import _pless_renyi_mask_logits

    v = 131072
    g = torch.Generator().manual_seed(3)
    for scale in (0.0, 1.0, 10.0, 100.0):
        logits = torch.randn(2, v, generator=g) * scale
        probs = torch.softmax(logits.float(), dim=-1)
        thr = probs.pow(0.5).sum(dim=-1).pow(-2.0)
        assert torch.isfinite(thr).all()
        assert (thr >= 1.0 / v * (1 - 1e-4)).all() and (thr <= 1.0 + 1e-6).all()
        out = _pless_renyi_mask_logits(logits.clone(), k=0.5)
        assert torch.isfinite(out).any(dim=-1).all()
