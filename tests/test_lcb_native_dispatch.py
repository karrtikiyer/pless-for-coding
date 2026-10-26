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
