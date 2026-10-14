"""Phase 2 of the multi-arm-vs-single-arm equivalence test — pure config
parity, zero GPU, zero tokenizer, zero model load.

Why this check and not a prompt-text diff: ``bench/apps/multi_arm_runner.py``
and ``bench/apps/runner.py`` both call the SAME shared functions
(``format_prompt_apps_instruct``, ``encode_prompt_for_vllm``, ``_method_key``,
``_output_path``, ``generate_samples_vllm``/``generate_samples_standard_vllm``)
-- a byte-diff of their prompt text or output paths would trivially pass by
construction and prove nothing. The actual risk surface is narrower and
specific to multi-arm: ``_preset_arm_spec()`` HARDCODES its own copy of what
it believes each preset arm's generation parameters are (its docstring says
"mirrors run_cf748_apps_qwen3.sh:arm_args() byte-for-byte"), independent of
``bench.apps.runner``'s own argparse defaults. If either side ever changes
without the other being updated, multi-arm's "pless" arm could silently stop
matching what running ``bench.apps.runner --method pless`` actually produces,
with no error -- both would complete successfully with EXTRA_ARG_KEY).  This
script catches that class of drift automatically, not by one-time manual
inspection.

For each preset arm, builds the generation-parameter dict two independent
ways and diffs them:
  (a) ``_preset_arm_spec(name)`` (the hardcoded mapping multi-arm uses)
  (b) parsing ``bench.apps.runner``'s REAL argparser with the CLI flags the
      preset's own docstring claims are equivalent, then reading the
      resulting ``Namespace``'s matching fields.

Also exercises ``_parse_arm_raw`` against a hand-built CLI-equivalent
Namespace for one custom spec, and does the same equality check for the
``_method_key`` filename-stem computation (shared function, but worth
confirming multi-arm's ``_arm_to_namespace`` populates every field
``_method_key`` reads, since a missing field would silently read a stale
Namespace default instead of raising).

Runs the full comparison twice -- once with ``enable_thinking=False``
(default) and once with ``True`` -- since that flag changes both
``_method_key``'s filename suffix and is forwarded from the real ``args``
object at call time in ``multi_arm_runner.main()`` (confirmed: NOT hardcoded
like the loop-force params below), so it's a genuinely live branch worth
covering on both settings, not just the default.

KNOWN LIMITATION (not fixable without changing multi_arm_runner.py itself,
so documented here rather than silently passed over): ``main()`` hardcodes
``loop_ngram_n=None, loop_ngram_k=None, loop_window=None`` as *literals*
directly in its ``generate_samples_vllm`` call -- they never flow through
``_arm_to_namespace``, so there is no Namespace field for this script to
diff against runner.py's ``--force-think-on-loop``-gated equivalents. Today
both default to the same effective None/None, confirmed by manual
inspection (2026-10-14 advisor review), not by this script. If multi-arm
ever grows loop-force support, it will ship unverified by this check.

Usage::

    uv run python scripts/compare_apps_arm_config_parity.py
"""
from __future__ import annotations

import sys

from bench.apps.multi_arm_runner import (
    _arm_to_namespace,
    _parse_arm_raw,
    _preset_arm_spec,
)
from bench.apps.runner import _build_argparser, _method_key

# (preset arm name, equivalent runner.py CLI flags per _preset_arm_spec's own
# docstring) -- the claim under test is that these two columns describe the
# same generation.
PRESET_EQUIVALENT_CLI: list[tuple[str, list[str]]] = [
    ("pless", ["--method", "pless", "--temperature", "1.0"]),
    ("pless_norm", ["--method", "pless_norm", "--temperature", "1.0"]),
    ("renyi_k0.1", ["--method", "pless_renyi", "--renyi-k", "0.1", "--temperature", "1.0"]),
    ("rec", ["--method", "temp", "--temperature", "0.6", "--top-p", "0.95", "--top-k", "20"]),
    ("topp0.9", ["--method", "temp", "--top-p", "0.9", "--top-k", "0", "--temperature", "1.0"]),
    ("temp", ["--method", "temp", "--temperature", "1.0"]),
]

# Fields that actually flow into generation (SamplingParams / custom-sampler
# cfg) or the output filename -- NOT every Namespace field, since the
# single-arm runner.py Namespace has many fields multi-arm's Namespace
# deliberately doesn't populate (scaffolds, paper-replica, etc.) -- those are
# out of scope for multi-arm by the module's own docstring ("loop-force,
# scaffolds, split-sampler... are left at their single-arm defaults").
COMPARE_FIELDS = ["method", "temperature", "top_p", "top_k", "renyi_k", "alpha"]

BASE_ARGS_CLI = [
    "--model", "placeholder/model", "--source", "ATCODER",
    "--difficulty", "introductory", "--results-dir", "/tmp/unused",
    "--method", "temp",  # required by argparse; value unused -- base_args is
                         # only ever read for its .enable_thinking field.
]


def _runner_namespace_for(extra_flags: list[str]):
    parser = _build_argparser()
    return parser.parse_args(BASE_ARGS_CLI + extra_flags)


def _run_all_checks(base_args, enable_thinking_label: str) -> bool:
    """Run every comparison for one enable_thinking setting. Returns True if
    any mismatch was found (so callers can OR this across both settings)."""
    any_fail = False

    print(f"=== Preset arm config parity [enable_thinking={enable_thinking_label}] ===")
    for arm_name, cli_flags in PRESET_EQUIVALENT_CLI:
        spec = _preset_arm_spec(arm_name)
        multi_arm_ns = _arm_to_namespace(spec, base_args)
        extra = cli_flags + (["--enable-thinking"] if base_args.enable_thinking else [])
        runner_ns = _runner_namespace_for(extra)

        mismatches = []
        for field in COMPARE_FIELDS:
            a, b = getattr(multi_arm_ns, field), getattr(runner_ns, field)
            if a != b:
                mismatches.append((field, a, b))

        if mismatches:
            any_fail = True
            print(f"  [FAIL] arm={arm_name!r}:")
            for field, a, b in mismatches:
                print(f"    {field}: multi-arm={a!r}  vs  runner.py-equivalent={b!r}")
        else:
            print(f"  [PASS] arm={arm_name!r} ({len(COMPARE_FIELDS)} fields match)")

    print(f"\n=== _method_key parity [enable_thinking={enable_thinking_label}] ===")
    for arm_name, cli_flags in PRESET_EQUIVALENT_CLI:
        spec = _preset_arm_spec(arm_name)
        multi_arm_ns = _arm_to_namespace(spec, base_args)
        extra = cli_flags + (["--enable-thinking"] if base_args.enable_thinking else [])
        runner_ns = _runner_namespace_for(extra)
        key_multi = _method_key(multi_arm_ns)
        key_runner = _method_key(runner_ns)
        if key_multi != key_runner:
            any_fail = True
            print(f"  [FAIL] arm={arm_name!r}: multi-arm key={key_multi!r} "
                  f"vs runner.py key={key_runner!r}")
        else:
            print(f"  [PASS] arm={arm_name!r}: key={key_multi!r}")

    print(f"\n=== --arm-raw custom spec parity [enable_thinking={enable_thinking_label}] ===")
    raw_spec_str = "name=custom1,method=temp,temperature=1.5,top_p=0.85"
    raw_spec = _parse_arm_raw(raw_spec_str)
    multi_arm_ns = _arm_to_namespace(raw_spec, base_args)
    extra = ["--method", "temp", "--temperature", "1.5", "--top-p", "0.85"]
    extra += ["--enable-thinking"] if base_args.enable_thinking else []
    runner_ns = _runner_namespace_for(extra)
    mismatches = [
        (f, getattr(multi_arm_ns, f), getattr(runner_ns, f))
        for f in COMPARE_FIELDS
        if getattr(multi_arm_ns, f) != getattr(runner_ns, f)
    ]
    if mismatches:
        any_fail = True
        print(f"  [FAIL] arm-raw={raw_spec_str!r}:")
        for field, a, b in mismatches:
            print(f"    {field}: multi-arm={a!r} vs runner.py-equivalent={b!r}")
    else:
        print(f"  [PASS] arm-raw={raw_spec_str!r} ({len(COMPARE_FIELDS)} fields match)")

    print()
    return any_fail


def main() -> None:
    parser = _build_argparser()
    any_fail = False
    for label, cli in [("False", BASE_ARGS_CLI), ("True", BASE_ARGS_CLI + ["--enable-thinking"])]:
        base_args = parser.parse_args(cli)
        any_fail = _run_all_checks(base_args, label) or any_fail

    print("NOTE: loop-force params (loop_ngram_n/loop_ngram_k/loop_window) are "
          "hardcoded as literals directly in multi_arm_runner.main()'s generation "
          "call, never routed through _arm_to_namespace -- structurally untestable "
          "by this script. Verified by manual inspection (2026-10-14) that both "
          "sides currently default to the equivalent None/None; NOT covered by "
          "the checks above. If multi-arm ever wires loop-force support in, it "
          "will ship unverified by this script.")

    if any_fail:
        print("\nRESULT: at least one mismatch found -- multi_arm_runner.py's preset "
              "mapping has drifted from bench.apps.runner's actual CLI behavior. "
              "Do not trust multi-arm's generation for the affected arm(s) until fixed.")
        sys.exit(1)
    print("\nRESULT: all preset arms produce generation parameters identical to "
          "their documented-equivalent runner.py CLI invocation, in both "
          "enable_thinking settings.")


if __name__ == "__main__":
    main()
