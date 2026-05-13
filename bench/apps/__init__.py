"""APPS benchmark integration.

Mirrors ``bench/humaneval/`` and is built around the same generator
(:mod:`bench.generator`) and samplers (:mod:`bench.sampler_bridge`) used for
MBPP. Only the dataset loader and prompt formatter are APPS-specific.

We do **not** execute APPS test cases in v1 — the goal of this integration is
to feed Qwen3-8B generations into the algosim algorithmic-diversity pipeline,
which clusters raw samples without requiring correctness gating. Correctness
evaluation on APPS (stdin/stdout I/O with timeouts) is out of scope.
"""
