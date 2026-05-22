"""Teacher-forced per-token entropy.

Imports the reusable function from ``bench.eval.phase_entropy_probe``
without modifying it. That function is domain-agnostic — it just runs
one forward pass on a (prompt+completion) token sequence and returns
the per-token entropy of the model's predictive distribution at each
position after the prompt.
"""
from bench.eval.phase_entropy_probe import teacher_forced_entropy

__all__ = ["teacher_forced_entropy"]
