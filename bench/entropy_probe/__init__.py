"""Cross-domain entropy probe — tests whether per-token entropy is bimodal
on non-code domains the same way it is on code generation.

Built as an isolated module: zero edits to existing code-generation
pipeline. Imports `teacher_forced_entropy` from
`bench.eval.phase_entropy_probe` read-only.
"""
