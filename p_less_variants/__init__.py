"""New hyperparameter-free p-less variants (Sept 2026).

Each variant lives in its own file. None modifies the read-only
``p-less/p_less_samplers.py`` submodule. Variants use only the raw next-token
distribution (no temperature/top-k/top-p knobs, no α or G_k reparameterization
of the collision threshold — those are already surveyed elsewhere).

Available:
  - ``pless_effk``    — rank cut at ⌈1/Σpᵢ²⌉ (effective-support top-k).
  - ``pless_min2``    — plain p-less with a min-2 survivor floor.
  - ``pless_topmass`` — adaptive nucleus: cumulative mass ≥ 1 − Σpᵢ².
  - ``pless_top1half``— admit pᵢ ≥ max(pᵢ) / 2 (relative-to-top).
  - ``pless_gini``    — inverted threshold: pᵢ ≥ 1 − Σpᵢ².
"""
from p_less_variants.pless_effk import p_less_effk_decode
from p_less_variants.pless_min2 import p_less_min2_decode
from p_less_variants.pless_topmass import p_less_topmass_decode
from p_less_variants.pless_top1half import p_less_top1half_decode
from p_less_variants.pless_gini import p_less_gini_decode

__all__ = [
    "p_less_effk_decode",
    "p_less_min2_decode",
    "p_less_topmass_decode",
    "p_less_top1half_decode",
    "p_less_gini_decode",
]
