"""Per-task reseeding for the MBPP runner's opt-in ``--seed``.

A task's samples depend only on (seed, task_id), not on task order or on resume, because the runner calls
``seed_task(seed, task_id)`` right before generating each task. The per-task seed comes from
``numpy.random.SeedSequence([seed, task_id])``, not ``seed + task_id``: with the sum, base seeds S and S+1 would reuse
each other's streams shifted by one task.
"""
import random

import numpy as np
import torch


def task_seed(seed: int, task_id: int) -> int:
    """Deterministic 32-bit seed for one task (seed, task_id >= 0)."""
    return int(np.random.SeedSequence([int(seed), int(task_id)]).generate_state(1, dtype=np.uint32)[0])


def seed_task(seed: int, task_id: int) -> int:
    """Reseed python, numpy and torch (CPU + all CUDA devices) for one task; returns the derived seed."""
    s = task_seed(seed, task_id)
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)   # lazy no-op without CUDA; torch.manual_seed also seeds CUDA/MPS
    return s
