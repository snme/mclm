"""Shared element / composition helpers used by aux heads, training datasets,
and analysis scripts.

Centralizes the symbol→Z map, the multi-hot composition vector, and the
exact-count composition vector that previously lived as duplicate copies in
helper_scripts/probe_atoms_mapper_clusters.py and inline in train_stage3a.py.

Existing duplicates (helper_scripts/probe_*.py, train_stage3a.py:151) are left
in place to avoid breaking those scripts; new code should import from here.
"""
from __future__ import annotations

import numpy as np

# Atomic-number range we model: hydrogen (Z=1) through fermium (Z=100). Same
# bound used everywhere in the pipeline (matches CompositionHead's 100-d output).
N_ELEMENTS = 100

# Max per-element atom count the count-probe / count-aware aux head predicts.
# pairs.parquet is filtered to ≤20 atoms total per cell, so 20 is a safe upper
# bound for any single element. Counts above this are clamped (and logged once
# at dataset init).
MAX_COUNT = 20


def symbol_to_z() -> dict[str, int]:
    """ASE element symbol → atomic number, restricted to Z in [1, N_ELEMENTS]."""
    from ase.data import atomic_numbers
    return {s: z for s, z in atomic_numbers.items() if 1 <= z <= N_ELEMENTS}


def composition_multihot(elements: list[str], sym2z: dict | None = None) -> np.ndarray:
    """Multi-hot vector over Z=1..N_ELEMENTS (ASE 1-indexed; we use 0-indexed
    slot Z-1). Shape (N_ELEMENTS,) float32.

    `elements` is the per-atom element list (with repeats), e.g. ['Cu','Ni','Cu'].
    Repeats are collapsed to "1.0 = present" — for counts use composition_count_vec.
    """
    if sym2z is None:
        sym2z = symbol_to_z()
    v = np.zeros(N_ELEMENTS, dtype=np.float32)
    for s in elements:
        z = sym2z.get(s.strip())
        if z is not None and 1 <= z <= N_ELEMENTS:
            v[z - 1] = 1.0
    return v


def composition_count_vec(elements: list[str], sym2z: dict | None = None,
                          dtype: np.dtype = np.int64) -> np.ndarray:
    """Integer count per Z=1..N_ELEMENTS, clamped to [0, MAX_COUNT]. Shape
    (N_ELEMENTS,) of the requested dtype (default int64).

    Pass `dtype=np.float32` when the result is going into a tensor that will
    be torch.stack()'d alongside other float32 aux targets (avoids dtype mismatch).
    """
    if sym2z is None:
        sym2z = symbol_to_z()
    v = np.zeros(N_ELEMENTS, dtype=np.int64)
    for s in elements:
        z = sym2z.get(s.strip())
        if z is not None and 1 <= z <= N_ELEMENTS:
            v[z - 1] += 1
    np.clip(v, 0, MAX_COUNT, out=v)
    return v.astype(dtype, copy=False)
