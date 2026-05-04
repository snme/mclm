"""Shared structure-metric helpers for Stage 3b crystal-generation evaluation.

Wraps MatterGen's evaluation submodule (`external/mattergen/mattergen/evaluation/`)
to avoid reimplementing StructureMatcher / MatterSim relaxation / smact-based
validity. The thin layer here pins CDVAE/CrystaLLM tolerances and gives the eval
scripts a single import surface.

Public API
----------
- validity_geom(s) -> bool
- validity_charge(s) -> bool
- validity_full(s) -> dict[str, bool]
- match_one(gen, ref, matcher=None) -> (matched: bool, rmse: float | None)
- match_many(gens, ref) -> dict
- composition_set(s) -> set[str]
- composition_match_ratio(s, target_elements) -> float
- density_g_per_cm3(s) -> float
- novel_mask(structures, reference_structures, matcher=None) -> np.ndarray[bool]
- unique_indices(structures, matcher=None) -> list[int]
- relax_structures_mattersim(atoms_or_structs, device, potential_path, **kwargs)
    -> (relaxed_atoms, total_energies)
- e_above_hull_per_atom(structure, total_energy, hull_data) -> float

Tolerances
----------
We pin to the **CDVAE / CrystaLLM defaults** for direct comparability:
    ltol=0.3, stol=0.5, angle_tol=10
These are LOOSER than MatterGen's defaults (ltol=0.2, stol=0.3, angle_tol=5).
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from ase import Atoms
from pymatgen.core.composition import Composition
from pymatgen.core.structure import Structure
from pymatgen.io.ase import AseAtomsAdaptor

# MatterGen utilities — single source of truth for matcher + relaxation
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "external" / "mattergen"))
from mattergen.evaluation.utils.structure_matcher import OrderedStructureMatcher  # noqa: E402
from mattergen.evaluation.utils.dataset_matcher import (  # noqa: E402
    get_matches,
    get_unique,
)


# ─────────────────────────────────────────────────────────────────────────────
# Matcher — pinned to CDVAE/CrystaLLM tolerances
# ─────────────────────────────────────────────────────────────────────────────

CDVAE_TOLS = dict(ltol=0.3, stol=0.5, angle_tol=10)


def cdvae_matcher() -> OrderedStructureMatcher:
    """Fresh CDVAE-tolerance OrderedStructureMatcher. Cheap to construct; callers
    that loop should hold a single instance to avoid Python-level overhead."""
    return OrderedStructureMatcher(**CDVAE_TOLS)


_DEFAULT = None  # lazy singleton


def _default_matcher() -> OrderedStructureMatcher:
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = cdvae_matcher()
    return _DEFAULT


# ─────────────────────────────────────────────────────────────────────────────
# Validity
# ─────────────────────────────────────────────────────────────────────────────

def validity_geom(s: Structure, min_dist: float = 0.5) -> bool:
    """True iff structure is geometrically valid: no overlapping atoms (>= min_dist
    apart), positive cell volume, sensible lattice angles. Mirrors CDVAE/Crystal-text-LLM.
    """
    try:
        if s.volume <= 0:
            return False
        # Pymatgen's is_valid checks min interatomic distance against `tol` (default 0.5).
        if not s.is_valid(tol=min_dist):
            return False
        a, b, c = s.lattice.angles
        if not all(0 < ang < 180 for ang in (a, b, c)):
            return False
        return True
    except Exception:
        return False


def validity_charge(s: Structure) -> bool:
    """True iff a charge-neutral oxidation-state assignment exists (smact). The
    simplest/fastest charge-balance check; matches the CDVAE recipe.
    """
    try:
        import smact  # noqa: F401  (lazy import; smact is optional but installed)
        from smact.screening import pauling_test
    except Exception:
        return False
    try:
        comp = Composition(s.composition.reduced_formula)
        symbols = [str(el) for el in comp.elements]
        counts = [int(comp[el]) for el in comp.elements]
        # smact iterates oxidation-state combinations.
        from smact import element_dictionary, neutral_ratios
        elem_objs = [element_dictionary().get(sym) for sym in symbols]
        if any(e is None for e in elem_objs):
            return False
        ox_combos = [e.oxidation_states for e in elem_objs]
        from itertools import product
        for ox_states in product(*ox_combos):
            if neutral_ratios(ox_states, stoichs=[(c,) for c in counts])[0]:
                # Charge-balance found; also enforce Pauling electronegativity ordering.
                electronegs = [e.pauling_eneg for e in elem_objs if e.pauling_eneg is not None]
                if len(electronegs) != len(elem_objs):
                    return True  # missing eneg: accept charge-balance alone
                if pauling_test(ox_states, electronegs, symbols):
                    return True
        return False
    except Exception:
        return False


def validity_full(s: Structure) -> dict[str, bool]:
    g = validity_geom(s)
    c = validity_charge(s)
    return {"geom": g, "charge": c, "both": g and c}


# ─────────────────────────────────────────────────────────────────────────────
# Match rate / RMSE — CDVAE-style
# ─────────────────────────────────────────────────────────────────────────────

def match_one(
    gen: Structure, ref: Structure,
    matcher: OrderedStructureMatcher | None = None,
) -> tuple[bool, float | None]:
    """Returns (matched, rmse). RMSE is None if not matched. Uses the CDVAE matcher."""
    m = matcher or _default_matcher()
    try:
        if not m.fit(gen, ref):
            return False, None
        rms = m.get_rms_dist(gen, ref)
        if rms is None:
            return False, None
        # pymatgen returns (rms, max_dist); CDVAE/CrystaLLM report the rms component.
        return True, float(rms[0])
    except Exception:
        return False, None


def match_many(
    gens: Sequence[Structure], ref: Structure,
    matcher: OrderedStructureMatcher | None = None,
) -> dict:
    """For a list of generations against one reference, compute n=1 / n=K aggregates.

    Returns:
      {
        "n":            len(gens),
        "matched_n1":   bool — first generation matches ref
        "rmse_n1":      float | None
        "matched_nK":   bool — any generation matches ref
        "rmse_nK":      float | None — minimum rmse among matches
        "match_idx":    list[int] of the indices that matched
      }
    """
    m = matcher or _default_matcher()
    out = {
        "n": len(gens),
        "matched_n1": False, "rmse_n1": None,
        "matched_nK": False, "rmse_nK": None,
        "match_idx": [],
    }
    rmses = []
    for i, g in enumerate(gens):
        matched, rmse = match_one(g, ref, m)
        if matched:
            out["match_idx"].append(i)
            rmses.append(rmse)
            if i == 0:
                out["matched_n1"] = True
                out["rmse_n1"] = rmse
    if rmses:
        out["matched_nK"] = True
        out["rmse_nK"] = float(min(rmses))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Composition / property
# ─────────────────────────────────────────────────────────────────────────────

def composition_set(s: Structure) -> set[str]:
    """Set of element symbols present in s."""
    return {str(el) for el in s.composition.elements}


def composition_match_ratio(s: Structure, target_elements: Iterable[str]) -> float:
    """Fraction of target elements actually present in the generated structure."""
    target = {str(t) for t in target_elements}
    if not target:
        return 0.0
    have = composition_set(s)
    return len(target & have) / len(target)


def density_g_per_cm3(s: Structure) -> float:
    """Mass density in g/cm³. Uses pymatgen's built-in density (already in g/cm³)."""
    try:
        return float(s.density)
    except Exception:
        return float("nan")


# ─────────────────────────────────────────────────────────────────────────────
# Uniqueness / novelty
# ─────────────────────────────────────────────────────────────────────────────

def unique_indices(
    structures: Sequence[Structure],
    matcher: OrderedStructureMatcher | None = None,
) -> list[int]:
    """Indices of unique structures in a batch (StructureMatcher-equivalent).

    Reuses MatterGen's `get_unique`. O(N²) — acceptable for N ≤ 1024.
    """
    m = matcher or _default_matcher()
    return get_unique(m, list(structures))


def novel_mask(
    structures: Sequence[Structure],
    reference_structures: Sequence[Structure],
    matcher: OrderedStructureMatcher | None = None,
) -> np.ndarray:
    """Boolean mask: True for structures NOT matched in the reference dataset.

    O(N × M). For large reference sets, pre-filter by reduced formula upstream.
    """
    m = matcher or _default_matcher()
    matches = get_matches(m, list(structures), list(reference_structures))
    novel = np.ones(len(structures), dtype=bool)
    for idx, ref_hits in matches.items():
        if ref_hits:
            novel[idx] = False
    return novel


def novel_mask_by_formula(
    structures: Sequence[Structure],
    reference_structures: Sequence[Structure],
    matcher: OrderedStructureMatcher | None = None,
) -> np.ndarray:
    """Faster novelty: only compare against references with matching reduced formula.
    O(N × M_per_formula) instead of O(N × M_total). Defaults to this in practice.
    """
    m = matcher or _default_matcher()
    # Bucket references by reduced formula.
    by_formula: dict[str, list[Structure]] = {}
    for r in reference_structures:
        rf = r.composition.reduced_formula
        by_formula.setdefault(rf, []).append(r)
    novel = np.ones(len(structures), dtype=bool)
    for i, s in enumerate(structures):
        rf = s.composition.reduced_formula
        candidates = by_formula.get(rf, [])
        for r in candidates:
            try:
                if m.fit(s, r):
                    novel[i] = False
                    break
            except Exception:
                continue
    return novel


# ─────────────────────────────────────────────────────────────────────────────
# Relaxation + energetics (MatterSim)
# ─────────────────────────────────────────────────────────────────────────────

def relax_structures_mattersim(
    inputs: Sequence[Structure | Atoms],
    device: str = "cuda",
    potential_path: str | None = None,
    fmax: float = 0.05,
    max_n_steps: int = 500,
    output_extxyz: str | Path | None = None,
) -> tuple[list[Atoms], np.ndarray]:
    """Relax a batch with MatterSim's BatchRelaxer (EXPCELLFILTER, fmax=0.05).

    Mirrors `mattergen.evaluation.utils.relaxation.relax_atoms` but accepts either
    Structures or Atoms. Returns (relaxed_atoms, total_energies). Total energies
    are MatterSim's internal scale (eV/cell — divide by len(atoms) for per-atom).
    """
    from mattergen.evaluation.utils.relaxation import relax_atoms
    atoms_list: list[Atoms] = []
    for x in inputs:
        if isinstance(x, Structure):
            atoms_list.append(AseAtomsAdaptor.get_atoms(x))
        elif isinstance(x, Atoms):
            atoms_list.append(x)
        else:
            raise TypeError(f"unsupported input type: {type(x)}")
    relaxed, energies = relax_atoms(
        atoms_list, device=device, potential_load_path=potential_path,
        fmax=fmax, max_n_steps=max_n_steps,
        output_path=str(output_extxyz) if output_extxyz else None,
    )
    return relaxed, energies


def total_energy_per_atom(atoms: Atoms) -> float:
    """Per-atom total energy from a relaxed Atoms object's info dict (MatterSim sets
    `total_energy` on the relaxed atoms). NaN if missing.
    """
    e = atoms.info.get("total_energy")
    if e is None:
        return float("nan")
    return float(e) / max(1, len(atoms))


# ─────────────────────────────────────────────────────────────────────────────
# Hull reference loader
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_HULL_DIR = Path("/home/sathyae/orcd/pool/eval_data/mp_hull")


def load_hull_reference(hull_dir: Path | str = DEFAULT_HULL_DIR):
    """Return the convex-hull reference for E_hull scoring.

    Resolution order:
      1. `<hull_dir>/preferred.txt` names a specific file in `<hull_dir>` (set by
         `helper_scripts/fetch_mp_hull.py`).
      2. Otherwise, look for `reference_MP2020correction.gz` (MatterGen format)
         then `reference_TRI2024correction.gz`, then `reference_mp_api.pkl`.
      3. `.gz` files are decoded via MatterGen's `LMDBGZSerializer.deserialize` →
         a `ReferenceDataset`. The eval scripts then materialize a list of
         `ComputedStructureEntry` for the chemsystems they care about.
      4. `.pkl` files are loaded as a list of `ComputedStructureEntry`.

    Returns:
      Either a `ReferenceDataset` (LMDB-backed, lazy) or a list of
      `ComputedStructureEntry`. Use duck typing in callers.
    """
    hull_dir = Path(hull_dir)
    if not hull_dir.exists():
        raise FileNotFoundError(
            f"hull dir {hull_dir} does not exist — run helper_scripts/fetch_mp_hull.py"
        )

    preferred = hull_dir / "preferred.txt"
    if preferred.exists():
        fname = preferred.read_text().strip()
        path = hull_dir / fname
    else:
        for candidate in ("reference_MP2020correction.gz",
                          "reference_TRI2024correction.gz",
                          "reference_mp_api.pkl"):
            if (hull_dir / candidate).exists():
                path = hull_dir / candidate
                break
        else:
            raise FileNotFoundError(
                f"no hull reference found in {hull_dir} — run helper_scripts/fetch_mp_hull.py"
            )

    if path.suffix == ".gz":
        # Detect Git-LFS-pointer-instead-of-real-file (common after a fresh
        # submodule clone without `git lfs pull`).
        with open(path, "rb") as f:
            head = f.read(64)
        if head.startswith(b"version https://git-lfs.github.com/spec"):
            raise FileNotFoundError(
                f"{path} is a Git LFS pointer, not the actual gzipped reference. "
                f"Run `cd external/mattergen && git lfs install --local && git lfs pull` "
                f"to materialize it."
            )
        from mattergen.evaluation.reference.reference_dataset_serializer import LMDBGZSerializer
        return LMDBGZSerializer().deserialize(str(path))
    if path.suffix == ".pkl":
        import pickle
        with open(path, "rb") as f:
            return pickle.load(f)
    raise ValueError(f"unknown hull reference format: {path}")


def hull_entries_for_chemsys(reference, chemsys_elements: Iterable[str]) -> list:
    """Materialize a list of ComputedStructureEntry covering the requested chemsys.

    For a `ReferenceDataset`, walks its `.entries_by_chemsys` lookup over all
    sub-systems of the requested elements (i.e., to build a hull that's valid for
    a structure with composition strictly inside the requested chemsys).

    Returns a flat list, deduped on `entry_id`.
    """
    elems = sorted({str(e) for e in chemsys_elements})
    if hasattr(reference, "entries_by_chemsys"):
        # ReferenceDataset path. The chemsys key is a "-"-joined SORTED element list.
        out = []
        seen_ids = set()
        # All sub-chemsystems of `elems` (power-set minus empty).
        from itertools import combinations
        for k in range(1, len(elems) + 1):
            for combo in combinations(elems, k):
                key = "-".join(combo)
                bucket = reference.entries_by_chemsys.get(key, [])
                for e in bucket:
                    eid = getattr(e, "entry_id", None) or id(e)
                    if eid in seen_ids:
                        continue
                    seen_ids.add(eid)
                    out.append(e)
        return out
    # Plain list of ComputedStructureEntry
    return [e for e in reference
            if set(str(el) for el in e.composition.elements).issubset(set(elems))]


# ─────────────────────────────────────────────────────────────────────────────
# Energy above hull
# ─────────────────────────────────────────────────────────────────────────────

def e_above_hull_per_atom(
    structure: Structure,
    total_energy_eV: float,
    hull_reference,
) -> float:
    """Compute E_hull (eV/atom) for one structure given its total energy.

    `hull_reference` can be:
      - a `ReferenceDataset` (the MatterGen LMDB-backed type),
      - a list of `ComputedStructureEntry`,
      - or a pre-built `PhaseDiagram`.

    Returns NaN if the chemsys is missing from the hull or if a PhaseDiagram
    cannot be constructed.
    """
    from pymatgen.analysis.phase_diagram import PhaseDiagram, PDEntry
    pd_entry = PDEntry(structure.composition, total_energy_eV)
    if hasattr(hull_reference, "get_e_above_hull"):
        try:
            return float(hull_reference.get_e_above_hull(pd_entry))
        except Exception:
            return float("nan")
    elems = [str(el) for el in structure.composition.elements]
    relevant = hull_entries_for_chemsys(hull_reference, elems)
    if not relevant:
        return float("nan")
    try:
        pd = PhaseDiagram(list(relevant) + [pd_entry])
        return float(pd.get_e_above_hull(pd_entry))
    except Exception:
        return float("nan")


# ─────────────────────────────────────────────────────────────────────────────
# Self-test (cheap; callable from a smoke script)
# ─────────────────────────────────────────────────────────────────────────────

def _smoke():
    from pymatgen.core import Lattice
    s = Structure(
        Lattice.cubic(4.2),
        ["Na", "Cl"],
        [[0, 0, 0], [0.5, 0.5, 0.5]],
    )
    print("validity_geom:", validity_geom(s))
    print("validity_charge:", validity_charge(s))
    print("composition_set:", composition_set(s))
    print("density:", density_g_per_cm3(s))
    print("match_one self vs self:", match_one(s, s))


if __name__ == "__main__":
    _smoke()
