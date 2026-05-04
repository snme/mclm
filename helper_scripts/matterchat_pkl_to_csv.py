"""Convert MatterChat train.pkl → train.csv without loading pymatgen structures.

Stubs every pymatgen class so the pickle unpickles fast without OOM.
Outputs the 12 scalar columns that match val.csv.
"""
import argparse
import pickle
import sys
from pathlib import Path

import pandas as pd


class _Stub:
    """Placeholder for any pymatgen object — stores nothing."""
    def __init__(self, *a, **kw):
        pass
    def __setstate__(self, state):
        pass


class _SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if "pymatgen" in module or "monty" in module:
            return _Stub
        return super().find_class(module, name)


_SCALAR_COLS = [
    "material_id", "chemical_formula", "space_group", "crystal_system",
    "magnetic_order", "is_metal", "is_magnetic", "direct_bandgap",
    "stable", "bandgap", "formation_energy", "energy_above_hull",
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pkl", required=True)
    p.add_argument("--out_csv", required=True)
    args = p.parse_args()

    print(f"Loading {args.pkl} with structure-stubbing unpickler...", flush=True)
    with open(args.pkl, "rb") as f:
        data = _SafeUnpickler(f).load()

    print(f"  type={type(data).__name__}", flush=True)

    if isinstance(data, pd.DataFrame):
        df = data
    elif isinstance(data, dict):
        # dict of lists or dict of records
        df = pd.DataFrame(data)
    elif isinstance(data, list):
        df = pd.DataFrame(data)
    else:
        print(f"Unexpected type: {type(data)}", file=sys.stderr)
        sys.exit(1)

    print(f"  shape={df.shape}, cols={list(df.columns[:15])}", flush=True)

    # Keep only scalar columns that exist
    keep = [c for c in _SCALAR_COLS if c in df.columns]
    missing = [c for c in _SCALAR_COLS if c not in df.columns]
    if missing:
        print(f"  [warn] missing columns: {missing}")
    df_out = df[keep].copy()

    # Drop rows with _Stub in any column (structure leaked into a scalar slot)
    mask = df_out.apply(lambda col: col.map(lambda x: isinstance(x, _Stub))).any(axis=1)
    if mask.any():
        print(f"  [warn] dropping {mask.sum()} rows with stub objects in scalar columns")
        df_out = df_out[~mask]

    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out, index=False)
    print(f"Wrote {len(df_out)} rows → {out}", flush=True)


if __name__ == "__main__":
    main()
