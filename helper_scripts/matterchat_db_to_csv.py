"""ASE db → CSV with id + label columns for AtomisticLanguageDataset.

Reads each matterchat ASE db built by matterchat_pkl_to_ase.py, dumps:
  material_id, chemical_formula, space_group, crystal_system, magnetic_order,
  is_metal, is_magnetic, direct_bandgap, stable,
  bandgap, formation_energy, energy_above_hull

Output CSV lives next to the db. Pass to eval_matterchat.py via --data_csv.
"""
import argparse
from pathlib import Path

import polars as pl
from ase.db import connect
from tqdm import tqdm


_COLS = [
    "material_id", "chemical_formula", "space_group", "crystal_system",
    "magnetic_order", "is_metal", "is_magnetic", "direct_bandgap", "stable",
    "bandgap", "formation_energy", "energy_above_hull",
]


def dump(db_path: Path, csv_path: Path):
    db = connect(db_path)
    rows = []
    for row in tqdm(db.select(), total=len(db), desc=db_path.name, unit="row"):
        d = {**row.key_value_pairs, **row.data}
        rows.append({c: d.get(c) for c in _COLS})
    pl.DataFrame(rows, schema={c: pl.Utf8 for c in _COLS}).write_csv(csv_path)
    print(f"wrote {len(rows):,} rows → {csv_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--db", required=True, help="MatterChat ASE db (built by matterchat_pkl_to_ase.py)")
    p.add_argument("--out", default=None, help="default: <db>.csv")
    args = p.parse_args()
    db_path = Path(args.db)
    out_path = Path(args.out) if args.out else db_path.with_suffix(".csv")
    dump(db_path, out_path)
