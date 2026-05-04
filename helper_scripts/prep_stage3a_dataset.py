"""Convert pairs.parquet → MatterGen structure-only dataset cache (no alm_embedding).

For joint training (train_stage3a.py) this script is NOT needed — the training loop
reads pairs.parquet directly and builds ChemGraphs on the fly.

This script is useful for:
  - Visual inspection of the dataset in MatterGen's format
  - Running unconditional MatterGen generation on the same distribution
  - Future eval workflows that need pre-built structure caches

Output per-split directory:
  pos.npy             (sum_atoms_in_split, 3)   float32 fractional coords
  cell.npy            (N, 3, 3)                 float32 lattice
  atomic_numbers.npy  (sum_atoms_in_split,)     int64
  num_atoms.npy       (N,)                      int64
  structure_id.npy    (N,)                      str (row_id from pairs)

Usage:
  python helper_scripts/prep_stage3a_dataset.py \\
      --pairs_parquet /home/sathyae/orcd/pool/stage3a/pairs.parquet \\
      --out_root      /home/sathyae/orcd/pool/stage3a/mattergen_structures
"""
import argparse
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from ase import Atoms
from tqdm import tqdm


def atoms_struct_to_arrays(struct: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """atoms_struct dict → (frac_pos float32, cell float32, atomic_numbers int64)."""
    cell = np.asarray(struct["lattice_mat"], dtype=np.float32)
    coords = np.asarray(struct["coords"], dtype=np.float64)
    symbols = [s.strip() for s in struct["elements"]]
    if struct.get("cartesian", False):
        atoms = Atoms(symbols=symbols, positions=coords, cell=cell, pbc=True)
        frac = atoms.get_scaled_positions(wrap=True).astype(np.float32)
    else:
        frac = (coords % 1.0).astype(np.float32)
    Z = np.asarray(
        [Atoms(symbols=[s]).get_atomic_numbers()[0] for s in symbols], dtype=np.int64
    )
    return frac, cell, Z


def main(args):
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    pf = pq.ParquetFile(args.pairs_parquet)
    total = pf.metadata.num_rows
    print(f"reading {total:,} pairs from {args.pairs_parquet}", flush=True)

    rng = np.random.default_rng(args.seed)
    is_val = rng.random(total) < args.val_fraction
    n_val = int(is_val.sum())
    n_train = total - n_val
    print(f"split: {n_train:,} train / {n_val:,} val ({args.val_fraction:.1%})")

    # First pass: count atoms and assign rows to splits
    train_rows, val_rows = [], []
    n_atoms_all = np.zeros(total, dtype=np.int64)
    row_id_all = []
    cursor = 0
    for batch in tqdm(pf.iter_batches(batch_size=4096, columns=["row_id", "n_atoms"]),
                      total=(total + 4095) // 4096, desc="count"):
        for i in range(batch.num_rows):
            n_atoms_all[cursor] = int(batch.column("n_atoms")[i].as_py())
            row_id_all.append(batch.column("row_id")[i].as_py())
            (val_rows if is_val[cursor] else train_rows).append(cursor)
            cursor += 1

    def write_split(split_name, row_indices):
        split_dir = out_root / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        N = len(row_indices)
        sum_atoms = int(n_atoms_all[row_indices].sum())
        pos = np.zeros((sum_atoms, 3), dtype=np.float32)
        cell = np.zeros((N, 3, 3), dtype=np.float32)
        atomic_numbers = np.zeros((sum_atoms,), dtype=np.int64)
        num_atoms = np.zeros((N,), dtype=np.int64)
        structure_id = np.empty((N,), dtype=object)

        split_pos = {ri: i for i, ri in enumerate(row_indices)}
        atom_offset = np.zeros(N + 1, dtype=np.int64)
        atom_offset[1:] = np.cumsum(n_atoms_all[row_indices])

        cursor = 0
        for batch in tqdm(pf.iter_batches(batch_size=4096, columns=["row_id", "atoms_struct"]),
                          total=(total + 4095) // 4096, desc=f"build {split_name}"):
            for i in range(batch.num_rows):
                if cursor in split_pos:
                    j = split_pos[cursor]
                    struct = batch.column("atoms_struct")[i].as_py()
                    frac, c, Z = atoms_struct_to_arrays(struct)
                    n = frac.shape[0]
                    pos[atom_offset[j]:atom_offset[j] + n] = frac
                    atomic_numbers[atom_offset[j]:atom_offset[j] + n] = Z
                    cell[j] = c
                    num_atoms[j] = n
                    structure_id[j] = row_id_all[cursor]
                cursor += 1

        sid_max = max((len(s) for s in structure_id), default=8)
        np.save(split_dir / "pos.npy", pos)
        np.save(split_dir / "cell.npy", cell)
        np.save(split_dir / "atomic_numbers.npy", atomic_numbers)
        np.save(split_dir / "num_atoms.npy", num_atoms)
        np.save(split_dir / "structure_id.npy", np.array(structure_id, dtype=f"<U{sid_max}"))
        print(f"{split_name}: N={N:,} structures, {sum_atoms:,} atoms → {split_dir}")

    write_split("train", train_rows)
    write_split("val", val_rows)
    test_dir = out_root / "test"
    if not test_dir.exists():
        test_dir.symlink_to(out_root / "val")
    print(f"\nDone. {out_root}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs_parquet", required=True)
    parser.add_argument("--out_root", required=True)
    parser.add_argument("--val_fraction", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    main(parser.parse_args())
