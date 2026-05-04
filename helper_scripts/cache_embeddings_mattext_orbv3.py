"""Pre-compute OrbV3 atom-wise embeddings for the MatText (n0w0f/MatText) train-filtered
configs, dedupe across folds, and dump per-fold label CSVs.

Why we need this: to test whether ALM breaks the GNN-LLM wall (Alampara et al.
NeurIPS 2024 D&B), we need to LoRA fine-tune ALM on the MatText train splits and
eval on the held-out fold. Live-encoding ~70k structures per task on every fine-tune
run is wasteful — cache OrbV3 once, train many times.

Source: HuggingFace `n0w0f/MatText`, configs:
  - perovskites-train-filtered (~42k unique mbids; targets = heat of formation in eV/cell)
  - kvrh-train-filtered        (~21k unique mbids; targets = log10(K-VRH))
  - gvrh-train-filtered        (~22k unique mbids; targets = log10(G-VRH))

Each config ships 5 folds (`fold_0`..`fold_4`); for cross-validation, structures
recur across folds. We dedupe by mbid so each structure is OrbV3-encoded once.

Output layout (per task):
  {out_root}/mattext/{task}/atoms.db                        — ASE SQLite of unique structures
  {out_root}/mattext/{task}/atoms.id_index.json             — {mbid: ase_row_id}
  {out_root}/mattext/{task}/embeddings/{model}_atom.flat.bin
  {out_root}/mattext/{task}/embeddings/{model}_atom.flat.idx.json  — {mbid: [offset, n_atoms]}
  {out_root}/mattext/{task}/fold_{k}.csv                    — mbid, label (one row per
                                                              fold; ALM training reads
                                                              these as the CSV split)

The .idx.json key is the parquet column `mbid` (e.g. "mb-perovskites-00001"),
matching what AtomisticLanguageDataset expects when `cached_embs_path` is set.

Usage (run on a node with a GPU):
  python helper_scripts/cache_embeddings_mattext_orbv3.py \
      --tasks perovskites,kvrh,gvrh \
      --out_root /home/sathyae/orcd/pool/cached_embs_mattext \
      --batch_size 32

  # Subset for a smoke test:
  python helper_scripts/cache_embeddings_mattext_orbv3.py \
      --tasks perovskites --max_rows 200
"""
import argparse
import json
from io import StringIO
from pathlib import Path

import numpy as np
import polars as pl
import torch
from ase.db import connect
from ase.io import read as ase_read
from datasets import load_dataset
from tqdm import tqdm

from orb_models.forcefield import atomic_system, pretrained
from orb_models.forcefield.base import batch_graphs


# n0w0f/MatText config names + the column carrying the numeric target.
_CONFIGS = {
    "perovskites": "perovskites-train-filtered",
    "kvrh":        "kvrh-train-filtered",
    "gvrh":        "gvrh-train-filtered",
}
_FOLDS = ["fold_0", "fold_1", "fold_2", "fold_3", "fold_4"]


def _cif_to_atoms(cif_str):
    return ase_read(StringIO(cif_str), format="cif")


def _collect_unique_rows(task, max_rows=None):
    """Walk all 5 folds of {task}-train-filtered, dedupe by mbid, and emit
    (mbid, ase_atoms) for each unique structure plus a fold-membership map
    {mbid: {fold_idx: label_value}} for the per-fold CSV writes.
    """
    config = _CONFIGS[task]
    seen = {}        # mbid -> ase Atoms (encode once)
    fold_labels = {}  # mbid -> {fold_idx: label}
    for k, fold in enumerate(_FOLDS):
        ds = load_dataset("n0w0f/MatText", config, split=fold)
        if max_rows:
            ds = ds.select(range(min(max_rows, len(ds))))
        for row in ds:
            mbid = row["mbid"]
            if mbid not in seen:
                seen[mbid] = _cif_to_atoms(row["cif_p1"])
            fold_labels.setdefault(mbid, {})[k] = float(row["labels"])
    return seen, fold_labels


def _encode_and_write(task, mbid_to_atoms, args, device, orbff):
    """Stream the unique structures through OrbV3 in batches; write the flat.bin
    + idx.json + ASE db for this task."""
    out_dir = Path(args.out_root) / "mattext" / task
    embs_dir = out_dir / "embeddings"
    embs_dir.mkdir(parents=True, exist_ok=True)

    bin_path = embs_dir / f"{args.model_name}_atom.flat.bin"
    idx_path = embs_dir / f"{args.model_name}_atom.flat.idx.json"
    db_path = out_dir / "atoms.db"
    db_idx_path = out_dir / "atoms.id_index.json"
    if db_path.exists():
        db_path.unlink()   # ASE db appends; start fresh

    db = connect(db_path)
    id_index = {}
    index = {}
    byte_offset = 0

    items = list(mbid_to_atoms.items())   # stable order; small enough to materialize
    with open(bin_path, "wb") as fout, tqdm(total=len(items), desc=f"{task}") as pbar:
        for chunk_start in range(0, len(items), args.batch_size):
            chunk = items[chunk_start : chunk_start + args.batch_size]
            graphs = [
                atomic_system.ase_atoms_to_atom_graphs(a, orbff.system_config, device=device)
                for _, a in chunk
            ]
            graph = batch_graphs(graphs)
            node_feats = orbff.model(graph)["node_features"].detach().cpu().numpy().astype(np.float32)
            n_per = graph.n_node.cpu().tolist()
            ptr = 0
            for (mbid, atoms), n_atoms in zip(chunk, n_per):
                node_feats[ptr : ptr + n_atoms].tofile(fout)
                index[mbid] = [byte_offset, n_atoms]
                id_index[mbid] = db.write(atoms, data={"smiles": mbid})
                byte_offset += n_atoms
                ptr += n_atoms
            pbar.update(len(chunk))

    with open(idx_path, "w") as f:
        json.dump(index, f)
    with open(db_idx_path, "w") as f:
        json.dump(id_index, f)
    print(f"[{task}] wrote {byte_offset:,} atoms ({bin_path.stat().st_size / 1e9:.2f} GB) "
          f"+ {len(index):,} structures → {bin_path}")
    return out_dir


def _write_fold_csvs(task, out_dir, fold_labels):
    """One CSV per fold (matches what AtomisticLanguageDataset eats: id column +
    label column). Caller picks one fold as eval and concatenates the other four
    as train. The id column name is `mbid` so AtomisticLanguageDataset's id_name
    keeps a stable convention.
    """
    LABEL_COL = "labels"
    rows_by_fold = {k: [] for k in range(len(_FOLDS))}
    for mbid, fmap in fold_labels.items():
        for k, label in fmap.items():
            rows_by_fold[k].append((mbid, label))
    for k, rows in rows_by_fold.items():
        if not rows:
            continue
        df = pl.DataFrame({"mbid": [r[0] for r in rows], LABEL_COL: [r[1] for r in rows]})
        path = out_dir / f"fold_{k}.csv"
        df.write_csv(path)
        print(f"[{task}] fold_{k}: {len(rows):,} rows → {path}")


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("[warn] no GPU detected; OrbV3 encoding will be very slow")
    orbff = getattr(pretrained, args.model_name)(device=device, precision="float32-high")

    tasks = [t.strip() for t in args.tasks.split(",")]
    for task in tasks:
        if task not in _CONFIGS:
            print(f"[skip] unknown task {task}; known: {list(_CONFIGS)}")
            continue
        print(f"\n=== {task} ===")
        mbid_to_atoms, fold_labels = _collect_unique_rows(task, max_rows=args.max_rows)
        print(f"[{task}] {len(mbid_to_atoms):,} unique structures across {len(_FOLDS)} folds")
        out_dir = _encode_and_write(task, mbid_to_atoms, args, device, orbff)
        _write_fold_csvs(task, out_dir, fold_labels)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", default="perovskites,kvrh,gvrh",
                        help="Comma list of MatText tasks to cache")
    parser.add_argument("--out_root", default="/home/sathyae/orcd/pool/cached_embs_mattext",
                        help="Output root; per-task layout is {root}/mattext/{task}/")
    parser.add_argument("--model_name", default="orb_v3_direct_20_omat")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_rows", type=int, default=0,
                        help="Cap rows PER FOLD for smoke tests (0 = full)")
    args = parser.parse_args()
    if args.max_rows == 0:
        args.max_rows = None
    main(args)
