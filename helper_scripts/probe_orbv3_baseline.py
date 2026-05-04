"""Baseline probe: do OrbV3 mean embeddings themselves linearly encode composition?

This is a sanity check on the auxiliary target signal. If OrbV3 means cluster
cleanly by has_O / has_transition_metal / most_common_element, then OrbV3 features
DO contain the structural info we want. Our run7b's failure (linear probe recall
0.008 from AtomsMapper output trained against OrbV3 means) was then about the
regression task not transferring well.

If OrbV3 means *don't* cluster cleanly, then OrbV3 never had a chance to teach
AtomsMapper composition — there's no element-level signal in the source.

Mirrors probe_atoms_mapper_clusters.py but operates directly on the precomputed
OrbV3 means (no ALM forward). Same plots, same probe, same metrics.

Usage:
  python helper_scripts/probe_orbv3_baseline.py \\
      --pairs_parquet  /home/sathyae/orcd/pool/stage3a/pairs.parquet \\
      --orbv3_means_path /home/sathyae/orcd/pool/stage3a/orbv3_means.bin \\
      --out_dir        /home/sathyae/orcd/pool/stage3a/diagnostics/orbv3_baseline \\
      --n_samples 800
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torch.nn.functional as F
import umap

# Reuse the helpers from probe_atoms_mapper_clusters
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from probe_atoms_mapper_clusters import (
    N_ELEMENTS, _symbol_to_z, composition_multihot,
    linear_probe_train, make_color_arrays, plot_2d,
)


def sample_held_out_rows(parquet_path, n_samples, seed):
    """Same as probe_atoms_mapper_clusters but only needs row_id + atoms_struct."""
    pf = pq.ParquetFile(parquet_path)
    total = pf.metadata.num_rows
    rng = np.random.default_rng(seed)
    target_idxs = sorted(rng.choice(total, size=min(n_samples, total), replace=False))
    target_set = set(int(i) for i in target_idxs)

    collected = []
    cursor = 0
    cols = ["row_id", "atoms_struct"]
    for batch in pf.iter_batches(batch_size=8192, columns=cols):
        rb = batch.to_pydict()
        n = len(rb["row_id"])
        for i in range(n):
            if cursor in target_set:
                collected.append({
                    "row_id": rb["row_id"][i],
                    "atoms_struct": rb["atoms_struct"][i],
                })
            cursor += 1
            if len(collected) == len(target_set):
                break
        if len(collected) == len(target_set):
            break
    return collected


def main(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load OrbV3 means + index ──────────────────────────────────────────
    bin_path = Path(args.orbv3_means_path)
    idx_path = bin_path.with_suffix(".idx.json")
    print(f"[probe] loading OrbV3 means from {bin_path} ...")
    with open(idx_path) as f:
        row_to_means_idx = json.load(f)
    means = np.memmap(bin_path, dtype=np.float32, mode="r").reshape(-1, 256)
    print(f"[probe] {means.shape[0]} OrbV3 means available, dim=256")

    # ── Sample rows + compute labels ──────────────────────────────────────
    print(f"[probe] sampling {args.n_samples} rows from {args.pairs_parquet} ...")
    rows = sample_held_out_rows(args.pairs_parquet, args.n_samples, args.seed)
    print(f"[probe] got {len(rows)} rows")

    sym2z = _symbol_to_z()
    comp = np.zeros((len(rows), N_ELEMENTS), dtype=np.float32)
    orbv3_outs = np.zeros((len(rows), 256), dtype=np.float32)
    missing = []
    for i, r in enumerate(rows):
        struct = r["atoms_struct"]
        if hasattr(struct, "as_py"):
            struct = struct.as_py()
        comp[i] = composition_multihot(struct["elements"], sym2z)
        midx = row_to_means_idx.get(r["row_id"])
        if midx is None:
            missing.append(r["row_id"])
        else:
            orbv3_outs[i] = means[midx]

    if missing:
        print(f"[probe] WARNING: {len(missing)} rows had no OrbV3 mean; first few: {missing[:5]}")

    print(f"[probe] orbv3_outs shape={orbv3_outs.shape}, "
          f"mean={orbv3_outs.mean():+.4f}, std={orbv3_outs.std():.4f}")

    # ── Linear probe ──────────────────────────────────────────────────────
    print(f"\n[probe] training linear probe (orbv3_mean → composition) ...")
    history, _ = linear_probe_train(
        orbv3_outs, comp,
        n_epochs=args.probe_epochs, lr=args.probe_lr,
        train_frac=0.8, seed=args.seed,
    )
    for h in history:
        print(f"  epoch={h['epoch']:4d}  train_loss={h['loss']:.4f}  "
              f"val_precision={h['val_precision']:.3f}  "
              f"val_recall={h['val_recall']:.3f}  "
              f"({h['n_pos_classes']} positive classes)")
    final = history[-1]
    print(f"\n[probe] FINAL: val precision={final['val_precision']:.3f}  "
          f"recall={final['val_recall']:.3f}")

    # ── Verdict for the OrbV3 source signal ───────────────────────────────
    rec = final["val_recall"]
    print(f"\n[probe] OrbV3-baseline verdict (recall={rec:.3f}):")
    if rec > 0.70:
        print("  ✅ OrbV3 means linearly encode composition strongly. Aux loss against")
        print("     OrbV3 SHOULD have transferred — failure was in the regression task,")
        print("     not the source signal. Likely fix: better OrbV3 head (deeper MLP) or")
        print("     longer training.")
    elif rec >= 0.40:
        print("  ⚠ OrbV3 means partially encode composition. Acceptable upper bound;")
        print("    AtomsMapper trained against OrbV3 might reach this.")
    else:
        print("  ❌ OrbV3 means themselves don't linearly encode composition.")
        print("     This means OrbV3-aux training had no chance to teach composition")
        print("     to AtomsMapper. Composition aux is the right target; OrbV3 mean was a")
        print("     mistake. Could still be useful as a *secondary* signal (composition")
        print("     primary, OrbV3 secondary) for fine-grained local-environment encoding.")

    # ── PCA / UMAP ─────────────────────────────────────────────────────────
    print(f"\n[probe] computing PCA + UMAP projections ...")
    colors = make_color_arrays(rows, sym2z)

    # Standardize before PCA
    centered = orbv3_outs - orbv3_outs.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    pca50 = U[:, :50] * S[:50]
    pca2 = U[:, :2] * S[:2]
    var_explained = (S ** 2) / (S ** 2).sum()
    print(f"[probe] PCA: top-2 var explained = "
          f"{var_explained[0]:.3f} + {var_explained[1]:.3f} = {var_explained[:2].sum():.3f}; "
          f"top-50 sum = {var_explained[:50].sum():.3f}")

    plot_2d(pca2, colors, "pca2", out_dir)

    print(f"[probe] running UMAP ...")
    reducer = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.1, random_state=args.seed)
    umap2 = reducer.fit_transform(pca50)
    plot_2d(umap2, colors, "umap", out_dir)

    # ── Save artifacts ─────────────────────────────────────────────────────
    np.savez(out_dir / "orbv3_baseline_arrays.npz",
             orbv3_outs=orbv3_outs, composition=comp, pca2=pca2, pca50=pca50)
    with open(out_dir / "orbv3_baseline_metrics.json", "w") as f:
        json.dump({
            "orbv3_means_path": str(bin_path),
            "n_samples":        len(rows),
            "n_missing":        len(missing),
            "linear_probe_history": history,
            "final_val_precision":  final["val_precision"],
            "final_val_recall":     final["val_recall"],
        }, f, indent=2)
    print(f"\n[probe] all artifacts in {out_dir}")
    print("        - {pca2,umap}_by_*.png plots")
    print("        - orbv3_baseline_arrays.npz")
    print("        - orbv3_baseline_metrics.json")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--pairs_parquet", required=True)
    p.add_argument("--orbv3_means_path", required=True,
                   help="Path to orbv3_means.bin built by precompute_orbv3_means.py")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--n_samples", type=int, default=800)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--probe_epochs", type=int, default=300)
    p.add_argument("--probe_lr", type=float, default=1e-2)
    main(p.parse_args())
