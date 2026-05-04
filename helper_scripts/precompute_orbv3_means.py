"""One-shot: compute mean OrbV3 per-atom feature per row in pairs.parquet.

Reads the existing per-parent OrbV3 narrative caches
(/home/sathyae/orcd/pool/cached_embs_narratives/{parent}/embeddings/...), looks up each
pair's source structure by `(parent, source_idx)`, mean-pools the 256-d per-atom
features, and writes a single dense (n_pairs, 256) float32 array as the aux target
for `--aux_target_kind=orbv3_mean` in train_stage3a.py.

Output layout (mirrors the existing flat.bin/.idx.json convention):
  {out_dir}/orbv3_means.bin           (n_means, 256) float32, contiguous
  {out_dir}/orbv3_means.idx.json      {row_id: int row in means file}
  {out_dir}/orbv3_means.meta.json     provenance (cache_root, model name, n_pairs, n_missing)

Usage:
  python helper_scripts/precompute_orbv3_means.py \\
      --pairs_parquet /home/sathyae/orcd/pool/stage3a/pairs.parquet \\
      --cache_root    /home/sathyae/orcd/pool/cached_embs_narratives \\
      --model         orb_v3_direct_20_omat \\
      --out_dir       /home/sathyae/orcd/pool/stage3a

Cost: ~1 min CPU on 1.35M rows. Memory: ~1.4 GB output (1.35M × 256 × 4 bytes).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from tqdm import tqdm


def main(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs_path = Path(args.pairs_parquet)
    cache_root = Path(args.cache_root)

    # ── Inventory parents present in pairs.parquet ─────────────────────────
    pf = pq.ParquetFile(pairs_path)
    total = pf.metadata.num_rows
    print(f"reading {total:,} rows from {pairs_path}", flush=True)

    # Pre-load each parent's OrbV3 cache (memmap + idx)
    parents_seen = set()
    parent_caches: dict = {}  # parent → (memmap, idx_dict)

    def _load_parent_cache(parent: str):
        if parent in parent_caches:
            return parent_caches[parent]
        bin_path = cache_root / parent / "embeddings" / f"{args.model}_atom.flat.bin"
        idx_path = bin_path.with_suffix(".idx.json")
        if not bin_path.exists() or not idx_path.exists():
            print(f"  [warn] no cache for parent={parent} at {bin_path}")
            parent_caches[parent] = (None, None)
            return parent_caches[parent]
        memmap = np.memmap(bin_path, dtype=np.float32, mode="r").reshape(-1, 256)
        with open(idx_path) as f:
            idx = json.load(f)
        # idx maps str(row_idx_within_parent) → [byte_offset_in_atoms, n_atoms]
        # The "byte_offset" in cache_embeddings_narratives_orbv3.py is actually the
        # atom-row offset (since memmap is shape (-1, 256)).
        parent_caches[parent] = (memmap, idx)
        print(f"  loaded {parent}: memmap shape {memmap.shape}, {len(idx):,} indexed rows")
        return parent_caches[parent]

    # First pass: collect (row_id, parent, source_idx) triplets
    triplets = []
    for batch in pf.iter_batches(batch_size=8192, columns=["row_id", "parent", "source_idx"]):
        rb = batch.to_pydict()
        n = len(rb["row_id"])
        for i in range(n):
            triplets.append((rb["row_id"][i], rb["parent"][i], int(rb["source_idx"][i])))
            parents_seen.add(rb["parent"][i])

    print(f"\nparents seen in pairs.parquet: {sorted(parents_seen)}")
    for p in sorted(parents_seen):
        _load_parent_cache(p)

    # Allocate output
    means = np.zeros((total, 256), dtype=np.float32)
    row_to_means_idx: dict = {}     # row_id → its row in `means`
    n_missing = 0
    n_written = 0

    for i, (row_id, parent, source_idx) in enumerate(tqdm(triplets, desc="pooling")):
        memmap, idx = parent_caches[parent]
        if memmap is None or idx is None:
            n_missing += 1
            continue
        key = str(source_idx)
        if key not in idx:
            n_missing += 1
            continue
        offset, n_atoms = idx[key]
        chunk = memmap[offset : offset + n_atoms]   # (n_atoms, 256)
        means[n_written] = chunk.mean(axis=0)
        row_to_means_idx[row_id] = n_written
        n_written += 1

    means = means[:n_written]   # truncate unused
    print(f"\nwrote {n_written:,} means / {total:,} pairs ({n_missing:,} missing)")

    bin_out = out_dir / "orbv3_means.bin"
    idx_out = out_dir / "orbv3_means.idx.json"
    meta_out = out_dir / "orbv3_means.meta.json"

    means.tofile(bin_out)
    with open(idx_out, "w") as f:
        json.dump(row_to_means_idx, f)
    with open(meta_out, "w") as f:
        json.dump({
            "pairs_parquet": str(pairs_path),
            "cache_root": str(cache_root),
            "model": args.model,
            "n_pairs_total": total,
            "n_means_written": n_written,
            "n_missing": n_missing,
            "shape": list(means.shape),
            "dtype": "float32",
        }, f, indent=2)

    sz = bin_out.stat().st_size
    print(f"\nwrote:")
    print(f"  {bin_out}  ({sz/1e9:.2f} GB)")
    print(f"  {idx_out}")
    print(f"  {meta_out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--pairs_parquet", required=True)
    p.add_argument("--cache_root", required=True,
                   help="e.g. /home/sathyae/orcd/pool/cached_embs_narratives")
    p.add_argument("--model", default="orb_v3_direct_20_omat",
                   help="OrbV3 model name; matches the prefix of the .flat.bin file")
    p.add_argument("--out_dir", required=True,
                   help="Output dir (recommend /home/sathyae/orcd/pool/stage3a)")
    main(p.parse_args())
