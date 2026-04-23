"""One-time conversion: dict-of-numpy .pt → flat .bin + .idx.json pair.

The training cache at /home/sathyae/orcd/pool/cached_embs/{ds}/embeddings/
orb_v3_direct_20_omat_{split}_atom.pt is a `dict[str(id)] → ndarray(N, 256) float32`.
Loading the whole dict into every DDP rank blows host RAM (52 GB × 8 ranks ≈ 416 GB
on a 250 GB node). This script re-packs each cache into a flat mmap-friendly layout:

    {basename}.flat.bin         — concat float32 array, shape (total_atoms, 256) on disk
    {basename}.flat.idx.json    — {id: [offset, n_atoms]}

At train time the .bin is opened read-only with np.memmap; the Linux page cache
shares a single copy across all ranks on a node.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


EMBED_DIM = 256


def flatten_one(src_pt: Path, dst_bin: Path, dst_idx: Path):
    cache = torch.load(src_pt, map_location="cpu", weights_only=False)
    if not isinstance(cache, dict):
        raise RuntimeError(f"{src_pt}: expected dict, got {type(cache)}")

    # First pass: compute total atom count so we can preallocate the flat file.
    total_atoms = 0
    for arr in cache.values():
        if arr.ndim != 2 or arr.shape[1] != EMBED_DIM:
            raise RuntimeError(f"{src_pt}: unexpected shape {arr.shape}")
        total_atoms += arr.shape[0]

    dst_bin.parent.mkdir(parents=True, exist_ok=True)
    flat = np.memmap(dst_bin, dtype=np.float32, mode="w+", shape=(total_atoms, EMBED_DIM))

    index: dict[str, list[int]] = {}
    offset = 0
    for sample_id, arr in tqdm(cache.items(), desc=src_pt.name, leave=False):
        arr = np.ascontiguousarray(arr, dtype=np.float32)
        n = arr.shape[0]
        flat[offset : offset + n] = arr
        index[str(sample_id)] = [offset, n]
        offset += n
    assert offset == total_atoms

    flat.flush()
    del flat
    with open(dst_idx, "w") as f:
        json.dump(index, f)
    print(f"wrote {dst_bin} ({total_atoms:,} atoms) and {dst_idx} ({len(index):,} ids)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", type=str, default="/home/sathyae/orcd/pool/cached_embs",
                        help="parent dir containing {dataset}/embeddings/*.pt")
    parser.add_argument("--model_name", type=str, default="orb_v3_direct_20_omat")
    parser.add_argument("--splits", nargs="+", default=["train", "validation"])
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    parent = Path(args.parent)
    for ds_dir in sorted(parent.iterdir()):
        if not ds_dir.is_dir():
            continue
        for split in args.splits:
            src = ds_dir / "embeddings" / f"{args.model_name}_{split}_atom.pt"
            if not src.exists():
                print(f"skip (no cache): {src}")
                continue
            dst_bin = src.with_suffix(".flat.bin")
            dst_idx = src.with_suffix(".flat.idx.json")
            if dst_bin.exists() and dst_idx.exists() and not args.overwrite:
                print(f"skip (exists): {dst_bin}")
                continue
            flatten_one(src, dst_bin, dst_idx)


if __name__ == "__main__":
    main()
