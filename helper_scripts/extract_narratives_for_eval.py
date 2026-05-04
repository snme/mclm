"""Extract a few real narratives from pairs.parquet to disk for in-distribution Stage 3a
inference testing. Also writes ground-truth metadata (composition, lattice) so we can
compare conditional generations against the structure the narrative actually describes.

Usage:
  python helper_scripts/extract_narratives_for_eval.py \\
      --pairs_parquet /home/sathyae/orcd/pool/stage3a/pairs.parquet \\
      --row_ids mp_3d_2020-118833,oqmd-336794 \\
      --out_dir /home/sathyae/orcd/pool/stage3a/eval_prompts
"""
import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq


def main(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    target = set(args.row_ids.split(","))
    pf = pq.ParquetFile(args.pairs_parquet)

    found = {}
    for batch in pf.iter_batches(
        batch_size=8192,
        columns=["row_id", "parent", "n_atoms", "narrative", "atoms_struct"],
    ):
        rb = batch.to_pydict()
        for i in range(len(rb["row_id"])):
            rid = rb["row_id"][i]
            if rid in target:
                struct = rb["atoms_struct"][i]
                found[rid] = {
                    "row_id": rid,
                    "parent": rb["parent"][i],
                    "n_atoms": rb["n_atoms"][i],
                    "narrative": rb["narrative"][i],
                    "elements": list(struct["elements"]),
                    "lattice_mat": [list(r) for r in struct["lattice_mat"]],
                }
        if len(found) == len(target):
            break

    if len(found) < len(target):
        missing = target - set(found.keys())
        print(f"WARNING: did not find rows: {missing}")

    for rid, rec in found.items():
        safe_rid = rid.replace("/", "_")
        narr_path = out_dir / f"{safe_rid}.narrative.txt"
        meta_path = out_dir / f"{safe_rid}.meta.json"
        narr_path.write_text(rec["narrative"])
        meta_path.write_text(json.dumps({k: v for k, v in rec.items() if k != "narrative"},
                                        indent=2))
        print(f"  wrote {narr_path.name} ({len(rec['narrative'])} chars) "
              f"+ {meta_path.name} (n_atoms={rec['n_atoms']}, elements={rec['elements']})")

    print(f"\nDone. {len(found)} narratives in {out_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--pairs_parquet", required=True)
    p.add_argument("--row_ids", required=True,
                   help="Comma-separated row IDs (e.g. dft_3d-0,oqmd-12345)")
    p.add_argument("--out_dir", required=True)
    main(p.parse_args())
