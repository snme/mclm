"""Build the (text-prompt, structure) pairs parquet for Stage 3a training.

Reads each GPT-Narratives parquet (dft_3d, mp_3d_2020, aflow2, oqmd), filters rows
with > max_atoms atoms (default 20, MatterGen Alex-MP-20 distribution), and writes a
single combined parquet keyed by row_id := f"{parent}-{source_idx}".

Each row carries a *pre-templated* user_prompt (one of N paraphrases, deterministically
chosen per row) and an assistant_anchor (one of M anchor variants). This moves all
ChatML-formatting logic out of the trainer — the trainer just applies chat_template
and tokenizes. Two benefits:
  - Reproducibility: the exact prompt format each row sees is frozen at build time,
    not subject to script-side `idx % N` arithmetic. Held-out splits stay clean.
  - Inspectability: `head` on the parquet shows the literal prompt the model trained
    on, not "the narrative + an inferred template."

Output columns (one row per kept structure):
  row_id           str     unique global id ("{parent}-{source_idx}")
  parent           str     source parquet name
  source_idx       int64   original parquet row index within parent
  n_atoms          int32   number of atoms (= len(atoms.elements))
  narrative        str     raw GPT narrative (kept for debugging / reformulation)
  user_prompt      str     pre-templated user message — fed to ChatML user role
  assistant_anchor str     pre-templated anchor that prepends the K [atoms_i] tokens
  atoms_struct     struct  elements / coords / lattice_mat / cartesian — feeds ASE

The output parquet is consumed by:
  - alm/train_stage3a.py (Stage3aDataset reads user_prompt + assistant_anchor)
  - helper_scripts/generate_stage3a.py (canonical anchor at inference)

Usage:
  python helper_scripts/build_stage3a_pairs.py \\
      --out_path /home/sathyae/orcd/pool/stage3a/pairs.parquet
"""
import argparse
import hashlib
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm


DEFAULT_INPUT_ROOT = Path("/home/sathyae/orcd/pool/GPT-Narratives-for-Materials")
DEFAULT_PARENTS = ["dft_3d", "mp_3d_2020", "aflow2", "oqmd"]


# Diversified user-side templates. We rotate through these per-sample so the LLM
# doesn't memorize one phrasing — improves stylistic robustness at inference time.
USER_TEMPLATES = [
    "Generate a crystal structure described as: {narrative}",
    "Create the atomistic structure for the following material. {narrative}",
    "Produce a crystal matching this description: {narrative}",
    "Synthesize the structure for the material described below.\n\n{narrative}",
    "Given this description of a material, generate its crystal structure.\n\n{narrative}",
    "Output a crystal structure that fits the following: {narrative}",
    "Design a structure consistent with: {narrative}",
    "{narrative}\n\nGenerate the crystal structure for this material.",
]

# Anchor variants prepended to [atoms_i]. Light variation gives the model robustness
# to anchor phrasing at inference; literature standard is a brief preamble before
# the modality tokens (GILL/DreamLLM "text-then-special-tokens" pattern).
ASSISTANT_ANCHORS = [
    "Structure: ",
    "Generated structure: ",
    "Here is the structure: ",
    "The crystal structure: ",
    "Output: ",
]


def _det_pick(row_id: str, options: list[str]) -> str:
    """Deterministic pick from `options` keyed by row_id (stable hash, not Python's)."""
    h = int(hashlib.md5(row_id.encode()).hexdigest(), 16)
    return options[h % len(options)]


OUTPUT_SCHEMA = pa.schema([
    ("row_id", pa.string()),
    ("parent", pa.string()),
    ("source_idx", pa.int64()),
    ("n_atoms", pa.int32()),
    ("narrative", pa.string()),
    ("user_prompt", pa.string()),
    ("assistant_anchor", pa.string()),
    ("atoms_struct", pa.struct([
        ("elements", pa.list_(pa.string())),
        ("coords", pa.list_(pa.list_(pa.float64()))),
        ("lattice_mat", pa.list_(pa.list_(pa.float64()))),
        ("cartesian", pa.bool_()),
    ])),
])


def main(args):
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    writer = pq.ParquetWriter(out_path, OUTPUT_SCHEMA, compression="zstd")
    total_kept = 0
    total_seen = 0
    for parent in args.parents:
        parquet = Path(args.input_root) / f"{parent}_gpt_narratives.parquet"
        if not parquet.exists():
            print(f"skip {parent} (not found at {parquet})")
            continue
        pf = pq.ParquetFile(parquet)
        n = pf.metadata.num_rows
        print(f"{parent}: {n:,} rows")

        seen_in_parent = 0
        kept_in_parent = 0
        n_batches = (n + args.batch_size - 1) // args.batch_size
        for batch in tqdm(
            pf.iter_batches(batch_size=args.batch_size, columns=["atoms", "gpt_text"]),
            total=n_batches, desc=parent,
        ):
            atoms_arr = batch.column("atoms")
            gpt_arr = batch.column("gpt_text")

            row_ids = []
            parents = []
            source_idxs = []
            n_atoms_arr = []
            narratives = []
            user_prompts = []
            assistant_anchors = []
            structs = []
            for i in range(batch.num_rows):
                a = atoms_arr[i].as_py()
                source_idx = seen_in_parent + i
                if a is None or a.get("elements") is None:
                    continue
                n_at = len(a["elements"])
                if n_at == 0 or n_at > args.max_atoms:
                    continue
                narr = gpt_arr[i].as_py()
                if not narr:
                    continue
                row_id = f"{parent}-{source_idx}"
                user_prompt = _det_pick(row_id, USER_TEMPLATES).format(narrative=narr)
                assistant_anchor = _det_pick(row_id + "/anchor", ASSISTANT_ANCHORS)
                row_ids.append(row_id)
                parents.append(parent)
                source_idxs.append(source_idx)
                n_atoms_arr.append(n_at)
                narratives.append(narr)
                user_prompts.append(user_prompt)
                assistant_anchors.append(assistant_anchor)
                structs.append({
                    "elements": [s.strip() for s in a["elements"]],
                    "coords": a["coords"],
                    "lattice_mat": a["lattice_mat"],
                    "cartesian": bool(a["cartesian"]),
                })
            seen_in_parent += batch.num_rows
            if not row_ids:
                continue
            out_batch = pa.RecordBatch.from_pydict({
                "row_id": row_ids,
                "parent": parents,
                "source_idx": source_idxs,
                "n_atoms": pa.array(n_atoms_arr, pa.int32()),
                "narrative": narratives,
                "user_prompt": user_prompts,
                "assistant_anchor": assistant_anchors,
                "atoms_struct": pa.array(structs, OUTPUT_SCHEMA.field("atoms_struct").type),
            }, schema=OUTPUT_SCHEMA)
            writer.write_batch(out_batch)
            kept_in_parent += len(row_ids)

        total_seen += seen_in_parent
        total_kept += kept_in_parent
        print(f"  {parent}: kept {kept_in_parent:,} / {seen_in_parent:,}")

    writer.close()
    print(f"\nwrote {total_kept:,} pairs (kept) of {total_seen:,} (seen) → {out_path}")
    print(f"size: {out_path.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_root", type=str, default=str(DEFAULT_INPUT_ROOT))
    parser.add_argument("--parents", nargs="+", default=DEFAULT_PARENTS)
    parser.add_argument("--out_path", type=str,
                        default="/home/sathyae/orcd/pool/stage3a/pairs.parquet")
    parser.add_argument("--max_atoms", type=int, default=20,
                        help="MatterGen Alex-MP-20 distribution; rows above are dropped.")
    parser.add_argument("--batch_size", type=int, default=4096)
    args = parser.parse_args()
    main(args)
