#!/usr/bin/env bash
# Merge per-batch `generated_crystals.extxyz` files from any Stage 3 generation
# script (generate_stage3a.py, csp_resample_one_row.py, eval_dng.py's per-prompt
# subdirs) into one extxyz, filter out structures whose elements aren't in the
# hull reference's terminal systems, then run `mattergen-evaluate --relax=True`
# to produce the headline stability/SUN/MSUN/RMSD numbers.
#
# This is the "merge ad-hoc generations with the headline-eval pipeline"
# wrapper. Works equally well for a single behavioral run or a sweep — point it
# at the directory and it figures out the rest.
#
# Usage:
#   bash helper_scripts/score_manual_gens.sh SOURCE DEST [GPU] [REFERENCE]
#
#   SOURCE     a single .extxyz file, OR a directory (recursively searched for
#              generated_crystals.extxyz files).
#   DEST       output dir; receives merged.extxyz + mattergen_metrics.json +
#              mattergen_metrics_detailed.json.
#   GPU        CUDA_VISIBLE_DEVICES (default 7).
#   REFERENCE  hull reference path. Default: MP2020-corrected (matches our other
#              evals). Pass a TRI2024 path here for the broader-coverage eval.
#
# Examples:
#   # Score a single behavioral generation
#   bash helper_scripts/score_manual_gens.sh \
#       /home/sathyae/orcd/pool/stage3_outputs/stage3a/generations/run11_stage3b_K2_2node/g1_0_ood_topological \
#       /tmp/score_topological
#
#   # Score everything from a manual N=100 csp_resample run
#   bash helper_scripts/score_manual_gens.sh \
#       /tmp/csp_resample_mp-1225695_N100 \
#       /tmp/csp_resample_mp-1225695_N100_scored
#
#   # Score the conditional eval_dng output (already supported by the orchestrator,
#   # but useful when re-running just the eval)
#   bash helper_scripts/score_manual_gens.sh \
#       /home/sathyae/orcd/pool/eval_results/stage3b_dng_g10/run9_step4500_n32_cond_g10 \
#       /orcd/pool/003/sathyae/stage3_outputs/stage3b/run9_step4500_dng32_cond_g10_v2

set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 SOURCE DEST [GPU] [REFERENCE]" >&2
  exit 1
fi

SOURCE="${1%/}"
DEST="${2%/}"
GPU="${3:-7}"
REFERENCE="${4:-/home/sathyae/orcd/pool/eval_data/mp_hull/reference_MP2020correction.gz}"

if [[ ! -e "$SOURCE" ]]; then
  echo "error: source missing: $SOURCE" >&2
  exit 1
fi
if [[ ! -f "$REFERENCE" ]]; then
  echo "error: hull reference missing: $REFERENCE" >&2
  echo "       run: python helper_scripts/fetch_mp_hull.py" >&2
  exit 1
fi
if head -c 64 "$REFERENCE" 2>/dev/null | grep -q "git-lfs.github.com/spec"; then
  echo "error: $REFERENCE is still a Git LFS pointer." >&2
  echo "       run: cd external/mattergen && git lfs install --local && git lfs pull" >&2
  exit 1
fi

mkdir -p "$DEST"
MERGED="$DEST/merged.extxyz"
DROPPED="$DEST/merged_dropped.txt"
MANIFEST="$DEST/merged_manifest.txt"

# ── 1. Gather + filter (Python) ────────────────────────────────────────────
# - Recursively finds generated_crystals.extxyz files when SOURCE is a dir,
#   or treats SOURCE as a single extxyz file.
# - Drops structures whose elements aren't all present in the hull reference's
#   terminal systems, since mattergen-evaluate refuses to compute energy
#   metrics for the WHOLE batch when even one is missing (the Pu trap).
export MERGE_SOURCE="$SOURCE" MERGE_DEST="$DEST" MERGE_REF="$REFERENCE"
/home/sathyae/.conda/envs/llm/bin/python <<'PY'
import os, sys
from pathlib import Path
from ase.io import read, write

src = Path(os.environ["MERGE_SOURCE"])
dest = Path(os.environ["MERGE_DEST"])
ref = os.environ["MERGE_REF"]

# Discover input extxyz files
if src.is_file():
    files = [src]
else:
    files = sorted(src.rglob("generated_crystals.extxyz"))
    if not files:
        # Fall back: any *.extxyz under the tree
        files = sorted(src.rglob("*.extxyz"))

if not files:
    print(f"[merge] no extxyz files found under {src}", file=sys.stderr)
    sys.exit(2)

print(f"[merge] found {len(files)} extxyz file(s)")
with open(dest / "merged_manifest.txt", "w") as f:
    for p in files:
        f.write(str(p) + "\n")

# Resolve hull terminal systems (single-element entries) to get the allowed set.
# `entries_by_chemsys` keys are dash-joined element strings ('Au', 'Au-Cl',
# 'Au-Cl-La', ...). Terminal systems are the keys with no dash (single element).
sys.path.insert(0, str(Path("/home/sathyae/mclm/external/mattergen")))
from mattergen.evaluation.reference.reference_dataset_serializer import LMDBGZSerializer
hull = LMDBGZSerializer().deserialize(ref)
allowed = {k for k in hull.entries_by_chemsys.keys() if "-" not in k}
print(f"[merge] hull reference covers {len(allowed)} terminal systems")

kept, dropped = [], []
for fp in files:
    for atoms in read(fp, ":"):
        syms = set(atoms.get_chemical_symbols())
        missing = syms - allowed
        if missing:
            dropped.append((fp, atoms.get_chemical_formula(), sorted(missing)))
        else:
            kept.append(atoms)

print(f"[merge] kept {len(kept)} structures, dropped {len(dropped)}")
with open(dest / "merged_dropped.txt", "w") as f:
    for fp, formula, miss in dropped:
        f.write(f"{fp}\t{formula}\tmissing={','.join(miss)}\n")
write(dest / "merged.extxyz", kept, format="extxyz")
print(f"[merge] wrote {dest / 'merged.extxyz'} ({len(kept)} frames)")
PY

n_kept=$(grep -c '^Lattice=' "$MERGED" 2>/dev/null || echo 0)
echo "[merge] manifest:           $MANIFEST"
echo "[merge] dropped log:        $DROPPED  ($(wc -l < "$DROPPED") rows)"
echo "[merge] merged.extxyz:      $MERGED  ($n_kept frames)"
echo

if [[ "$n_kept" -lt 1 ]]; then
  echo "[merge] no surviving frames after element-filter; nothing to score." >&2
  exit 1
fi

# ── 2. Run mattergen-evaluate ──────────────────────────────────────────────
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONPATH="/home/sathyae/mclm/external/mattergen:${PYTHONPATH:-}"

DEVICE="cuda"
if ! /home/sathyae/.conda/envs/llm/bin/python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
  DEVICE="cpu"
  echo "[mg-eval] WARNING: CUDA unavailable, using CPU (slow)."
fi

echo "[mg-eval] structures: $MERGED ($n_kept frames)"
echo "[mg-eval] hull ref:   $REFERENCE"
echo "[mg-eval] device:     $DEVICE"
echo "[mg-eval] dest:       $DEST"
echo

mattergen-evaluate \
    "$MERGED" \
    --relax=True \
    --structure_matcher='disordered' \
    --reference_dataset_path="$REFERENCE" \
    --device="$DEVICE" \
    --save_as="$DEST/mattergen_metrics.json" \
    --save_detailed_as="$DEST/mattergen_metrics_detailed.json" \
    2>&1 | tee "$DEST/mattergen_evaluate.log"

echo
echo "[mg-eval] DONE. Headline:"
export METRICS_JSON="$DEST/mattergen_metrics.json"
/home/sathyae/.conda/envs/llm/bin/python - <<'PY'
import json, os
m = json.load(open(os.environ["METRICS_JSON"]))
def fmt(v):
    if isinstance(v, dict): v = v.get("value")
    if isinstance(v, float): return f"{v:.4f}"
    return str(v)
for k in sorted(m.keys()):
    print(f"  {k:42s} = {fmt(m[k])}")
PY
