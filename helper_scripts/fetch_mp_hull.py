"""Resolve the MP convex-hull reference for E_hull scoring.

Two paths, tried in order:

1. **MatterGen-bundled** (default): the submodule ships
   `external/mattergen/data-release/alex-mp/reference_{MP2020,TRI2024}correction.gz`,
   a gzipped LMDB serialization of `ReferenceDataset[ComputedStructureEntry]`. We
   copy/symlink this into `/home/sathyae/orcd/pool/eval_data/mp_hull/` so the
   eval scripts have a stable path that's not coupled to the submodule layout.

2. **mp-api fallback**: if the bundled file is missing, query Materials Project
   for entries with `MP_API_KEY` set in env. Writes the entries as a pickled
   list to `/home/sathyae/orcd/pool/eval_data/mp_hull/reference_mp_api.pkl`.

Either way, downstream eval scripts call
`alm.eval.structure_metrics.load_hull_reference()` — that function knows the
preferred path order and loads transparently.

Usage:
    python helper_scripts/fetch_mp_hull.py [--variant MP2020|TRI2024] [--force]
"""
from __future__ import annotations

import argparse
import os
import pickle
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MATTERGEN_BUNDLED = REPO_ROOT / "external" / "mattergen" / "data-release" / "alex-mp"
DEST_ROOT = Path("/home/sathyae/orcd/pool/eval_data/mp_hull")


def resolve_bundled(variant: str) -> Path | None:
    fname = f"reference_{variant}correction.gz"
    candidate = MATTERGEN_BUNDLED / fname
    if not candidate.exists():
        return None
    # Detect Git LFS pointer (the file is on disk but its contents are still a
    # ~130-byte LFS pointer rather than the actual gzipped reference dataset).
    try:
        with open(candidate, "rb") as f:
            head = f.read(64)
        if head.startswith(b"version https://git-lfs.github.com/spec"):
            print(
                f"[fetch] {candidate} is a Git LFS pointer, not the actual data.\n"
                f"        Run:\n"
                f"          cd /home/sathyae/mclm/external/mattergen && \\\n"
                f"            git lfs install --local && git lfs pull\n"
                f"        Then re-run this script.",
                file=sys.stderr,
            )
            return None
    except Exception:
        pass
    return candidate


def link_bundled(src: Path, dest_dir: Path, force: bool) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / src.name
    if dest.exists() and not force:
        print(f"[fetch] already present: {dest}")
        return dest
    if dest.exists():
        dest.unlink()
    # symlink (cheap; bundled file is in the repo)
    dest.symlink_to(src.resolve())
    print(f"[fetch] linked {src} → {dest}")
    return dest


def fetch_via_mp_api(out_path: Path) -> Path:
    api_key = os.environ.get("MP_API_KEY")
    if not api_key:
        print(
            "[fetch] MP_API_KEY not set and no MatterGen-bundled reference found. "
            "Either set MP_API_KEY (https://next-gen.materialsproject.org/api) or "
            "run `bash external/setup_mattergen.sh` to ensure the submodule is "
            "checked out with its data-release/ directory.",
            file=sys.stderr,
        )
        sys.exit(2)
    try:
        from mp_api.client import MPRester  # type: ignore
    except Exception:
        print(
            "[fetch] mp-api not installed. Install via `pip install mp-api>=0.4` "
            "or use the MatterGen-bundled file (preferred).",
            file=sys.stderr,
        )
        sys.exit(3)
    print("[fetch] querying Materials Project for thermo entries...")
    with MPRester(api_key) as mpr:
        # Pull ComputedStructureEntry objects for ALL stable + within-hull ICSD entries.
        # Conservative bound: e_above_hull < 0.2 keeps the hull dense without exploding
        # disk; tighten/loosen as needed.
        entries = mpr.thermo.search(thermo_types=["GGA_GGA+U"])
        # `entries` is a list of ThermoDoc; convert to ComputedStructureEntry.
        cse_list = []
        for d in entries:
            cse = d.entries.get("GGA_GGA+U")
            if cse is None:
                cse = next(iter(d.entries.values()))
            cse_list.append(cse)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(cse_list, f)
    print(f"[fetch] wrote {len(cse_list)} ComputedStructureEntry objects to {out_path}")
    return out_path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--variant", choices=["MP2020", "TRI2024"], default="MP2020",
                   help="Which MatterGen-bundled hull to use. MP2020 matches the "
                        "Materials Project default and is what most papers use.")
    p.add_argument("--force", action="store_true",
                   help="Re-link/re-fetch even if destination already exists.")
    p.add_argument("--out_dir", type=Path, default=DEST_ROOT,
                   help="Where to place the resolved reference (default: "
                        "/home/sathyae/orcd/pool/eval_data/mp_hull/).")
    args = p.parse_args()

    bundled = resolve_bundled(args.variant)
    if bundled is not None:
        print(f"[fetch] using MatterGen-bundled {args.variant}: {bundled}")
        link_bundled(bundled, args.out_dir, force=args.force)
        # Drop a tiny sentinel so structure_metrics.load_hull_reference() picks the right path.
        marker = args.out_dir / "preferred.txt"
        marker.write_text(f"reference_{args.variant}correction.gz\n")
        print(f"[fetch] preferred marker written: {marker}")
        return 0

    print(f"[fetch] no bundled reference at {MATTERGEN_BUNDLED} — trying mp-api...")
    out_pkl = args.out_dir / "reference_mp_api.pkl"
    if out_pkl.exists() and not args.force:
        print(f"[fetch] already present: {out_pkl}")
        return 0
    fetch_via_mp_api(out_pkl)
    marker = args.out_dir / "preferred.txt"
    marker.write_text("reference_mp_api.pkl\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
