"""Visualize a Stage 3a generation directory: render each structure to PNG (3 views)
and copy everything to the shared pool filesystem.

Usage:
  python helper_scripts/visualize_stage3a_gen.py \\
      --gen_dir /tmp/stage3a_gen \\
      --dest /home/sathyae/orcd/pool/stage3a/generations/smoke_step50
"""
import argparse
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from ase import io as ase_io
from ase.visualize.plot import plot_atoms


def render_structure(atoms, out_path: Path):
    """3-panel rendering: down each cell axis."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, rot, label in zip(
        axes, ["0x,0y,0z", "-90x,0y,0z", "0x,-90y,0z"], ["c-axis", "b-axis", "a-axis"]
    ):
        plot_atoms(atoms, ax, rotation=rot, radii=0.4, show_unit_cell=2)
        ax.set_axis_off()
        ax.set_title(label)
    formula = atoms.get_chemical_formula(empirical=False)
    cell = atoms.cell.cellpar()
    fig.suptitle(
        f"{formula}   "
        f"a={cell[0]:.2f} b={cell[1]:.2f} c={cell[2]:.2f} Å, "
        f"α={cell[3]:.1f}° β={cell[4]:.1f}° γ={cell[5]:.1f}°, "
        f"n_atoms={len(atoms)}",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main(args):
    gen_dir = Path(args.gen_dir)
    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)

    extxyz = gen_dir / "generated_crystals.extxyz"
    if not extxyz.exists():
        raise FileNotFoundError(f"missing {extxyz}")

    structures = ase_io.read(str(extxyz), index=":")
    print(f"[viz] read {len(structures)} structures from {extxyz}")

    summary_lines = ["idx\tformula\tn_atoms\tdensity_g_cm3\ta\tb\tc\talpha\tbeta\tgamma"]
    for i, atoms in enumerate(structures):
        formula = atoms.get_chemical_formula(empirical=False)
        cell = atoms.cell.cellpar()
        # density: g/cm^3
        masses = atoms.get_masses().sum()
        vol_A3 = atoms.cell.volume
        density = masses / vol_A3 / 0.6022  # amu / A^3 → g/cm^3
        png_path = dest / f"structure_{i:02d}_{formula}.png"
        render_structure(atoms, png_path)
        print(f"  [{i}] {formula} (n={len(atoms)}, d={density:.2f} g/cm^3) → {png_path.name}")
        summary_lines.append(
            f"{i}\t{formula}\t{len(atoms)}\t{density:.3f}\t"
            f"{cell[0]:.3f}\t{cell[1]:.3f}\t{cell[2]:.3f}\t"
            f"{cell[3]:.2f}\t{cell[4]:.2f}\t{cell[5]:.2f}"
        )

    # Copy raw outputs (extxyz, zip, meta.pt) to dest for archival.
    for fname in ["generated_crystals.extxyz",
                  "generated_crystals.zip",
                  "stage3a_inference_meta.pt"]:
        src = gen_dir / fname
        if src.exists():
            shutil.copy(src, dest / fname)
            print(f"  copied {fname}")

    (dest / "summary.tsv").write_text("\n".join(summary_lines) + "\n")
    print(f"\n[viz] done. Outputs in {dest}")
    print(f"  → {len(structures)} PNG renders + summary.tsv + raw extxyz/zip/meta.pt")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--gen_dir", required=True,
                   help="Output dir from generate_stage3a.py (contains generated_crystals.extxyz)")
    p.add_argument("--dest", required=True,
                   help="Destination on shared filesystem (e.g. /home/.../orcd/pool/...)")
    main(p.parse_args())
