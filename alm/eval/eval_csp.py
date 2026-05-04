"""CSP eval — direct CrystaLLM head-to-head on MP-20 and MPTS-52.

For each test composition, generate `n` structures conditioned on a synthetic
prompt of the form:
    "Generate a crystal structure with formula {formula}, space group {sg}."
Then compare each generation against the ground-truth structure via
`OrderedStructureMatcher(ltol=0.3, stol=0.5, angle_tol=10)` (CDVAE / CrystaLLM
defaults — see `alm/eval/structure_metrics.py`).

Outputs (under `$ALM_EVAL_RESULTS_ROOT/stage3b_csp_{benchmark}/{run_id}/`):
  metrics.json
  predictions.jsonl     one row per test composition

Usage:
  python alm/eval/eval_csp.py \\
      --alm_checkpoint /home/sathyae/orcd/pool/stage3a/ckpts/run9_stage3b_2node/step=4500 \\
      --atoms_mapper   /home/sathyae/orcd/pool/stage3a/ckpts/run9_stage3b_2node/step=4500/atoms_mapper.pt \\
      --benchmark mp_20 \\
      --n 20 \\
      --max_rows 100 \\
      --guidance_factor 1.0
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import torch
from pymatgen.core.structure import Structure
from pymatgen.io.cif import CifParser
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

# ALM eval imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # alm/
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "external" / "mattergen"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "helper_scripts"))

from eval.runs import run_dir, write_run  # noqa: E402
from eval.structure_metrics import (  # noqa: E402
    cdvae_matcher,
    match_many,
    validity_full,
)


CSV_COLUMN_CANDIDATES = ("cif_string", "cif", "structure", "structure_cif")
DEFAULT_BENCH_ROOT = Path("/home/sathyae/orcd/pool/eval_data/csp")


def _safe_float(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def read_test_rows(benchmark: str, max_rows: int = -1, bench_root: Path | None = None):
    """Yield dicts with keys (row_id, formula, sg_symbol, crystal_system,
    ref_structure, formation_energy_per_atom, band_gap, e_above_hull,
    density, elements) from a CDVAE-format benchmark's test split.

    Tolerates the CIF column being `cif` / `cif_string` / etc. Optional
    property columns are extracted when present (CrystaLLM CSVs ship them).
    """
    bench_root = bench_root or DEFAULT_BENCH_ROOT
    test_csv = bench_root / benchmark / "test.csv"
    if not test_csv.exists():
        raise FileNotFoundError(
            f"missing {test_csv} — run helper_scripts/download_csp_benchmarks.sh"
        )
    with open(test_csv) as f:
        reader = csv.DictReader(f)
        cif_col = next((c for c in CSV_COLUMN_CANDIDATES if c in reader.fieldnames), None)
        if cif_col is None:
            raise ValueError(
                f"no recognized CIF column in {test_csv}; "
                f"have {reader.fieldnames}; expected one of {CSV_COLUMN_CANDIDATES}"
            )
        for i, row in enumerate(reader):
            if max_rows > 0 and i >= max_rows:
                break
            cif_str = row[cif_col]
            row_id = row.get("material_id") or row.get("id") or f"{benchmark}-{i}"
            try:
                struct = CifParser.from_str(cif_str).parse_structures(primitive=False)[0]
            except Exception as exc:
                print(f"[csp] skip {row_id}: CIF parse failed ({exc})")
                continue
            try:
                sg_analyzer = SpacegroupAnalyzer(struct)
                sg_symbol = sg_analyzer.get_space_group_symbol()
                crystal_system = sg_analyzer.get_crystal_system()
            except Exception:
                sg_symbol = "P1"
                crystal_system = "triclinic"
            formula = row.get("pretty_formula") or struct.composition.reduced_formula
            elements = sorted({str(el) for el in struct.composition.elements})
            yield {
                "row_id": row_id,
                "formula": formula,
                "sg_symbol": sg_symbol,
                "crystal_system": crystal_system,
                "ref_structure": struct,
                "formation_energy_per_atom": _safe_float(row.get("formation_energy_per_atom")),
                "band_gap": _safe_float(row.get("band_gap")),
                "e_above_hull": _safe_float(row.get("e_above_hull")),
                "density": float(struct.density),
                "elements": elements,
            }


# ─────────────────────────────────────────────────────────────────────────────
# Prompt templates
# ─────────────────────────────────────────────────────────────────────────────
# `minimal` is CrystaLLM-style (formula + sg). It's what we tested first; the
# 0/30 smoke result showed mclm doesn't strongly bias on this format because it
# was never trained on it.
#
# `rich_v1`, `rich_v2`, `rich_v3` mirror the three GPT-Narratives prose patterns
# the model actually saw at training time:
#   - rich_v1: dft_3d-style — single rich paragraph with property weaving
#   - rich_v2: mp_3d_2020-style — formula-first, compact property prose
#   - rich_v3: oqmd-style — header + bulleted Key Properties block
#
# All three put the FORMULA literal early (the strongest signal that survives
# AtomsMapper's compression) and mention space-group + crystal-system explicitly.

def _band_gap_descriptor(bg) -> str:
    if bg is None:
        return ""
    if bg <= 0.05:
        return "indicating that it is a metal"
    if bg < 1.0:
        return "indicating that it is a narrow-gap semiconductor"
    if bg < 3.0:
        return "indicating that it is a semiconductor"
    return "indicating that it is a wide-gap insulator"


def _stability_sentence(e_hull) -> str:
    if e_hull is None:
        return ""
    if e_hull <= 0.001:
        return "The material is considered stable."
    if e_hull <= 0.1:
        return "The material is considered metastable."
    return "The material is considered unstable."


def make_prompt_minimal(row: dict) -> str:
    """CrystaLLM-style — terse formula + space group only. Baseline."""
    return (
        f"Generate a crystal structure with formula {row['formula']}, "
        f"space group {row['sg_symbol']}."
    )


def make_prompt_rich_v1(row: dict) -> str:
    """dft_3d-style: single rich paragraph, formula + sg + properties woven in.

    Mirrors phrasing the model saw at training time on the dft_3d slice of
    GPT-Narratives.
    """
    parts = [
        f"The material with the formula {row['formula']} has a "
        f"{row['crystal_system']} crystal system with a space group symbol of "
        f"{row['sg_symbol']}."
    ]
    if row["density"] is not None:
        parts.append(f"It has a density of {row['density']:.3f} g/cm³.")
    if row["e_above_hull"] is not None:
        parts.append(
            f"The energy above the hull is {row['e_above_hull']:.4f} eV/atom."
        )
    if row["formation_energy_per_atom"] is not None:
        parts.append(
            f"The formation energy per atom is {row['formation_energy_per_atom']:.4f} eV/atom."
        )
    if row["band_gap"] is not None:
        bg_desc = _band_gap_descriptor(row["band_gap"])
        if bg_desc:
            parts.append(f"It has a band gap of {row['band_gap']:.4f} eV, {bg_desc}.")
        else:
            parts.append(f"It has a band gap of {row['band_gap']:.4f} eV.")
    stab = _stability_sentence(row.get("e_above_hull"))
    if stab:
        parts.append(stab)
    return " ".join(parts)


def make_prompt_rich_v2(row: dict) -> str:
    """mp_3d_2020-style: formula-first, compact property prose."""
    parts = [
        f"{row['formula']} is a {row['crystal_system']} crystalline material "
        f"with a space group symbol {row['sg_symbol']}."
    ]
    if row["formation_energy_per_atom"] is not None:
        parts.append(
            f"Its formation energy per atom is {row['formation_energy_per_atom']:.4f} eV."
        )
    if row["e_above_hull"] is not None:
        parts.append(
            f"The energy above the hull is {row['e_above_hull']:.4f} eV/atom."
        )
    if row["band_gap"] is not None:
        bg_desc = _band_gap_descriptor(row["band_gap"])
        if bg_desc:
            parts.append(
                f"The band gap of the material is {row['band_gap']:.4f} eV, {bg_desc}."
            )
        else:
            parts.append(f"The band gap of the material is {row['band_gap']:.4f} eV.")
    if row["density"] is not None:
        parts.append(
            f"The density of the material is {row['density']:.3f} grams per cubic centimeter."
        )
    return " ".join(parts)


def make_prompt_rich_v3(row: dict) -> str:
    """oqmd-style: header + bulleted Key Properties block."""
    elements_phrase = ", ".join(row["elements"][:-1]) + (
        f" and {row['elements'][-1]}" if len(row["elements"]) > 1 else row["elements"][0]
    )
    header = (
        f"The material under consideration is a {row['crystal_system']} compound "
        f"with the chemical formula {row['formula']}, space group {row['sg_symbol']}.\n\n"
        f"Key Properties:"
    )
    bullets = []
    if row["formation_energy_per_atom"] is not None:
        bullets.append(
            f"- Formation energy per atom: {row['formation_energy_per_atom']:.4f} eV/atom."
        )
    if row["band_gap"] is not None:
        bullets.append(f"- Band gap: {row['band_gap']:.4f} eV.")
    if row["e_above_hull"] is not None:
        bullets.append(
            f"- Energy above hull per atom: {row['e_above_hull']:.4f} eV/atom."
        )
    if row["density"] is not None:
        bullets.append(f"- Density: {row['density']:.3f} g/cm³.")
    bullets.append(f"- Elements in this compound: {elements_phrase}.")
    return header + "\n" + "\n".join(bullets)


PROMPT_TEMPLATES = {
    "minimal": make_prompt_minimal,
    "rich_v1": make_prompt_rich_v1,
    "rich_v2": make_prompt_rich_v2,
    "rich_v3": make_prompt_rich_v3,
}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--alm_checkpoint", required=True,
                    help="Stage 3b step= dir (with lora_adapter/ + projector_and_state.pt).")
    ap.add_argument("--atoms_mapper", required=True,
                    help="Path to atoms_mapper.pt produced by train_stage3a.py")
    ap.add_argument("--benchmark", required=True, choices=["mp_20", "mpts_52", "perov_5", "carbon_24"],
                    help="CDVAE/CrystaLLM-format benchmark to evaluate against.")
    ap.add_argument("--n", type=int, default=20,
                    help="Number of generations per composition (CrystaLLM reports 1 and 20; "
                         "this CLI accepts any positive int — useful for resample-many test-time "
                         "compute experiments where you crank N up to see if the upper-bound "
                         "match rate climbs).")
    ap.add_argument("--prompt_template", default="rich_v1",
                    choices=list(PROMPT_TEMPLATES.keys()),
                    help="Prompt format. `minimal` matches CrystaLLM's terse 'formula + sg' "
                         "prompt. `rich_v1/v2/v3` mirror GPT-Narratives prose patterns the "
                         "model saw at training time (dft_3d / mp_3d_2020 / oqmd respectively). "
                         "Default `rich_v1` is the closest training-distribution match.")
    ap.add_argument("--mask_elements", action=argparse.BooleanOptionalAction, default=True,
                    help="Hard-mask atomic-number logits at sample time so generations only "
                         "use elements that appear in the target row's `elements` field. "
                         "Default ON. Disable with --no-mask_elements for the unrestricted "
                         "baseline (what we measured at 0/100 lattice match before).")
    ap.add_argument("--max_rows", type=int, default=-1,
                    help="Cap test rows for smoke runs; -1 = full set.")
    ap.add_argument("--guidance_factor", type=float, default=1.0,
                    help="CFG guidance scale (run9 sweet spot is 1.0).")
    ap.add_argument("--mattergen_pretrained", default="mattergen_base")
    ap.add_argument("--num_atoms_distribution", default="ALEX_MP_20")
    ap.add_argument("--bench_root", type=Path, default=DEFAULT_BENCH_ROOT)
    ap.add_argument("--out_root", type=Path, default=None,
                    help="Override $ALM_EVAL_RESULTS_ROOT.")
    ap.add_argument("--run_id", type=str, default=None)
    args = ap.parse_args()

    # Lazy import: pulls in MatterGen + ALM (heavy)
    from generate_stage3a import generate_for_prompts, load_alm_and_pl_module  # noqa: E402

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    alm, tokenizer, pl_module, K = load_alm_and_pl_module(
        alm_checkpoint=args.alm_checkpoint,
        atoms_mapper=args.atoms_mapper,
        mattergen_pretrained=args.mattergen_pretrained,
        device=device,
    )

    # Resolve output directory under standard runs.py schema. Append the prompt
    # template to the benchmark name so different prompt-format experiments don't
    # collide in a single run_dir.
    tmpl_tag = "" if args.prompt_template == "minimal" else f"_{args.prompt_template}"
    benchmark_name = (
        f"stage3b_csp_{args.benchmark}_n{args.n}_g{int(args.guidance_factor*10):02d}"
        f"{tmpl_tag}"
    )
    if args.out_root is not None:
        os.environ["ALM_EVAL_RESULTS_ROOT"] = str(args.out_root)
    rd = run_dir(benchmark_name, args.alm_checkpoint, run_id=args.run_id)
    print(f"[csp] writing results to {rd}", flush=True)
    print(f"[csp] prompt_template={args.prompt_template}", flush=True)

    rows = list(read_test_rows(args.benchmark, max_rows=args.max_rows, bench_root=args.bench_root))
    print(f"[csp] loaded {len(rows)} test rows from {args.benchmark}", flush=True)

    if not rows:
        print("[csp] no rows — nothing to do", flush=True)
        return 0

    # Build prompts + ids in input order.
    prompt_fn = PROMPT_TEMPLATES[args.prompt_template]
    prompts = [prompt_fn(r) for r in rows]
    prompt_ids = [r["row_id"] for r in rows]
    if rows:
        # Show one example so the log captures what was actually fed to the model.
        print(f"[csp] example prompt for {prompt_ids[0]}:")
        for line in prompts[0].splitlines():
            print(f"      {line}")

    # Generation: one MatterGen call per prompt, batch_size=n.
    gen_root = rd / "generations"
    gen_root.mkdir(parents=True, exist_ok=True)
    allowed_elements_per_prompt = (
        [r["elements"] for r in rows] if args.mask_elements else None
    )
    if args.mask_elements:
        print(f"[csp] hard-masking atomic-number logits to per-row element sets "
              f"(e.g. row 0 → {rows[0]['elements']})", flush=True)
    structures_per_prompt = generate_for_prompts(
        prompts=prompts,
        alm=alm, tokenizer=tokenizer, pl_module=pl_module,
        out_root=gen_root,
        batch_size=args.n,
        num_batches=1,
        diffusion_guidance_factor=args.guidance_factor,
        num_atoms_distribution=args.num_atoms_distribution,
        prompt_ids=prompt_ids,
        save_meta=False,
        allowed_elements_per_prompt=allowed_elements_per_prompt,
    )

    # Score: match each prompt's generations against its ground-truth structure.
    matcher = cdvae_matcher()
    predictions = []
    n_matched_n1 = 0
    n_matched_nK = 0
    rmses_n1 = []
    rmses_nK = []
    n_invalid_geom = 0
    for r, gens in zip(rows, structures_per_prompt):
        rid = r["row_id"]
        formula = r["formula"]
        sg = r["sg_symbol"]
        ref = r["ref_structure"]
        # gens is list of pymatgen Structures (draw_samples_from_sampler returns them).
        gen_structs = []
        for g in gens:
            if isinstance(g, Structure):
                gen_structs.append(g)
            else:
                # Handles ASE Atoms or any other variant
                try:
                    gen_structs.append(AseAtomsAdaptor.get_structure(g))
                except Exception:
                    n_invalid_geom += 1
        if not gen_structs:
            predictions.append({
                "row_id": rid, "formula": formula, "space_group": sg,
                "n_gen": 0,
                "matched_n1": False, "matched_nK": False,
                "rmse_n1": None, "rmse_nK": None,
                "validity": {"geom_pct": 0.0, "charge_pct": 0.0},
                "skipped": True,
            })
            continue
        mm = match_many(gen_structs, ref, matcher=matcher)
        # Per-prompt validity rate
        v_geom = sum(validity_full(s)["geom"] for s in gen_structs) / len(gen_structs)
        v_charge = sum(validity_full(s)["charge"] for s in gen_structs) / len(gen_structs)
        if mm["matched_n1"]:
            n_matched_n1 += 1
            rmses_n1.append(mm["rmse_n1"])
        if mm["matched_nK"]:
            n_matched_nK += 1
            rmses_nK.append(mm["rmse_nK"])
        predictions.append({
            "row_id": rid, "formula": formula, "space_group": sg,
            "n_gen": len(gen_structs),
            "matched_n1": mm["matched_n1"], "rmse_n1": mm["rmse_n1"],
            "matched_nK": mm["matched_nK"], "rmse_nK": mm["rmse_nK"],
            "match_idx": mm["match_idx"],
            "validity": {"geom_pct": v_geom, "charge_pct": v_charge},
            "skipped": False,
        })

    n = len(predictions)
    n_scored = sum(1 for p in predictions if not p["skipped"])
    metrics = {
        "benchmark": args.benchmark,
        "n_test": n,
        "n_scored": n_scored,
        "n_invalid_geom_skipped": n_invalid_geom,
        "n": args.n,
        "guidance_factor": args.guidance_factor,
        "match_rate_n1": n_matched_n1 / n_scored if n_scored else 0.0,
        "match_rate_nK": n_matched_nK / n_scored if n_scored else 0.0,
        "rmse_n1": float(sum(rmses_n1) / len(rmses_n1)) if rmses_n1 else None,
        "rmse_nK": float(sum(rmses_nK) / len(rmses_nK)) if rmses_nK else None,
        "rmse_n1_min": float(min(rmses_n1)) if rmses_n1 else None,
        "rmse_nK_min": float(min(rmses_nK)) if rmses_nK else None,
        "alm_checkpoint": str(args.alm_checkpoint),
        "atoms_mapper": str(args.atoms_mapper),
    }
    write_run(rd, metrics, predictions)

    print()
    print(f"[csp] {args.benchmark} n={args.n} g={args.guidance_factor}")
    print(f"  match_rate@1  = {metrics['match_rate_n1']:.3f}")
    print(f"  match_rate@K  = {metrics['match_rate_nK']:.3f}")
    print(f"  rmse@1 (mean) = {metrics['rmse_n1']}")
    print(f"  rmse@K (mean) = {metrics['rmse_nK']}")
    print(f"  results in {rd}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
