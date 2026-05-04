"""Test-time-compute experiment: for ONE MP-20/MPTS-52 test row, sample N
structures from the SAME prompt and report:
  - composition_match_rate: fraction of generations whose reduced formula
    matches the target's reduced formula exactly
  - lattice_match_rate: fraction of generations that pass StructureMatcher
    (CDVAE tolerances) against the target structure
  - lattice_match | composition_match: conditional rate — given the formula
    is right, how often does the lattice also match?

Decomposes the eval_csp.py 0/30 result into:
  (a) "model never generates the right formula" → upstream prompt-conditioning
      problem; rich_v1 prompt should help.
  (b) "model generates right formula but lattice is off" → fundamental
      mclm-task mismatch (mclm is generative, not a CSP solver).

Usage (after eval_csp.py has been run, so we have a row_id to target):
  python helper_scripts/csp_resample_one_row.py \\
      --alm_checkpoint /home/sathyae/orcd/pool/stage3a/ckpts/run9_stage3b_2node/step=4500 \\
      --atoms_mapper   /home/sathyae/orcd/pool/stage3a/ckpts/run9_stage3b_2node/step=4500/atoms_mapper.pt \\
      --benchmark mp_20 \\
      --row_id mp-1225695 \\
      --N 100 \\
      --batch_size 20 \\
      --prompt_template rich_v1
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from pymatgen.core.structure import Structure
from pymatgen.io.ase import AseAtomsAdaptor

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "alm"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "external" / "mattergen"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "helper_scripts"))

from eval.eval_csp import read_test_rows, PROMPT_TEMPLATES  # noqa: E402
from eval.structure_metrics import (  # noqa: E402
    cdvae_matcher, match_one,
    e_above_hull_per_atom, load_hull_reference,
    relax_structures_mattersim, total_energy_per_atom,
    validity_full,
)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--alm_checkpoint", required=True)
    ap.add_argument("--atoms_mapper", required=True)
    ap.add_argument("--benchmark", required=True,
                    choices=["mp_20", "mpts_52", "perov_5", "carbon_24"])
    ap.add_argument("--row_id", required=True,
                    help="Specific test row to target (e.g. mp-1225695). "
                         "Looked up in <bench>/test.csv.")
    ap.add_argument("--N", type=int, default=100,
                    help="Total number of structures to generate from the same prompt.")
    ap.add_argument("--batch_size", type=int, default=20,
                    help="Per-MatterGen-call batch size; total ≈ N rounded up to a multiple.")
    ap.add_argument("--prompt_template", default="rich_v1",
                    choices=list(PROMPT_TEMPLATES.keys()))
    ap.add_argument("--guidance_factor", type=float, default=1.0)
    ap.add_argument("--mattergen_pretrained", default="mattergen_base")
    ap.add_argument("--num_atoms_distribution", default="ALEX_MP_20")
    ap.add_argument("--max_search_rows", type=int, default=20000,
                    help="How far into test.csv to scan for the row_id. -1 = full.")
    ap.add_argument("--out_dir", default=None,
                    help="Where to drop generated CIFs + summary. Default: "
                         "/tmp/csp_resample_<row_id>_N<N>.")
    ap.add_argument("--relax_and_score", action="store_true",
                    help="Relax all N generations with MatterSim and compute E_hull + "
                         "displacement vs target. Lets us distinguish the two failure "
                         "modes for composition-matched generations: (A) valid alternative "
                         "polymorph (E_hull ≤ 0.1, small displacement) vs (B) unphysical "
                         "lattice (E_hull ≫ 0.1 or large displacement). Adds ~15s × N "
                         "to wallclock — for N=100, ~25 min.")
    ap.add_argument("--mattersim_potential_path", default=None,
                    help="MatterSim checkpoint path; None = MatterSim's default.")
    ap.add_argument("--relax_chunk_size", type=int, default=20,
                    help="Per-call BatchRelaxer chunk size. Smaller = better isolation if "
                         "a single structure crashes the relax (the rest of the chunk is "
                         "lost when one fails). Default 20.")
    ap.add_argument("--mask_elements", action=argparse.BooleanOptionalAction, default=True,
                    help="Hard-mask atomic-number logits to only allow elements in the "
                         "target row's `elements` field. Default ON. Disable with "
                         "--no-mask_elements for the unmasked baseline.")
    args = ap.parse_args()

    # ── 1. Find the target row ───────────────────────────────────────────────
    target_row = None
    for r in read_test_rows(args.benchmark, max_rows=args.max_search_rows):
        if r["row_id"] == args.row_id:
            target_row = r
            break
    if target_row is None:
        raise ValueError(
            f"row_id={args.row_id} not found in first {args.max_search_rows} rows of "
            f"{args.benchmark}/test.csv. Pass --max_search_rows -1 to scan the whole file."
        )

    target_struct: Structure = target_row["ref_structure"]
    target_formula = target_struct.composition.reduced_formula
    prompt_fn = PROMPT_TEMPLATES[args.prompt_template]
    prompt = prompt_fn(target_row)
    print(f"\n[resample] target row : {args.row_id}")
    print(f"[resample] target formula: {target_formula}")
    print(f"[resample] target sg     : {target_row['sg_symbol']}")
    print(f"[resample] target cell   : a={target_struct.lattice.a:.3f} "
          f"b={target_struct.lattice.b:.3f} c={target_struct.lattice.c:.3f}")
    print(f"[resample] template      : {args.prompt_template}")
    print(f"[resample] prompt:")
    for line in prompt.splitlines():
        print(f"           {line}")
    print(f"[resample] N={args.N}, batch_size={args.batch_size}, g={args.guidance_factor}")
    print()

    # ── 2. Load model + sample ──────────────────────────────────────────────
    from generate_stage3a import generate_for_prompts, load_alm_and_pl_module  # noqa: E402

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    alm, tokenizer, pl_module, _ = load_alm_and_pl_module(
        alm_checkpoint=args.alm_checkpoint,
        atoms_mapper=args.atoms_mapper,
        mattergen_pretrained=args.mattergen_pretrained,
        device=device,
    )

    out_dir = Path(args.out_dir or f"/tmp/csp_resample_{args.row_id}_N{args.N}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Issue ceil(N / batch_size) MatterGen calls, each producing batch_size structures.
    n_calls = (args.N + args.batch_size - 1) // args.batch_size
    prompts = [prompt] * n_calls
    prompt_ids = [f"{args.row_id}__chunk_{i:03d}" for i in range(n_calls)]
    allowed_elements_per_prompt = (
        [target_row["elements"]] * n_calls if args.mask_elements else None
    )
    if args.mask_elements:
        print(f"[resample] hard-masking atomic numbers to allowed elements: "
              f"{target_row['elements']}", flush=True)
    structures_per_chunk = generate_for_prompts(
        prompts=prompts,
        alm=alm, tokenizer=tokenizer, pl_module=pl_module,
        out_root=out_dir,
        batch_size=args.batch_size,
        num_batches=1,
        diffusion_guidance_factor=args.guidance_factor,
        num_atoms_distribution=args.num_atoms_distribution,
        prompt_ids=prompt_ids,
        save_meta=False,
        allowed_elements_per_prompt=allowed_elements_per_prompt,
    )
    raw = []
    for chunk in structures_per_chunk:
        for s in chunk:
            raw.append(s if isinstance(s, Structure) else AseAtomsAdaptor.get_structure(s))
    raw = raw[:args.N]
    print(f"\n[resample] generated {len(raw)} structures (target was {args.N})")

    # ── 3. Decompose: composition match vs lattice match ────────────────────
    matcher = cdvae_matcher()
    n_comp_match = 0
    n_lattice_match = 0
    n_lattice_match_given_comp = 0
    n_comp_match_total = 0
    rmses_lattice = []
    detail = []
    for i, s in enumerate(raw):
        gen_formula = s.composition.reduced_formula
        comp_match = (gen_formula == target_formula)
        lattice_match, rmse = match_one(s, target_struct, matcher=matcher)
        if comp_match:
            n_comp_match += 1
            n_comp_match_total += 1
            if lattice_match:
                n_lattice_match_given_comp += 1
        if lattice_match:
            n_lattice_match += 1
            if rmse is not None:
                rmses_lattice.append(rmse)
        detail.append({
            "idx": i, "formula": gen_formula,
            "n_atoms": len(s),
            "density": float(s.density),
            "composition_match": comp_match,
            "lattice_match": lattice_match,
            "rmse": rmse,
        })

    # ── 4. Report ───────────────────────────────────────────────────────────
    print()
    print(f"[resample] === RESULTS for row_id={args.row_id} target_formula={target_formula} ===")
    print(f"  composition_match: {n_comp_match}/{len(raw)} = {n_comp_match/len(raw):.3f}")
    print(f"  lattice_match    : {n_lattice_match}/{len(raw)} = {n_lattice_match/len(raw):.3f}")
    if n_comp_match_total:
        print(f"  P(lattice | composition) = {n_lattice_match_given_comp}/{n_comp_match_total} = "
              f"{n_lattice_match_given_comp/n_comp_match_total:.3f}")
    else:
        print(f"  P(lattice | composition) = N/A (no composition matches)")
    if rmses_lattice:
        import statistics
        print(f"  RMSE among lattice matches: mean={statistics.mean(rmses_lattice):.4f}, "
              f"min={min(rmses_lattice):.4f}, max={max(rmses_lattice):.4f}")

    # ── 4b. (Optional) MatterSim relax + E_hull stability scoring ───────────
    # Disambiguates two failure modes for composition-matched generations:
    #  (A) valid alternative polymorph: low E_hull (≤ 0.1 eV/atom) and small
    #      relax displacement (< ~0.5 Å mean). The lattice is physical, just
    #      not the *specific* MP-20 entry's polymorph. CSP failure here is a
    #      task-framing issue, not a model failure.
    #  (B) unphysical lattice: high E_hull or large relax displacement. The
    #      composition match was incidental; the geometry is broken.
    # Off by default (~25 min for N=100). Enable with --relax_and_score.
    if args.relax_and_score and raw:
        import numpy as np
        print()
        print(f"[resample] === STABILITY: relaxing {len(raw)} structures with MatterSim ===")
        # Pre-validity check — only relax geom-valid structures (mattersim crashes
        # on overlapping atoms).
        prevalid = [validity_full(s) for s in raw]
        relax_idx = [i for i, v in enumerate(prevalid) if v["geom"]]
        print(f"[resample] {len(relax_idx)}/{len(raw)} are geom-valid; "
              f"skipping {len(raw) - len(relax_idx)} for relax.")

        e_per_atom = [None] * len(raw)
        e_hull_arr = [None] * len(raw)
        relax_displacement = [None] * len(raw)

        # Relax in chunks for crash isolation. mattersim's BatchRelaxer can
        # hard-fail on a single bad structure, taking the whole batch with it.
        chunk_size = max(1, args.relax_chunk_size)
        try:
            hull_ref = load_hull_reference()
        except FileNotFoundError as exc:
            print(f"[resample] WARNING: hull reference missing ({exc}); "
                  f"will report relaxed energy without E_hull.")
            hull_ref = None

        for start in range(0, len(relax_idx), chunk_size):
            chunk = relax_idx[start: start + chunk_size]
            atoms_to_relax = [AseAtomsAdaptor.get_atoms(raw[i]) for i in chunk]
            try:
                relaxed_atoms, _ = relax_structures_mattersim(
                    atoms_to_relax, device="cuda" if torch.cuda.is_available() else "cpu",
                    potential_path=args.mattersim_potential_path,
                    fmax=0.05, max_n_steps=300,
                )
            except Exception as exc:
                print(f"[resample]   chunk {start}-{start+len(chunk)}: relax FAILED "
                      f"({type(exc).__name__}: {exc}); skipping")
                continue
            # MatterSim's BatchRelaxer.relax returns a dict; relax_structures_mattersim
            # iterates .values() so output ordering matches input ordering when all
            # succeed. If lengths mismatch, we can't safely align — skip the chunk.
            if len(relaxed_atoms) != len(chunk):
                print(f"[resample]   chunk {start}-{start+len(chunk)}: only "
                      f"{len(relaxed_atoms)}/{len(chunk)} relaxed; ordering ambiguous, "
                      f"skipping the rest")
                continue
            for sub_i, idx in enumerate(chunk):
                ra = relaxed_atoms[sub_i]
                e_per = total_energy_per_atom(ra)
                e_per_atom[idx] = e_per if np.isfinite(e_per) else None
                # Mean per-atom displacement during relax (proxy for "did the
                # lattice need to fix itself?").
                pre_pos = atoms_to_relax[sub_i].get_positions()
                post_pos = ra.get_positions()
                if pre_pos.shape == post_pos.shape:
                    disp = float(np.linalg.norm(pre_pos - post_pos, axis=1).mean())
                    relax_displacement[idx] = disp
                # E_hull
                if hull_ref is not None and np.isfinite(e_per):
                    relaxed_struct = AseAtomsAdaptor.get_structure(ra)
                    e_total = ra.info.get("total_energy", float("nan"))
                    e_hull = e_above_hull_per_atom(relaxed_struct, e_total, hull_ref)
                    e_hull_arr[idx] = float(e_hull) if np.isfinite(e_hull) else None
            print(f"[resample]   chunk {start}-{start+len(chunk)}: relaxed "
                  f"{len(relaxed_atoms)} structures.")

        # Stamp the relaxation fields into detail
        for d in detail:
            i = d["idx"]
            d["e_per_atom_relaxed"] = e_per_atom[i]
            d["e_hull_per_atom"] = e_hull_arr[i]
            d["mean_relax_displacement_A"] = relax_displacement[i]
            if e_hull_arr[i] is None:
                d["stable"] = None
                d["metastable"] = None
            else:
                d["stable"] = bool(e_hull_arr[i] <= 0.0)
                d["metastable"] = bool(e_hull_arr[i] <= 0.1)

        # Population-level stability summary
        n_metastable = sum(1 for d in detail if d.get("metastable"))
        n_stable = sum(1 for d in detail if d.get("stable"))
        n_with_ehull = sum(1 for d in detail if d.get("e_hull_per_atom") is not None)
        print()
        print(f"[resample] population stability ({n_with_ehull}/{len(detail)} with E_hull):")
        print(f"  metastable (E_hull ≤ 0.1 eV/atom): {n_metastable}/{len(detail)} "
              f"= {n_metastable/len(detail):.3f}")
        print(f"  stable     (E_hull ≤ 0.0):         {n_stable}/{len(detail)} "
              f"= {n_stable/len(detail):.3f}")

        # Composition-matched-specific verdict (the headline question)
        comp_matched = [d for d in detail if d["composition_match"]]
        if comp_matched:
            print()
            print(f"[resample] === STABILITY VERDICT for {len(comp_matched)} composition-matched "
                  f"generations ===")
            print(f"  {'idx':>4s}  {'formula':<24s} {'E_hull':>8s}  {'disp':>6s}  verdict")
            for d in comp_matched:
                eh = d["e_hull_per_atom"]
                disp = d["mean_relax_displacement_A"]
                eh_str = f"{eh:+.4f}" if eh is not None else "   —    "
                disp_str = f"{disp:.3f}" if disp is not None else "  —  "
                if eh is None:
                    verdict = "(?) relax/hull failed"
                elif eh <= 0.1 and (disp is None or disp < 0.5):
                    verdict = "(A) valid alt polymorph"
                elif eh > 0.1 or (disp is not None and disp >= 0.5):
                    verdict = "(B) unphysical / off-hull"
                else:
                    verdict = "(?) ambiguous"
                print(f"  {d['idx']:>4d}  {d['formula']:<24s} {eh_str:>8s}  "
                      f"{disp_str:>6s}  {verdict}")
        else:
            print(f"\n[resample] no composition-matched generations to verdict.")

    # Print first ~20 records (whichever is smaller).
    print()
    print("[resample] first generations:")
    cols = "    {:>4s}  {:<24s} {:>3s} {:>7s}  comp  lattice  rmse"
    if args.relax_and_score:
        cols += "    E_hull   disp"
    print(cols.format("idx", "formula", "n", "density"))
    for d in detail[: min(20, len(detail))]:
        rmse_str = f"{d['rmse']:.4f}" if d['rmse'] is not None else "  —  "
        comp_str = "✓" if d['composition_match'] else "."
        latt_str = "✓" if d['lattice_match'] else "."
        line = (f"    {d['idx']:>4d}  {d['formula']:<24s} {d['n_atoms']:>3d} "
                f"{d['density']:>7.3f}    {comp_str}      {latt_str}    {rmse_str}")
        if args.relax_and_score:
            eh = d.get("e_hull_per_atom")
            disp = d.get("mean_relax_displacement_A")
            eh_str = f"{eh:+.3f}" if eh is not None else "  —  "
            disp_str = f"{disp:.3f}" if disp is not None else "  —  "
            line += f"   {eh_str:>7s} {disp_str:>6s}"
        print(line)

    # Save full detail as JSON
    import json
    summary = {
        "row_id": args.row_id,
        "target_formula": target_formula,
        "target_sg": target_row["sg_symbol"],
        "N": len(raw),
        "prompt_template": args.prompt_template,
        "guidance_factor": args.guidance_factor,
        "composition_match_rate": n_comp_match / len(raw),
        "lattice_match_rate": n_lattice_match / len(raw),
        "lattice_match_given_composition": (
            n_lattice_match_given_comp / n_comp_match_total
            if n_comp_match_total else None
        ),
        "details": detail,
    }
    if args.relax_and_score:
        n_metastable = sum(1 for d in detail if d.get("metastable"))
        n_stable = sum(1 for d in detail if d.get("stable"))
        n_with_ehull = sum(1 for d in detail if d.get("e_hull_per_atom") is not None)
        summary["stability"] = {
            "n_with_e_hull": n_with_ehull,
            "metastable_pct": n_metastable / len(raw),
            "stable_pct": n_stable / len(raw),
        }
    with open(out_dir / "resample_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[resample] full details in {out_dir}/resample_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
