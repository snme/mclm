"""Phase 0 diagnostic: does the trained AtomsMapper output linearly encode composition?

Three complementary tests, all running on a held-out slice of pairs.parquet:

1. **Presence probe** — train a frozen Linear(512 → 100) on (am_out, multi-hot
   composition). Held-out macro-recall tells us whether *which elements are
   present* is linearly decodable. If recall is high (>0.7), AtomsMapper has
   the presence info and the bottleneck is downstream (cond_adapt/mixin); if
   low (<0.4), the info isn't there yet and a supervised aux loss is needed.

2. **Count probe** — train a frozen Linear(512 → 100*(MAX_COUNT+1)) decoding
   exact per-element atom counts via per-slot cross-entropy. Reports MAE on
   present elements, per-element exact-count accuracy, whole-composition exact
   match, and reduced-formula match (Cu2Ni2 ≡ CuNi). This tells us whether
   stoichiometry is in `am_out` — the missing signal CSP needs after element
   masking handles presence.

3. **PCA + UMAP visualizations** — render AtomsMapper outputs in 2D, colored by
   various composition labels. Visual confirmation of the probe numbers; clean
   clusters by element indicate good encoding, noise indicates collapse.

This script does NOT modify AtomsMapper weights — it only probes the existing checkpoint.

Usage (single GPU, ~5–15 min):
  PYTHONPATH=/home/sathyae/mclm/alm:/home/sathyae/mclm/external/mattergen:$PYTHONPATH \\
  python helper_scripts/probe_atoms_mapper_clusters.py \\
      --alm_checkpoint /home/sathyae/orcd/pool/alm_checkpoints/stage2_checkpoints/step=12000 \\
      --atoms_mapper   /home/sathyae/orcd/pool/stage3a/ckpts/run5_contrastive/step=5000/atoms_mapper.pt \\
      --pairs_parquet  /home/sathyae/orcd/pool/stage3a/pairs.parquet \\
      --out_dir        /home/sathyae/orcd/pool/stage3a/diagnostics/run5_step5000 \\
      --n_samples 800
"""
import argparse
import json
import sys
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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "alm"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "alm" / "eval"))
from atoms_mapper import AtomsMapper  # noqa: E402
from loader import load_alm  # noqa: E402


# Match the canonical training-time format (USER_TEMPLATES[0] in build_stage3a_pairs.py)
SYSTEM_PROMPT = "You are an expert materials scientist."
USER_TEMPLATE = "Generate a crystal structure described as: {narrative}"
ASSISTANT_ANCHOR = "Structure: "

# ASE-style element symbol → Z. Composition is a multi-hot over Z = 1..100.
N_ELEMENTS = 100
# Max per-element atom count the count-probe can predict. pairs.parquet is
# filtered to ≤20 atoms total per cell, so 20 is a safe upper bound for any
# single element. Counts above this are clamped (and logged).
MAX_COUNT = 20


def _symbol_to_z():
    """Map element symbol → atomic number using ASE's atomic_numbers dict."""
    from ase.data import atomic_numbers
    return {sym: z for sym, z in atomic_numbers.items() if 1 <= z <= N_ELEMENTS}


def composition_multihot(elements: list[str], sym2z: dict) -> np.ndarray:
    """Multi-hot vector over Z=1..N_ELEMENTS (1-indexed in ASE; we use 0-indexed slot Z-1)."""
    v = np.zeros(N_ELEMENTS, dtype=np.float32)
    for s in elements:
        z = sym2z.get(s.strip())
        if z is not None and 1 <= z <= N_ELEMENTS:
            v[z - 1] = 1.0
    return v


def composition_count_vec(elements: list[str], sym2z: dict) -> np.ndarray:
    """Integer count per Z=1..N_ELEMENTS, clamped to [0, MAX_COUNT]. Shape (N_ELEMENTS,)."""
    v = np.zeros(N_ELEMENTS, dtype=np.int64)
    for s in elements:
        z = sym2z.get(s.strip())
        if z is not None and 1 <= z <= N_ELEMENTS:
            v[z - 1] += 1
    np.clip(v, 0, MAX_COUNT, out=v)
    return v


def _reduce_counts(c: np.ndarray) -> tuple:
    """Normalize an integer count vector to its reduced-formula tuple by dividing
    nonzero entries by their GCD. Used for reduced-formula match (Cu2Ni2 == CuNi)."""
    nz = c[c > 0]
    if nz.size == 0:
        return tuple(c.tolist())
    g = int(np.gcd.reduce(nz))
    if g <= 1:
        return tuple(c.tolist())
    return tuple((c // g).tolist())


def sample_held_out_rows(parquet_path, n_samples, seed):
    """Sample n_samples random rows uniformly. Returns list of dicts with row_id,
    user_prompt, narrative, atoms_struct."""
    pf = pq.ParquetFile(parquet_path)
    total = pf.metadata.num_rows
    rng = np.random.default_rng(seed)
    target_idxs = sorted(rng.choice(total, size=min(n_samples, total), replace=False))
    target_set = set(int(i) for i in target_idxs)

    collected = []
    cursor = 0
    cols = ["row_id", "narrative", "user_prompt", "atoms_struct"]
    for batch in pf.iter_batches(batch_size=8192, columns=cols):
        rb = batch.to_pydict()
        n = len(rb["row_id"])
        for i in range(n):
            if cursor in target_set:
                collected.append({
                    "row_id":      rb["row_id"][i],
                    "narrative":   rb["narrative"][i],
                    "user_prompt": rb["user_prompt"][i],
                    "atoms_struct": rb["atoms_struct"][i],
                })
            cursor += 1
            if len(collected) == len(target_set):
                break
        if len(collected) == len(target_set):
            break
    return collected


def build_chat_ids(tokenizer, user_prompt: str, output_atoms_str: str, device, max_len: int):
    msgs = [
        {"role": "system",    "content": SYSTEM_PROMPT},
        {"role": "user",      "content": user_prompt},
        {"role": "assistant", "content": ASSISTANT_ANCHOR + output_atoms_str},
    ]
    full = tokenizer.apply_chat_template(
        msgs, tokenize=True, add_generation_prompt=False,
        enable_thinking=False, truncation=True, max_length=max_len,
    )
    return torch.tensor(full, dtype=torch.long, device=device)


def linear_probe_train(am_out: np.ndarray, comp: np.ndarray, n_epochs: int = 200, lr: float = 1e-2,
                       train_frac: float = 0.8, seed: int = 0):
    """Train a frozen Linear(D, N_ELEMENTS) probe on am_out → composition. Reports
    macro-precision / macro-recall on held-out split. AtomsMapper is NOT updated.
    """
    rng = np.random.default_rng(seed)
    n = am_out.shape[0]
    perm = rng.permutation(n)
    n_train = int(train_frac * n)
    train_idx, val_idx = perm[:n_train], perm[n_train:]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X = torch.from_numpy(am_out).to(device)
    y = torch.from_numpy(comp).to(device)
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    probe = nn.Linear(am_out.shape[1], N_ELEMENTS).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=1e-4)

    history = []
    for epoch in range(n_epochs):
        probe.train()
        logits = probe(X_train)
        loss = F.binary_cross_entropy_with_logits(logits, y_train)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if epoch % 20 == 0 or epoch == n_epochs - 1:
            probe.eval()
            with torch.no_grad():
                val_logits = probe(X_val)
                val_pred = (torch.sigmoid(val_logits) > 0.5).float()
                # Per-class precision/recall, macro-averaged (skip classes with no positives)
                tp = (val_pred * y_val).sum(dim=0)
                fp = (val_pred * (1 - y_val)).sum(dim=0)
                fn = ((1 - val_pred) * y_val).sum(dim=0)
                pos_classes = (y_val.sum(dim=0) > 0)
                prec = (tp / (tp + fp + 1e-9))[pos_classes].mean().item()
                rec = (tp / (tp + fn + 1e-9))[pos_classes].mean().item()
                history.append({"epoch": epoch, "loss": loss.item(),
                                "val_precision": prec, "val_recall": rec,
                                "n_pos_classes": int(pos_classes.sum().item())})
    return history, probe


def linear_probe_count_train(am_out: np.ndarray, counts: np.ndarray, n_epochs: int = 200,
                             lr: float = 1e-2, train_frac: float = 0.8, seed: int = 0):
    """Train a frozen Linear(D, N_ELEMENTS*(MAX_COUNT+1)) probe on am_out → exact
    per-element counts. Per-element cross-entropy over (MAX_COUNT+1) bins. AtomsMapper
    is NOT updated.

    Reports five metrics on the held-out split:
      - count_present_mae        : MAE on counts (target > 0); the metric most
                                   directly tied to "do you know the stoichiometry"
      - count_present_exact_pct  : per-(sample, element-with-count>0) exact match %
      - whole_composition_exact  : sample-level exact-match across all 100 slots
      - reduced_formula_exact    : sample-level match after gcd-normalization
                                   (Cu2Ni2 ≡ CuNi)
      - count_zero_pct           : per-(sample, element-with-count==0) accuracy
                                   on the negative class (sanity baseline)
    """
    rng = np.random.default_rng(seed)
    n = am_out.shape[0]
    perm = rng.permutation(n)
    n_train = int(train_frac * n)
    train_idx, val_idx = perm[:n_train], perm[n_train:]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_bins = MAX_COUNT + 1
    X = torch.from_numpy(am_out).to(device)
    y = torch.from_numpy(counts).to(device)              # (N, 100) int64
    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    probe = nn.Linear(am_out.shape[1], N_ELEMENTS * n_bins).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=1e-4)

    history = []
    for epoch in range(n_epochs):
        probe.train()
        logits = probe(X_train).reshape(-1, N_ELEMENTS, n_bins)
        loss = F.cross_entropy(logits.reshape(-1, n_bins),
                               y_train.reshape(-1).long())
        opt.zero_grad()
        loss.backward()
        opt.step()
        if epoch % 20 == 0 or epoch == n_epochs - 1:
            probe.eval()
            with torch.no_grad():
                val_logits = probe(X_val).reshape(-1, N_ELEMENTS, n_bins)
                val_pred = val_logits.argmax(dim=-1)        # (N_val, 100), int64
                pos_mask = (y_val > 0)
                neg_mask = (y_val == 0)
                # MAE / exact on the positive entries (the actually-present elements)
                if pos_mask.any():
                    diffs = (val_pred[pos_mask] - y_val[pos_mask]).abs().float()
                    count_present_mae = diffs.mean().item()
                    count_present_exact_pct = (diffs == 0).float().mean().item()
                else:
                    count_present_mae = float("nan")
                    count_present_exact_pct = float("nan")
                # Negative-class accuracy (probe correctly predicts "no atoms of Z")
                if neg_mask.any():
                    count_zero_pct = (val_pred[neg_mask] == 0).float().mean().item()
                else:
                    count_zero_pct = float("nan")
                # Sample-level whole-composition exact match: all 100 slots correct
                whole_exact = (val_pred == y_val).all(dim=-1).float().mean().item()
                # Reduced-formula match
                pred_np = val_pred.cpu().numpy()
                tgt_np = y_val.cpu().numpy()
                rf_hits = 0
                for k in range(pred_np.shape[0]):
                    if _reduce_counts(pred_np[k]) == _reduce_counts(tgt_np[k]):
                        rf_hits += 1
                reduced_formula_exact = rf_hits / max(pred_np.shape[0], 1)
                history.append({
                    "epoch": epoch, "loss": loss.item(),
                    "count_present_mae": count_present_mae,
                    "count_present_exact_pct": count_present_exact_pct,
                    "count_zero_pct": count_zero_pct,
                    "whole_composition_exact_pct": whole_exact,
                    "reduced_formula_exact_pct": reduced_formula_exact,
                    "n_val": int(val_pred.shape[0]),
                })
    return history, probe


def make_color_arrays(rows: list[dict], sym2z: dict):
    """Various coloring schemes for visualization."""
    most_common_el = []
    n_unique = []
    has_O = []
    has_tm = []  # transition metal: Z in [21..30, 39..48, 72..80]
    tm_zs = set(list(range(21, 31)) + list(range(39, 49)) + list(range(72, 81)))
    for r in rows:
        struct = r["atoms_struct"]
        if hasattr(struct, "as_py"):
            struct = struct.as_py()
        elements = [s.strip() for s in struct["elements"]]
        # most common element by count
        from collections import Counter
        c = Counter(elements)
        mc = c.most_common(1)[0][0] if c else "?"
        most_common_el.append(mc)
        n_unique.append(len(set(elements)))
        zs = {sym2z.get(e) for e in elements}
        has_O.append("O" in elements)
        has_tm.append(any(z in tm_zs for z in zs if z is not None))
    return {
        "most_common_element": most_common_el,
        "n_unique": np.array(n_unique),
        "has_O": np.array(has_O),
        "has_transition_metal": np.array(has_tm),
    }


def plot_2d(coords_2d: np.ndarray, colors: dict, title_prefix: str, out_dir: Path):
    """Generate one PNG per coloring scheme."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # most_common_element: categorical, color top-K most frequent, others gray
    el_list = colors["most_common_element"]
    from collections import Counter
    el_counter = Counter(el_list)
    top_K = 15
    top_set = {e for e, _ in el_counter.most_common(top_K)}
    cmap = plt.get_cmap("tab20")
    el_to_color = {e: cmap(i % 20) for i, e in enumerate(sorted(top_set))}
    point_colors = [el_to_color[e] if e in top_set else (0.7, 0.7, 0.7, 0.4) for e in el_list]
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(coords_2d[:, 0], coords_2d[:, 1], c=point_colors, s=14, edgecolors="none")
    handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=c,
                          markersize=8, label=e)
               for e, c in sorted(el_to_color.items())]
    handles.append(plt.Line2D([0], [0], marker="o", color="w",
                              markerfacecolor=(0.7, 0.7, 0.7, 0.4), markersize=8, label="other"))
    ax.legend(handles=handles, loc="best", fontsize=8, ncol=2)
    ax.set_title(f"{title_prefix} — colored by most-common element")
    ax.set_xlabel("axis 0"); ax.set_ylabel("axis 1")
    fig.tight_layout()
    fig.savefig(out_dir / f"{title_prefix}_by_most_common_element.png", dpi=130)
    plt.close(fig)

    # n_unique: numerical
    fig, ax = plt.subplots(figsize=(8, 8))
    sc = ax.scatter(coords_2d[:, 0], coords_2d[:, 1], c=colors["n_unique"], s=14,
                    cmap="viridis", edgecolors="none")
    plt.colorbar(sc, ax=ax, label="# unique elements")
    ax.set_title(f"{title_prefix} — colored by n_unique_elements")
    fig.tight_layout()
    fig.savefig(out_dir / f"{title_prefix}_by_n_unique.png", dpi=130)
    plt.close(fig)

    # has_O: binary
    for key, label in [("has_O", "contains oxygen"),
                      ("has_transition_metal", "contains transition metal")]:
        fig, ax = plt.subplots(figsize=(8, 8))
        c_arr = np.array(colors[key])
        ax.scatter(coords_2d[c_arr, 0], coords_2d[c_arr, 1], c="tab:red", s=14,
                   alpha=0.8, label=f"{label} (yes)", edgecolors="none")
        ax.scatter(coords_2d[~c_arr, 0], coords_2d[~c_arr, 1], c="lightgray", s=14,
                   alpha=0.6, label=f"{label} (no)", edgecolors="none")
        ax.legend(loc="best")
        ax.set_title(f"{title_prefix} — {label}")
        fig.tight_layout()
        fig.savefig(out_dir / f"{title_prefix}_by_{key}.png", dpi=130)
        plt.close(fig)


def main(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load ALM (frozen Stage 2) and trained AtomsMapper ──────────────────
    print(f"[probe] loading ALM from {args.alm_checkpoint} ...", flush=True)
    alm, tokenizer = load_alm(
        checkpoint=args.alm_checkpoint, merge_lora=True,
        use_cached_embeddings=True, device=device,
    )
    alm.eval()
    K = len(alm.output_atom_token_ids)
    hidden_dim = alm.llm_hidden_dim
    print(f"[probe] K={K}, hidden_dim={hidden_dim}")

    print(f"[probe] loading AtomsMapper from {args.atoms_mapper} ...", flush=True)
    ckpt = torch.load(args.atoms_mapper, map_location=device)
    mapper = AtomsMapper(hidden_dim=hidden_dim, mid_dim=args.mid_dim,
                         out_dim=args.out_dim, K=K).to(device).eval()
    mapper.load_state_dict(ckpt["atoms_mapper_state_dict"])

    # ── Sample rows + compute composition labels ───────────────────────────
    print(f"[probe] sampling {args.n_samples} rows from {args.pairs_parquet} ...")
    rows = sample_held_out_rows(args.pairs_parquet, args.n_samples, args.seed)
    print(f"[probe] got {len(rows)} rows")

    sym2z = _symbol_to_z()
    comp = np.zeros((len(rows), N_ELEMENTS), dtype=np.float32)
    counts = np.zeros((len(rows), N_ELEMENTS), dtype=np.int64)
    for i, r in enumerate(rows):
        struct = r["atoms_struct"]
        if hasattr(struct, "as_py"):
            struct = struct.as_py()
        comp[i] = composition_multihot(struct["elements"], sym2z)
        counts[i] = composition_count_vec(struct["elements"], sym2z)

    # ── Forward pass: get AtomsMapper outputs ──────────────────────────────
    print(f"[probe] running ALM + AtomsMapper forward on {len(rows)} prompts ...")
    output_atoms_str = "".join(alm.output_atom_tokens)
    am_outs = np.zeros((len(rows), args.out_dim), dtype=np.float32)
    with torch.no_grad():
        for i, r in enumerate(rows):
            ids = build_chat_ids(tokenizer, r["user_prompt"], output_atoms_str,
                                 device, max_len=args.max_len)
            attn = torch.ones_like(ids)
            empty = [torch.zeros(0, 256, dtype=torch.float32, device=device)]
            hidden = alm.extract_atoms_hidden_states([ids], [attn], atom_embeds=empty)
            # (1, K, hidden_dim) → flat → AtomsMapper → (1, out_dim)
            am = mapper(hidden.flatten(1).float())
            am_outs[i] = am.squeeze(0).cpu().numpy()
            if (i + 1) % 50 == 0 or i == len(rows) - 1:
                print(f"  {i+1}/{len(rows)}", flush=True)

    print(f"[probe] am_outs shape={am_outs.shape}, mean={am_outs.mean():+.4f}, "
          f"std={am_outs.std():.4f}")

    # ── Linear probe ──────────────────────────────────────────────────────
    print(f"\n[probe] training linear probe (am_out → composition) ...")
    history, probe_model = linear_probe_train(
        am_outs, comp, n_epochs=args.probe_epochs, lr=args.probe_lr,
        train_frac=0.8, seed=args.seed,
    )
    print(f"[probe] linear probe history (every 20 epochs):")
    for h in history:
        print(f"  epoch={h['epoch']:4d}  train_loss={h['loss']:.4f}  "
              f"val_precision={h['val_precision']:.3f}  "
              f"val_recall={h['val_recall']:.3f}  "
              f"({h['n_pos_classes']} positive classes)")
    final = history[-1]
    print(f"\n[probe] FINAL: val precision={final['val_precision']:.3f}  "
          f"recall={final['val_recall']:.3f}")

    # ── Count probe (exact stoichiometry) ─────────────────────────────────
    print(f"\n[probe] training count probe (am_out → exact per-element counts) ...")
    count_history, count_probe = linear_probe_count_train(
        am_outs, counts, n_epochs=args.probe_epochs, lr=args.probe_lr,
        train_frac=0.8, seed=args.seed,
    )
    print(f"[probe] count probe history (every 20 epochs):")
    for h in count_history:
        print(f"  epoch={h['epoch']:4d}  ce_loss={h['loss']:.4f}  "
              f"present_mae={h['count_present_mae']:.3f}  "
              f"present_exact={h['count_present_exact_pct']:.3f}  "
              f"whole_exact={h['whole_composition_exact_pct']:.3f}  "
              f"reduced_formula={h['reduced_formula_exact_pct']:.3f}")
    count_final = count_history[-1]
    print(f"\n[probe] COUNT FINAL: present_mae={count_final['count_present_mae']:.3f}  "
          f"present_exact={count_final['count_present_exact_pct']:.3f}  "
          f"whole_exact={count_final['whole_composition_exact_pct']:.3f}  "
          f"reduced_formula_exact={count_final['reduced_formula_exact_pct']:.3f}")

    # ── Verdict ────────────────────────────────────────────────────────────
    rec = final["val_recall"]
    if rec > 0.70:
        verdict = ("(b) Info IS encoded; bottleneck is downstream (cond_adapt/mixin).\n"
                   "    Aux loss likely marginal — investigate cond_adapt/mixin capacity\n"
                   "    or per-atom conditioning architecture instead.")
    elif rec >= 0.40:
        verdict = ("Mixed — info partially encoded. Aux loss should help; can also\n"
                   "    push downstream architectural improvements.")
    else:
        verdict = ("(a) Info is NOT yet encoded. Aux loss is necessary.\n"
                   "    Proceed with the composition-aux training plan.")
    print(f"\n[probe] VERDICT (recall={rec:.3f}):\n  {verdict}")

    # ── Plotting: PCA-2D + UMAP-2D from PCA-50D ────────────────────────────
    print(f"\n[probe] computing PCA + UMAP projections ...")
    colors = make_color_arrays(rows, sym2z)

    # Standardize before PCA for numerical stability
    am_centered = am_outs - am_outs.mean(axis=0, keepdims=True)
    # PCA via SVD (no sklearn dependency)
    U, S, Vt = np.linalg.svd(am_centered, full_matrices=False)
    pca50 = (U[:, :50] * S[:50])
    pca2 = (U[:, :2] * S[:2])
    var_explained = (S ** 2) / (S ** 2).sum()
    print(f"[probe] PCA: top-2 var explained = "
          f"{var_explained[0]:.3f} + {var_explained[1]:.3f} = "
          f"{var_explained[:2].sum():.3f}; top-50 sum = {var_explained[:50].sum():.3f}")

    plot_2d(pca2, colors, "pca2", out_dir)

    # UMAP from PCA-50
    print(f"[probe] running UMAP ...")
    reducer = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.1, random_state=args.seed)
    umap2 = reducer.fit_transform(pca50)
    plot_2d(umap2, colors, "umap", out_dir)

    # ── Persist arrays + metrics ───────────────────────────────────────────
    np.savez(out_dir / "probe_arrays.npz",
             am_outs=am_outs, composition=comp, counts=counts,
             pca2=pca2, pca50=pca50)
    with open(out_dir / "probe_metrics.json", "w") as f:
        json.dump({
            "alm_checkpoint": str(args.alm_checkpoint),
            "atoms_mapper":   str(args.atoms_mapper),
            "n_samples":      len(rows),
            "linear_probe_history": history,
            "verdict":        verdict.split("\n")[0],
            "final_val_precision": final["val_precision"],
            "final_val_recall":    final["val_recall"],
            "count_probe_history":         count_history,
            "count_probe_max_count":       MAX_COUNT,
            "final_count_present_mae":         count_final["count_present_mae"],
            "final_count_present_exact_pct":   count_final["count_present_exact_pct"],
            "final_whole_composition_exact_pct": count_final["whole_composition_exact_pct"],
            "final_reduced_formula_exact_pct":   count_final["reduced_formula_exact_pct"],
        }, f, indent=2)
    print(f"\n[probe] all artifacts in {out_dir}")
    print("        - {pca2,umap}_by_*.png plots")
    print("        - probe_arrays.npz (am_outs, composition, pca embeddings)")
    print("        - probe_metrics.json (linear probe history + verdict)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--alm_checkpoint", required=True,
                   help="Stage 2 ckpt dir (lora_adapter/ + projector_and_state.pt)")
    p.add_argument("--atoms_mapper", required=True,
                   help="Path to trained atoms_mapper.pt")
    p.add_argument("--pairs_parquet", required=True,
                   help="pairs.parquet built by build_stage3a_pairs.py")
    p.add_argument("--out_dir", required=True,
                   help="Output dir for plots, npz, metrics")
    p.add_argument("--n_samples", type=int, default=800,
                   help="Number of held-out narratives to sample")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_len", type=int, default=2048,
                   help="Tokenizer truncation length (matches training default)")
    p.add_argument("--mid_dim", type=int, default=2048,
                   help="AtomsMapper mid_dim — must match training")
    p.add_argument("--out_dim", type=int, default=512,
                   help="AtomsMapper out_dim — must match training (=MatterGen hidden_dim)")
    p.add_argument("--probe_epochs", type=int, default=300)
    p.add_argument("--probe_lr", type=float, default=1e-2)
    main(p.parse_args())
