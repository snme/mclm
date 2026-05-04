"""Diagnose Stage 3a conditioning by comparing AtomsMapper outputs across diverse prompts.

For a given Stage 3a step dir + a directory of {row}.narrative.txt files, this:
  1. Loads ALM (with the trained LoRA) and AtomsMapper from the step dir
  2. Runs ALM forward on each narrative, extracting (K, hidden_dim) at [atoms_i] positions
  3. Runs AtomsMapper to get the 512-d conditioning vector per prompt
  4. Prints mean / std / pairwise cosine similarity of:
       - the raw alm_embedding (K * hidden_dim flat) — what Qwen3 produces
       - the projected conditioning (out_dim=512) — what cond_adapt/mixin sees

If different narratives produce nearly-identical conditioning vectors (cosine ≈ 1.0
across pairs), the model has learned a degenerate "prototype direction" that ignores
the input — i.e., AtomsMapper is collapsing to a constant. That points to LoRA
never specializing the [atoms_i] hidden states across prompts.

If conditioning vectors differ across prompts (cosine < ~0.95) but generated
structures still look similar, the issue is downstream (cond_adapt/mixin not yet
strong enough to translate AtomsMapper differences into score differences).

Usage:
  PYTHONPATH=/home/sathyae/mclm/alm:/home/sathyae/mclm/external/mattergen:$PYTHONPATH \\
  python helper_scripts/diagnose_conditioning.py \\
      --alm_checkpoint /home/sathyae/orcd/pool/alm_checkpoints/stage2_checkpoints/step=12000 \\
      --atoms_mapper   /home/sathyae/orcd/pool/stage3a/ckpts/.../step=5000/atoms_mapper.pt \\
      --eval_dir       /home/sathyae/orcd/pool/stage3a/eval_prompts
"""
import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "alm"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "alm" / "eval"))
from atoms_mapper import AtomsMapper  # noqa: E402
from loader import load_alm  # noqa: E402


SYSTEM_PROMPT = "You are an expert materials scientist."
# Match the canonical inference template (USER_TEMPLATES[0] in build_stage3a_pairs.py)
USER_TEMPLATE = "Generate a crystal structure described as: {narrative}"
ASSISTANT_ANCHOR = "Structure: "


def build_chat_ids(tokenizer, narrative: str, output_atoms_str: str, device, max_len: int = 2048):
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(narrative=narrative)},
        {"role": "assistant", "content": ASSISTANT_ANCHOR + output_atoms_str},
    ]
    full = tokenizer.apply_chat_template(
        msgs, tokenize=True, add_generation_prompt=False,
        enable_thinking=False, truncation=True, max_length=max_len,
    )
    return torch.tensor(full, dtype=torch.long, device=device)


def cos_matrix(X: torch.Tensor) -> torch.Tensor:
    """X: (N, D) → (N, N) cosine similarity matrix."""
    Xn = F.normalize(X, dim=-1)
    return Xn @ Xn.T


def fmt_matrix(M: torch.Tensor, labels, fmt: str = "+.6f") -> str:
    n = M.shape[0]
    rows = ["       " + "  ".join(f"{l[:8]:>10}" for l in labels)]
    for i in range(n):
        rows.append(
            f"{labels[i][:6]:>6} " + "  ".join(f"{M[i, j].item():{fmt}}" for j in range(n))
        )
    return "\n".join(rows)


def l2_dist_matrix(X: torch.Tensor) -> torch.Tensor:
    """X: (N, D) → (N, N) pairwise L2 distance."""
    return torch.cdist(X, X, p=2)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load ALM (trained LoRA + projector from step dir) ─────────────────
    print(f"[diag] loading ALM from {args.alm_checkpoint} ...", flush=True)
    alm, tokenizer = load_alm(
        checkpoint=args.alm_checkpoint,
        merge_lora=True,
        use_cached_embeddings=True,
        device=device,
    )
    alm.eval()
    K = len(alm.output_atom_token_ids)
    hidden_dim = alm.llm_hidden_dim

    # ── Load AtomsMapper if a path was provided ──────────────────────────
    am_path = Path(args.atoms_mapper) if args.atoms_mapper else None
    if am_path is not None and am_path.exists():
        print(f"[diag] loading AtomsMapper from {am_path} ...", flush=True)
        ckpt = torch.load(am_path, map_location=device)
        mapper = AtomsMapper(hidden_dim=hidden_dim, mid_dim=2048, out_dim=512, K=K).to(device)
        mapper.load_state_dict(ckpt["atoms_mapper_state_dict"])
        mapper.eval()
        print(f"[diag] AtomsMapper trained: {sum(p.numel() for p in mapper.parameters())/1e6:.2f}M params")
        ts = ckpt.get("trainable_state_dict", {})
        am_norm = sum(p.detach().float().pow(2).sum().item() for p in mapper.parameters()) ** 0.5
        ca_norm = sum(v.float().pow(2).sum().item() for k, v in ts.items() if "cond_adapt_layers" in k) ** 0.5
        cm_norm = sum(v.float().pow(2).sum().item() for k, v in ts.items() if "cond_mixin_layers" in k) ** 0.5
        n_ca = sum(1 for k in ts if "cond_adapt_layers" in k)
        n_cm = sum(1 for k in ts if "cond_mixin_layers" in k)
        print(f"[diag] weight L2 norms at this checkpoint:")
        print(f"          AtomsMapper:  {am_norm:.4f}")
        print(f"          cond_adapt:   {ca_norm:.4f}  ({n_ca} tensors)")
        print(f"          cond_mixin:   {cm_norm:.4f}  ({n_cm} tensors, zero-init by MatterGen)")
    else:
        # Stage 2 ckpt or any LoRA-only dir: synthesize a fresh-init AtomsMapper
        # so we can still measure Qwen3's hidden-state diversity. The cond stats
        # reported below are not meaningful in this case; ignore them.
        if am_path is None:
            print(f"[diag] --atoms_mapper not provided — using fresh-init AtomsMapper.")
        else:
            print(f"[diag] no atoms_mapper.pt at {am_path} — using fresh-init AtomsMapper.")
        print(f"[diag] (only the raw alm_embedding cosine matters here; cond is from random init.)")
        mapper = AtomsMapper(hidden_dim=hidden_dim, mid_dim=2048, out_dim=512, K=K).to(device)
        mapper.eval()

    # ── Read narratives ──────────────────────────────────────────────────
    eval_dir = Path(args.eval_dir)
    narrative_paths = sorted(eval_dir.glob("*.narrative.txt"))
    if not narrative_paths:
        raise FileNotFoundError(f"no *.narrative.txt files in {eval_dir}")
    print(f"[diag] found {len(narrative_paths)} narratives in {eval_dir}")
    narratives = [(p.stem.replace(".narrative", ""), p.read_text()) for p in narrative_paths]

    # ── Forward each narrative; collect alm_embedding (32768) and cond (512) ──
    output_atoms_str = "".join(alm.output_atom_tokens)
    alm_embs = []   # (N, K * hidden_dim)
    conds = []      # (N, out_dim)

    print(f"\n[diag] running ALM forward on each narrative (no_grad) ...")
    with torch.no_grad():
        for label, narrative in narratives:
            ids = build_chat_ids(tokenizer, narrative, output_atoms_str, device,
                                 max_len=args.max_len)
            attn = torch.ones_like(ids)
            empty = [torch.zeros(0, 256, dtype=torch.float32, device=device)]
            hidden = alm.extract_atoms_hidden_states([ids], [attn], atom_embeds=empty)
            # hidden: (1, K, hidden_dim)
            alm_emb = hidden.flatten(1).float()  # (1, K*hidden_dim)
            alm_embs.append(alm_emb)
            cond = mapper(alm_emb)  # (1, out_dim)
            conds.append(cond)
            print(f"    {label:30s}  "
                  f"alm_emb: shape={tuple(alm_emb.shape)}, mean={alm_emb.mean().item():+.4f}, "
                  f"std={alm_emb.std().item():.4f}    "
                  f"cond mean={cond.mean().item():+.4f} std={cond.std().item():.4f}")

    A = torch.cat(alm_embs, dim=0)  # (N, K*H)
    C = torch.cat(conds, dim=0)     # (N, out_dim)
    labels = [n[0] for n in narratives]

    # ── Pairwise cosine + L2 distance ────────────────────────────────────
    print(f"\n[diag] pairwise cosine similarity of raw alm_embedding (Qwen3 hidden states):")
    print(fmt_matrix(cos_matrix(A), labels, "+.6f"))
    print(f"\n[diag] pairwise L2 distance of raw alm_embedding (Qwen3 hidden states):")
    print(fmt_matrix(l2_dist_matrix(A), labels, ".4f"))
    print(f"\n[diag] pairwise cosine similarity of conditioning vector (AtomsMapper output):")
    print(fmt_matrix(cos_matrix(C), labels, "+.6f"))

    # ── Aggregate stats ──────────────────────────────────────────────────
    n = A.size(0)
    if n > 1:
        # off-diagonal mean/min cosine
        Csim = cos_matrix(C)
        Asim = cos_matrix(A)
        mask = ~torch.eye(n, dtype=torch.bool, device=Csim.device)
        print(f"\n[diag] off-diagonal cosine summary (N={n} prompts, N(N-1)={n*(n-1)} pairs):")
        print(f"    raw alm_embedding:   mean={Asim[mask].mean().item():.4f}  "
              f"min={Asim[mask].min().item():.4f}  max={Asim[mask].max().item():.4f}")
        print(f"    AtomsMapper output:  mean={Csim[mask].mean().item():.4f}  "
              f"min={Csim[mask].min().item():.4f}  max={Csim[mask].max().item():.4f}")

    # Verdict heuristic
    print("\n[diag] interpretation:")
    if n > 1:
        c_mean = cos_matrix(C)[mask].mean().item()
        a_mean = cos_matrix(A)[mask].mean().item()
        if c_mean > 0.99:
            print("    ❌ AtomsMapper outputs are nearly identical across all prompts (cos > 0.99).")
            print("       The mapper has collapsed to a near-constant; conditioning carries no")
            print("       prompt-specific information. Likely root cause: LoRA hasn't differentiated")
            print("       [atoms_i] hidden states across prompts, OR AtomsMapper init/training has")
            print("       saturated to a fixed output regardless of input.")
        elif c_mean > 0.95:
            print("    ⚠ AtomsMapper outputs are very similar (cos > 0.95) but not identical.")
            print("      Some prompt differentiation but signal is weak. Need more training, or")
            print("      higher lora_lr to give Qwen3 more freedom to specialize.")
        elif c_mean > 0.6:
            print("    ✓ AtomsMapper outputs DIFFER meaningfully across prompts (cos < 0.95).")
            print("      The mapper is producing prompt-specific signals. If generations still")
            print("      look the same, the issue is downstream (cond_adapt/mixin not yet strong).")
        else:
            print(f"    AtomsMapper outputs differ a lot (cos={c_mean:.3f}). If this seems too")
            print(f"    different, the training may be amplifying noise rather than signal.")

        if a_mean > 0.99:
            print("    Note: raw alm_embeddings (Qwen3 hidden states) are also nearly identical")
            print("    across prompts. The bottleneck is upstream of AtomsMapper — the LLM itself")
            print("    isn't producing prompt-specific [atoms_i] hidden states.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--alm_checkpoint", required=True,
                   help="Stage 2 ckpt dir (lora_adapter/ + projector_and_state.pt). Stage 3a "
                        "doesn't train the LLM, so this is just the Stage 2 ckpt used at training.")
    p.add_argument("--atoms_mapper", default=None,
                   help="Path to atoms_mapper.pt. If omitted, a fresh-init AtomsMapper is used "
                        "(useful for control runs that just measure Qwen3 hidden-state diversity).")
    p.add_argument("--eval_dir", required=True,
                   help="Directory of *.narrative.txt prompts")
    p.add_argument("--max_len", type=int, default=2048,
                   help="Truncate prompts to this many tokens (matches training max)")
    main(p.parse_args())
