"""Smoke-test a Stage 3a checkpoint: load → ALM forward → AtomsMapper → conditioning vector.

Verifies the full Stage 3a save layout round-trips:
  step=N/
    lora_adapter/         ← LoRA continued from Stage 2 (PEFT save_pretrained)
    projector_and_state.pt ← projector state (mirrors Stage 2 layout)
    atoms_mapper.pt        ← AtomsMapper + cond_adapt/mixin + (optional) optimizer

Usage:
  PYTHONPATH=/home/sathyae/mclm/alm:/home/sathyae/mclm/external/mattergen:$PYTHONPATH \\
  python helper_scripts/verify_stage3a_ckpt.py --stage3a_dir /tmp/stage3a_smoke/step=50
"""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "alm"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "alm" / "eval"))
from atoms_mapper import AtomsMapper  # noqa: E402
from loader import load_alm  # noqa: E402


SAMPLE_PROMPTS = [
    "Generate a stable cubic perovskite with bandgap > 2 eV.",
    "Generate the structure of a binary intermetallic with high bulk modulus.",
]


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[verify] loading ALM from {args.stage3a_dir} (LoRA continued from Stage 2) ...",
          flush=True)
    alm, tokenizer = load_alm(
        checkpoint=args.stage3a_dir,
        merge_lora=True,             # merge for inference speed
        use_cached_embeddings=True,
        device=device,
    )
    alm.eval()

    K = len(alm.output_atom_token_ids)
    in_dim = K * alm.llm_hidden_dim
    print(f"[verify] K={K}, in_dim={in_dim}, ALM hidden={alm.llm_hidden_dim}")

    print(f"[verify] loading AtomsMapper from {args.stage3a_dir}/atoms_mapper.pt ...")
    ckpt = torch.load(Path(args.stage3a_dir) / "atoms_mapper.pt", map_location=device)
    mapper = AtomsMapper(in_dim=in_dim, hidden=4096, out_dim=512).to(device)
    mapper.load_state_dict(ckpt["atoms_mapper_state_dict"])
    mapper.eval()
    n_mapper = sum(p.numel() for p in mapper.parameters())
    print(f"[verify] AtomsMapper loaded: {n_mapper/1e6:.1f}M params")

    cond_keys = list(ckpt.get("trainable_state_dict", {}).keys())
    n_cond = len(cond_keys)
    if n_cond:
        print(f"[verify] cond_adapt/mixin state present: {n_cond} tensors "
              f"(first: {cond_keys[0]})")
    else:
        print("[verify] WARNING: no trainable_state_dict in atoms_mapper.pt")

    # Build prompts → ALM forward → hidden states → AtomsMapper → conditioning
    print(f"\n[verify] running {len(SAMPLE_PROMPTS)} test prompts ...")
    output_atoms_str = "".join(alm.output_atom_tokens)
    for i, prompt in enumerate(SAMPLE_PROMPTS):
        msgs = [
            {"role": "system",
             "content": "You are an expert materials scientist."},
            {"role": "user",
             "content": f"Generate a crystal structure described as: {prompt}"},
            {"role": "assistant", "content": prompt + output_atoms_str},
        ]
        full_ids = tokenizer.apply_chat_template(
            msgs, tokenize=True, add_generation_prompt=False,
            enable_thinking=False, truncation=True, max_length=1024,
        )
        ids = torch.tensor(full_ids, dtype=torch.long, device=device)
        attn = torch.ones_like(ids)

        # Verify all K [atoms_i] tokens made it past truncation
        atoms_id_set = set(alm.output_atom_token_ids)
        n_present = sum(int(t.item()) in atoms_id_set for t in ids)
        if n_present != K:
            print(f"  prompt {i}: only {n_present}/{K} [atoms_i] tokens present — skipping")
            continue

        empty_embeds = [torch.zeros(0, 256, dtype=torch.float32, device=device)]
        with torch.no_grad():
            hidden = alm.extract_atoms_hidden_states(
                [ids], [attn], atom_embeds=empty_embeds,
            )  # (1, K, hidden_dim)
            cond = mapper(hidden.to(next(mapper.parameters()).dtype))  # (1, 512)

        print(f"  prompt {i}: hidden {tuple(hidden.shape)}, "
              f"cond {tuple(cond.shape)}, "
              f"cond mean={cond.mean().item():+.3f} std={cond.std().item():.3f}")

    print("\n[verify] OK — Stage 3a ckpt loads cleanly and produces conditioning vectors.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--stage3a_dir", required=True,
                   help="Stage 3a step dir (lora_adapter/ + projector_and_state.pt + atoms_mapper.pt)")
    main(p.parse_args())
