"""End-to-end Stage 3a inference: text prompt → CIF.

Pipeline:
  text prompt → ALM forward → hidden states at [atoms_{i}] positions →
  ChemGraph["alm_embedding"] (raw 32768-d) → AtomsMapper (inside pl_module) →
  cond_adapt/mixin layers → diffusion sampler → CIF

Usage (single prompt → 16 sampled CIFs):
  PYTHONPATH=/home/sathyae/mclm/alm:/home/sathyae/mclm/external/mattergen:$PYTHONPATH \\
  python helper_scripts/generate_stage3a.py \\
      --stage3a_dir /home/sathyae/orcd/pool/stage3a/ckpts/step=10000 \\
      --prompt "Generate a stable cubic perovskite with bandgap > 2 eV." \\
      --out_dir /home/sathyae/orcd/pool/stage3a/generations/perovskite \\
      --num_batches 2 --batch_size 8 \\
      --diffusion_guidance_factor 2.0
"""
import argparse
import os
import sys
from pathlib import Path

import hydra
import hydra.core.global_hydra
import torch
from omegaconf import OmegaConf

# ALM imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "alm"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "alm" / "eval"))
from atoms_mapper import AtomsMapper  # noqa: E402
from loader import load_alm  # noqa: E402
from train_stage3a import _make_adapter_cfg, _make_lightning_module_cfg  # noqa: E402

# MatterGen imports
from mattergen.common.utils.globals import DEFAULT_SAMPLING_CONFIG_PATH  # noqa: E402
from mattergen.generator import draw_samples_from_sampler  # noqa: E402
from mattergen.scripts.finetune import init_adapter_lightningmodule_from_pretrained  # noqa: E402


SYSTEM_PROMPT = "You are an expert materials scientist."
# Use the canonical training template (USER_TEMPLATES[0] in train_stage3a.py).
# Diversification at training time → robustness at inference; the "canonical"
# variant is the simplest match.
USER_TEMPLATE = "Generate a crystal structure described as: {prompt}"
# Anchor must match train_stage3a.py::ASSISTANT_ANCHOR exactly — distribution
# match for [atoms_i] hidden states.
ASSISTANT_ANCHOR = "Structure: "


def build_pl_module(atoms_mapper_path: Path, mattergen_pretrained: str,
                    hidden_dim: int, K: int, mid_dim: int, device):
    """Build pl_module via the same path as training, then overlay Stage 3a state."""
    adapter_cfg = _make_adapter_cfg(
        mattergen_pretrained, full_finetuning=False,
        hidden_dim=hidden_dim, K=K, mid_dim=mid_dim,
    )
    lm_cfg = _make_lightning_module_cfg(lr=1e-4)
    pl_module, _ = init_adapter_lightningmodule_from_pretrained(adapter_cfg, lm_cfg)
    pl_module = pl_module.to(device).eval()

    # Overlay Stage 3a's AtomsMapper + cond_adapt/mixin
    ckpt = torch.load(atoms_mapper_path, map_location=device)
    diffusion_model = pl_module.diffusion_module.model
    atoms_mapper = (
        diffusion_model.property_embeddings_adapt["alm_embedding"]
        .conditional_embedding_module
    )
    atoms_mapper.load_state_dict(ckpt["atoms_mapper_state_dict"])

    if "trainable_state_dict" in ckpt:
        cur = diffusion_model.state_dict()
        cur.update(ckpt["trainable_state_dict"])
        diffusion_model.load_state_dict(cur, strict=True)
        print(f"[gen] overlaid {len(ckpt['trainable_state_dict'])} trainable tensors "
              f"(cond_adapt/mixin)")

    return pl_module


def get_alm_embedding(alm, tokenizer, prompt: str, device) -> torch.Tensor:
    """Run ALM forward on the prompt, return the (K*4096,) flattened hidden states."""
    output_atoms_str = "".join(alm.output_atom_tokens)
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_TEMPLATE.format(prompt=prompt)},
        # Assistant turn = short anchor + K [atoms_i] tokens. Must match the
        # training-time format (ASSISTANT_ANCHOR in train_stage3a.py).
        {"role": "assistant", "content": ASSISTANT_ANCHOR + output_atoms_str},
    ]
    full_ids = tokenizer.apply_chat_template(
        msgs, tokenize=True, add_generation_prompt=False,
        enable_thinking=False, truncation=True, max_length=2048,
    )
    ids = torch.tensor(full_ids, dtype=torch.long, device=device)
    attn = torch.ones_like(ids)

    # Verify all K [atoms_i] tokens survived truncation
    K = len(alm.output_atom_token_ids)
    atoms_id_set = set(alm.output_atom_token_ids)
    n_present = sum(int(t.item()) in atoms_id_set for t in ids)
    if n_present != K:
        raise RuntimeError(f"prompt was truncated: only {n_present}/{K} [atoms_i] tokens present")

    empty_embeds = [torch.zeros(0, 256, dtype=torch.float32, device=device)]
    with torch.no_grad():
        hidden = alm.extract_atoms_hidden_states(
            [ids], [attn], atom_embeds=empty_embeds,
        )  # (1, K, 4096)
    return hidden.flatten().float()  # (K*4096,)


def build_sampler_and_loader(pl_module, batch_size: int, num_batches: int,
                             num_atoms_distribution: str, alm_emb_vec: torch.Tensor,
                             diffusion_guidance_factor: float):
    """Mirror generator.CrystalGenerator.generate's hydra-config pipeline, but with a
    runtime-supplied alm_embedding vector instead of a scalar property."""
    overrides = [
        f"+condition_loader_partial.num_atoms_distribution={num_atoms_distribution}",
        f"+condition_loader_partial.batch_size={batch_size}",
        f"+condition_loader_partial.num_samples={num_batches * batch_size}",
        f"sampler_partial.guidance_scale={diffusion_guidance_factor}",
    ]
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    with hydra.initialize_config_dir(os.path.abspath(str(DEFAULT_SAMPLING_CONFIG_PATH)),
                                     version_base="1.1"):
        sampling_cfg = hydra.compose(config_name="default", overrides=overrides)

    # Build the condition_loader. SetProperty stamps alm_embedding onto every
    # ChemGraph in the dataset; pyg collation concatenates along dim 0 of the
    # stamped tensor. We pass shape (1, K*4096) so cat → (B, K*4096); a flat
    # (K*4096,) tensor would cat to (B*K*4096,) and break AtomsMapper.
    condition_loader_partial = hydra.utils.instantiate(sampling_cfg.condition_loader_partial)
    condition_loader = condition_loader_partial(
        properties={"alm_embedding": alm_emb_vec.detach().cpu().unsqueeze(0)},
    )

    # Build the sampler. sampler_partial expects pl_module=...
    sampler_partial = hydra.utils.instantiate(sampling_cfg.sampler_partial)
    sampler = sampler_partial(pl_module=pl_module)
    return sampler, condition_loader


def main(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── 1. Load ALM from Stage 2 ckpt (LoRA + projector merged) ────────────
    # Stage 3a doesn't train LoRA — the LLM-side weights are exactly what Stage 2
    # produced. Inference takes the Stage 2 ckpt directly.
    print(f"[gen] loading ALM from {args.alm_checkpoint} ...", flush=True)
    alm, tokenizer = load_alm(
        checkpoint=args.alm_checkpoint,
        merge_lora=True,
        use_cached_embeddings=True,
        device=device,
    )
    alm.eval()
    K = len(alm.output_atom_token_ids)
    print(f"[gen] K={K}, hidden_dim={alm.llm_hidden_dim}")

    # ── 2. Build MatterGen pl_module + overlay trained AtomsMapper ─────────
    print(f"[gen] building MatterGen adapter ({args.mattergen_pretrained}) ...", flush=True)
    print(f"[gen] overlaying AtomsMapper + cond_adapt/mixin from {args.atoms_mapper}")
    pl_module = build_pl_module(
        Path(args.atoms_mapper), args.mattergen_pretrained,
        hidden_dim=alm.llm_hidden_dim, K=K, mid_dim=2048, device=device,
    )

    # ── 3. ALM forward → conditioning vector ───────────────────────────────
    print(f"[gen] computing alm_embedding for prompt: {args.prompt!r}", flush=True)
    alm_emb = get_alm_embedding(alm, tokenizer, args.prompt, device)  # (in_dim,)
    print(f"[gen] alm_embedding shape: {tuple(alm_emb.shape)}, "
          f"mean={alm_emb.mean().item():+.3f}, std={alm_emb.std().item():.3f}")

    # ALM no longer needed; free GPU memory before sampling.
    del alm
    torch.cuda.empty_cache()

    # ── 4. Build sampler + condition_loader ────────────────────────────────
    print(f"[gen] building sampler (batch_size={args.batch_size}, "
          f"num_batches={args.num_batches}, "
          f"guidance={args.diffusion_guidance_factor}, "
          f"num_atoms_distribution={args.num_atoms_distribution}) ...", flush=True)
    sampler, condition_loader = build_sampler_and_loader(
        pl_module=pl_module,
        batch_size=args.batch_size,
        num_batches=args.num_batches,
        num_atoms_distribution=args.num_atoms_distribution,
        alm_emb_vec=alm_emb,
        diffusion_guidance_factor=args.diffusion_guidance_factor,
    )

    # ── 4b. Optional element masking ──────────────────────────────────────
    if args.allowed_elements:
        elements = [s.strip() for s in args.allowed_elements.split(",") if s.strip()]
        _ensure_element_mask_installed(pl_module)
        pl_module._element_mask_state.allowed_z = _z_set_from_elements(elements)
        print(f"[gen] element mask ON: allowed Z = {sorted(pl_module._element_mask_state.allowed_z)} "
              f"(elements: {elements})", flush=True)

    # ── 5. Sample ──────────────────────────────────────────────────────────
    print(f"[gen] sampling {args.num_batches * args.batch_size} structures → {out_dir}",
          flush=True)
    # cfg is only used by dump_trajectories (we set record_trajectories=False by default,
    # but draw_samples_from_sampler asserts cfg is not None whenever output_path is set).
    structures = draw_samples_from_sampler(
        sampler=sampler,
        condition_loader=condition_loader,
        properties_to_condition_on={"alm_embedding": alm_emb.detach().cpu()},
        output_path=out_dir,
        cfg=OmegaConf.create({}),
        record_trajectories=args.record_trajectories,
    )

    # Save the prompt + alm_embedding alongside the CIFs for reproducibility.
    torch.save({
        "prompt": args.prompt,
        "alm_embedding": alm_emb.detach().cpu(),
        "alm_checkpoint": str(args.alm_checkpoint),
        "atoms_mapper": str(args.atoms_mapper),
        "mattergen_pretrained": args.mattergen_pretrained,
        "diffusion_guidance_factor": args.diffusion_guidance_factor,
    }, out_dir / "stage3a_inference_meta.pt")

    print(f"\n[gen] DONE — {len(structures)} structures generated. Outputs in {out_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# Element-mask plumbing (CSP-style hard composition constraint at sample time)
# ─────────────────────────────────────────────────────────────────────────────

class _ElementMaskState:
    """Mutable holder for the current per-prompt element mask. Read by
    `_masked_score_fn` on each denoising step, written by `generate_for_prompts`
    once per prompt before that prompt's batch is sampled."""
    def __init__(self):
        self.allowed_z: set[int] | None = None  # set of allowed atomic numbers (Z=1..100)


def _ensure_element_mask_installed(pl_module):
    """Idempotently install a score_fn wrapper that masks atomic_numbers logits.

    When `pl_module._element_mask_state.allowed_z` is None → wrapper is a no-op.
    When set to a set of Zs → atomic_numbers logits for any Z not in the set are
    set to -1e9 (effectively masked out at the categorical sampling step inside
    D3PMAncestralSamplingPredictor). The MASK token (last logit when
    with_mask_type=True) is left untouched so absorbing-state diffusion still
    works mid-trajectory.

    Mathematical correctness under CFG: if both conditional and unconditional
    score_fn calls go through this wrapper, both atomic_numbers logits get the
    SAME mask added, which commutes with the CFG linear combine. The final
    combined logits have the mask preserved.
    """
    if getattr(pl_module, "_element_mask_installed", False):
        return
    import torch as _torch
    state = _ElementMaskState()
    pl_module._element_mask_state = state
    diffusion_module = pl_module.diffusion_module
    orig_score_fn = diffusion_module.score_fn

    def masked_score_fn(x, t):
        output = orig_score_fn(x, t)
        if state.allowed_z is None:
            return output
        try:
            logits = output["atomic_numbers"]
        except (KeyError, TypeError):
            return output
        if not _torch.is_tensor(logits) or logits.ndim != 2:
            return output
        n_classes = logits.shape[-1]
        # Zero-based: index 0 ↔ Z=1, index 99 ↔ Z=100. Index 100 (if present) is
        # the MASK token used during absorbing-state diffusion — leave it unmasked.
        mask = _torch.full((n_classes,), -1.0e9, device=logits.device, dtype=logits.dtype)
        for z in state.allowed_z:
            i = int(z) - 1
            if 0 <= i < min(n_classes, 100):
                mask[i] = 0.0
        if n_classes > 100:
            mask[100:] = 0.0  # don't mask the MASK token(s)
        return output.replace(atomic_numbers=logits + mask)

    diffusion_module.score_fn = masked_score_fn
    pl_module._element_mask_installed = True


def _z_set_from_elements(elements):
    """Convert an iterable of element symbols (['Cu', 'Ni']) to {Z, ...} via ASE."""
    from ase.data import atomic_numbers as _ase_z
    out = set()
    for sym in elements:
        s = sym.strip()
        if s in _ase_z:
            out.add(int(_ase_z[s]))
    return out


def generate_for_prompts(
    prompts,
    alm,
    tokenizer,
    pl_module,
    out_root,
    batch_size: int = 4,
    num_batches: int = 1,
    diffusion_guidance_factor: float = 1.0,
    num_atoms_distribution: str = "ALEX_MP_20",
    record_trajectories: bool = False,
    save_meta: bool = True,
    prompt_ids=None,
    allowed_elements_per_prompt=None,
):
    """Batched generation: load model ONCE, loop over many prompts.

    Used by the eval harness (eval_csp.py, eval_dng.py, eval_text_conditional.py)
    to amortize the ~30s ALM+MatterGen load over many prompts. Each prompt's
    outputs land in `out_root/<id>/` (where <id> is from prompt_ids if provided,
    else `prompt_{i:04d}`).

    Returns
    -------
    list[list[Structure]]: outer index = prompt; inner = the batch_size*num_batches
                           generations for that prompt.
    """
    from pymatgen.io.ase import AseAtomsAdaptor  # noqa: F401  (used downstream)
    device = next(pl_module.parameters()).device
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    if prompt_ids is None:
        prompt_ids = [f"prompt_{i:04d}" for i in range(len(prompts))]
    if len(prompt_ids) != len(prompts):
        raise ValueError("prompt_ids must align 1:1 with prompts")
    if allowed_elements_per_prompt is not None and len(allowed_elements_per_prompt) != len(prompts):
        raise ValueError("allowed_elements_per_prompt must align 1:1 with prompts")
    # Lazy-install the element-mask wrapper once. State stays as None below
    # unless we actually have an allowed set for the current prompt.
    if allowed_elements_per_prompt is not None:
        _ensure_element_mask_installed(pl_module)

    all_results = []
    n_skipped = 0
    for i, (pid, prompt) in enumerate(zip(prompt_ids, prompts)):
        # Apply per-prompt element mask (or clear it for this prompt if None).
        if allowed_elements_per_prompt is not None:
            ae = allowed_elements_per_prompt[i]
            pl_module._element_mask_state.allowed_z = (
                _z_set_from_elements(ae) if ae is not None else None
            )
        prompt_dir = out_root / str(pid)
        prompt_dir.mkdir(parents=True, exist_ok=True)
        try:
            alm_emb = get_alm_embedding(alm, tokenizer, prompt, device)
            sampler, condition_loader = build_sampler_and_loader(
                pl_module=pl_module,
                batch_size=batch_size,
                num_batches=num_batches,
                num_atoms_distribution=num_atoms_distribution,
                alm_emb_vec=alm_emb,
                diffusion_guidance_factor=diffusion_guidance_factor,
            )
            structures = draw_samples_from_sampler(
                sampler=sampler,
                condition_loader=condition_loader,
                properties_to_condition_on={"alm_embedding": alm_emb.detach().cpu()},
                output_path=prompt_dir,
                cfg=OmegaConf.create({}),
                record_trajectories=record_trajectories,
            )
            if save_meta:
                torch.save({
                    "prompt": prompt,
                    "prompt_id": pid,
                    "alm_embedding": alm_emb.detach().cpu(),
                    "diffusion_guidance_factor": diffusion_guidance_factor,
                    "num_atoms_distribution": num_atoms_distribution,
                }, prompt_dir / "stage3a_inference_meta.pt")
        except Exception as exc:
            # MatterGen's sampler can crash mid-denoise on stochastic edge cases
            # (e.g., GemNet `torch.max(empty)` in efficient.py:158 when an
            # intermediate graph has no angular-basis triplets). One bad prompt
            # shouldn't kill a 100-row eval — log + skip + continue.
            n_skipped += 1
            print(f"[gen-batch] {i+1}/{len(prompts)}: id={pid} FAILED — "
                  f"{type(exc).__name__}: {exc}", flush=True)
            structures = []
            # Free any partial GPU state before the next iteration.
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
        all_results.append(structures)
        print(f"[gen-batch] {i+1}/{len(prompts)}: id={pid} → {len(structures)} structures"
              + (" (skipped)" if not structures else ""),
              flush=True)
    if n_skipped:
        print(f"[gen-batch] SUMMARY: {n_skipped}/{len(prompts)} prompts FAILED "
              f"(stochastic sampler crashes); rest succeeded.", flush=True)
    return all_results


def load_alm_and_pl_module(
    alm_checkpoint,
    atoms_mapper,
    mattergen_pretrained: str = "mattergen_base",
    device=None,
):
    """One-shot load helper for eval scripts. Returns (alm, tokenizer, pl_module, K).

    Mirrors steps 1-2 of `main()` but without freeing alm — the eval scripts
    keep alm alive across many `get_alm_embedding` calls.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[gen] loading ALM from {alm_checkpoint} ...", flush=True)
    alm, tokenizer = load_alm(
        checkpoint=alm_checkpoint, merge_lora=True,
        use_cached_embeddings=True, device=device,
    )
    alm.eval()
    K = len(alm.output_atom_token_ids)
    print(f"[gen] K={K}, hidden_dim={alm.llm_hidden_dim}")
    print(f"[gen] building MatterGen adapter ({mattergen_pretrained}) ...", flush=True)
    print(f"[gen] overlaying AtomsMapper + cond_adapt/mixin from {atoms_mapper}")
    pl_module = build_pl_module(
        Path(atoms_mapper), mattergen_pretrained,
        hidden_dim=alm.llm_hidden_dim, K=K, mid_dim=2048, device=device,
    )
    return alm, tokenizer, pl_module, K


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--alm_checkpoint", required=True,
                   help="Stage 2 ckpt dir (lora_adapter/ + projector_and_state.pt). Stage 3a "
                        "doesn't train the LLM, so this is just the Stage 2 ckpt used at training.")
    p.add_argument("--atoms_mapper", required=True,
                   help="Path to atoms_mapper.pt produced by train_stage3a.py")
    p.add_argument("--prompt", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--mattergen_pretrained", default="mattergen_base",
                   help="Same value used at training time")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--num_batches", type=int, default=2)
    p.add_argument("--diffusion_guidance_factor", type=float, default=2.0,
                   help="Classifier-free guidance scale. 0 = unconditional, ~2.0 typical")
    p.add_argument("--num_atoms_distribution", default="ALEX_MP_20",
                   help="Empirical distribution for sampling structure size")
    p.add_argument("--record_trajectories", action="store_true",
                   help="Save denoising trajectories (extra disk; nice for figures)")
    p.add_argument("--allowed_elements", type=str, default=None,
                   help="Comma-separated element symbols (e.g. 'Cu,Ni'). When set, the "
                        "score model's atomic-number logits are masked at every denoising "
                        "step so only these elements can be sampled. CFG-safe.")
    main(p.parse_args())
