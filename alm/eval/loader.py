"""Load an ALM checkpoint for evaluation.

Stage 2: LoRA adapter (safetensors) + projector blob from a step=N/ dir.
Stage 1: projector .pt only.

Merges LoRA into the base for inference speed by default; pass merge_lora=False
to keep adapters live (e.g. when comparing to an unmerged forward).
"""
import json
import sys
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from alm import AtomisticLanguageModel  # alm/alm.py — module, not package


LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"]


def load_alm(checkpoint=None, stage1_projector=None,
             base_model="Qwen/Qwen3-8B",
             lora_rank=None, lora_alpha=None,
             merge_lora=True, max_atoms=2048 - 256, device=None,
             use_cached_embeddings=True, is_trainable=False,
             num_output_atom_tokens: int = 8):
    """Returns (model, tokenizer).

    Exactly one of `checkpoint` (Stage 2 dir with lora_adapter/ + projector_and_state.pt)
    or `stage1_projector` (.pt) must be set. With Stage 1, no LoRA is attached and
    `merge_lora` is ignored.

    is_trainable=True: route LoRA load through PEFT's `from_pretrained(..., is_trainable=True)`
    so `lora_A.default.weight` / `lora_B.default.weight` come out with `requires_grad=True`.
    Used by Stage 3a joint training. Implies `merge_lora=False`. Caller sets train/eval mode.
    """
    if (checkpoint is None) == (stage1_projector is None):
        raise ValueError("pass exactly one of --checkpoint or --stage1_projector")
    if is_trainable and merge_lora:
        raise ValueError("is_trainable=True requires merge_lora=False")
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AtomisticLanguageModel(
        llm_name=base_model, atomistic_model_name="orb_v3_direct_20_omat",
        device=device, use_cached_embeddings=use_cached_embeddings, max_atoms=max_atoms,
        num_output_atom_tokens=num_output_atom_tokens,
    )

    if stage1_projector is not None:
        ckpt = torch.load(stage1_projector, map_location=device)
        proj_state = ckpt.get("projector_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        model.projector.load_state_dict(proj_state)
    else:
        adapter_dir = Path(checkpoint) / "lora_adapter"
        # Read rank/alpha from the saved adapter_config.json so the loader works
        # for any Stage 2 ckpt regardless of the rank it was trained at.
        with open(adapter_dir / "adapter_config.json") as f:
            saved_cfg = json.load(f)
        r = lora_rank if lora_rank is not None else saved_cfg["r"]
        a = lora_alpha if lora_alpha is not None else saved_cfg["lora_alpha"]
        lora_cfg = LoraConfig(
            r=r, lora_alpha=a, lora_dropout=0.0,
            bias="none", task_type="CAUSAL_LM", target_modules=LORA_TARGET_MODULES,
        )
        # get_peft_model creates fresh LoRA with requires_grad=True on lora_A/B.
        # The manual state_dict load below overwrites the LoRA values from the saved
        # adapter without changing requires_grad — so this path works for both eval
        # (is_trainable=False) and training (is_trainable=True). The PEFT-native
        # from_pretrained path can't be used because it does a strict shape-check
        # that the vocab-resize migration shim is here to side-step.
        model.llm = get_peft_model(model.llm, lora_cfg)
        sd = load_file(str(adapter_dir / "adapter_model.safetensors"))
        # PEFT strips the adapter name during save_pretrained; re-insert "default".
        sd = {k.replace(".lora_A.weight", ".lora_A.default.weight")
               .replace(".lora_B.weight", ".lora_B.default.weight"): v
              for k, v in sd.items()}
        # Vocab-resize migration: alm.py can grow OR shrink the vocab depending on
        # `num_output_atom_tokens` (e.g., Stage 2 ckpt was trained with K=8; loading
        # into a K=4 model needs the saved rows truncated, not extended). Both
        # directions copy the matching prefix; either drops or freshly-inits the rest.
        cur_sd = model.llm.state_dict()
        for k in list(sd.keys()):
            if k in cur_sd and sd[k].shape != cur_sd[k].shape:
                old, cur = sd[k], cur_sd[k]
                if old.ndim != cur.ndim:
                    continue
                if all(o <= c for o, c in zip(old.shape, cur.shape)):
                    # saved smaller than current → place saved into the matching prefix
                    new = cur.clone()
                    new[tuple(slice(0, s) for s in old.shape)] = old.to(new.dtype)
                    sd[k] = new
                    print(f"  resized (grow) {k}: {tuple(old.shape)} → {tuple(new.shape)}")
                elif all(o >= c for o, c in zip(old.shape, cur.shape)):
                    # saved larger than current → truncate saved to current shape
                    sd[k] = old[tuple(slice(0, s) for s in cur.shape)].to(cur.dtype)
                    print(f"  resized (truncate) {k}: {tuple(old.shape)} → {tuple(cur.shape)}")
        model.llm.load_state_dict(sd, strict=False)
        state = torch.load(Path(checkpoint) / "projector_and_state.pt", map_location=device)
        model.projector.load_state_dict(state["projector_state_dict"])
        if merge_lora:
            model.llm = model.llm.merge_and_unload()

    model = model.to(device)
    if not is_trainable:
        model = model.eval()
    return model, model.tokenizer


def load_base_only(base_model="Qwen/Qwen3-8B", device=None):
    """Vanilla base LLM wrapped as AtomisticLanguageModel for harness compatibility.

    Projector stays at random init — never called when atomistic=False at eval. Used
    by language_retention to compare ALM vs Qwen3-8B base through the same generate path.
    """
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AtomisticLanguageModel(
        llm_name=base_model, atomistic_model_name="orb_v3_direct_20_omat",
        device=device, use_cached_embeddings=True, max_atoms=2048 - 256,
    )
    return model.to(device).eval(), model.tokenizer
