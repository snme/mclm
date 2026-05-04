"""Stage 3a: Joint training with LoRA-trainable LLM (GILL/DreamLLM pattern).

Reuses Stage 2's existing LoRA infrastructure (rank 64, α 128 on q/k/v/o/gate/up/down)
as the LLM-side adaptation mechanism — gradient from L_diff flows back through
AtomsMapper into the LoRA adapters. Base Qwen3 weights, projector, encoder, and
[atoms_i] input embeddings stay frozen. MatterGen backbone stays frozen.

Trainable parameters:
  - AtomsMapper (~70M params): (B, K*4096) → (B, 512)
  - GemNetTCtrl cond_adapt/mixin layers (~1M params, zero-init at start)
  - Stage 2 LoRA on Qwen3 (~188M params, continued from Stage 2 ckpt)

Frozen:
  - Qwen3-8B base, Stage 2 projector, OrbV3 encoder
  - All token embedding rows including [atoms_i] (LoRA on attention/MLP linears
    is the LLM-side adaptation — no need for embedding-row grad hooks)
  - MatterGen pretrained backbone

Gradient path:
  L_diff → cond_adapt/mixin → AtomsMapper → LLM hidden states at [atoms_i] positions
         → through gradient-checkpointed Qwen3 → LoRA A/B matrices on every linear

Data (pairs.parquet from build_stage3a_pairs.py):
  row_id, parent, source_idx, n_atoms, narrative, atoms_struct

Run (single GPU smoke test):
  PYTHONPATH=/home/sathyae/mclm/alm:$PYTHONPATH python alm/train_stage3a.py \\
      --alm_checkpoint  /path/to/stage2/step=12000 \\
      --pairs_parquet   /path/to/stage3a/pairs.parquet \\
      --mattergen_pretrained mattergen_base \\
      --out_dir         /path/to/stage3a_ckpts \\
      --total_steps     200 --batch_size 2

Run (8× H200 DDP):
  PYTHONPATH=/home/sathyae/mclm/alm:$PYTHONPATH \\
  torchrun --nnodes=1 --nproc-per-node=8 --rdzv-backend=c10d \\
      --rdzv-endpoint=$(hostname):29508 alm/train_stage3a.py \\
      --alm_checkpoint  /path/to/stage2/step=12000 \\
      --pairs_parquet   /path/to/stage3a/pairs.parquet \\
      --mattergen_pretrained mattergen_base \\
      --out_dir         /path/to/stage3a_ckpts

Resume:
  Add --resume_atoms_mapper /path/to/stage3a_ckpts/step=N/atoms_mapper.pt
"""
import argparse
import json
import os
import sys
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler

# MatterGen imports
from omegaconf import OmegaConf, open_dict
from hydra.utils import instantiate
import hydra.core.global_hydra

from mattergen.scripts.finetune import init_adapter_lightningmodule_from_pretrained
from mattergen.common.data.chemgraph import ChemGraph
from mattergen.common.data.collate import collate as mg_collate
from mattergen.common.data.transform import symmetrize_lattice, set_chemical_system_string
from mattergen.property_embeddings import SetEmbeddingType

# ALM imports (PYTHONPATH must include /home/sathyae/mclm/alm)
from alm import AtomisticLanguageModel
from aux_heads import build_aux_head
from eval.loader import load_alm

try:
    import wandb
    _WANDB = True
except ImportError:
    _WANDB = False


# ─────────────────────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are an expert materials scientist. When asked to generate a crystal "
    "structure, provide a detailed description and conclude with the structure "
    "generation tokens."
)

# NOTE: user_prompt and assistant_anchor are pre-baked into pairs.parquet by
# helper_scripts/build_stage3a_pairs.py — see USER_TEMPLATES / ASSISTANT_ANCHORS
# there. The trainer only consumes the literal columns, no template logic here.


def _atoms_struct_to_tensors(struct: dict):
    """Convert atoms_struct dict → (frac_pos, cell, atomic_numbers)."""
    from ase import Atoms
    from ase.data import atomic_numbers as ase_Z

    cell = np.asarray(struct["lattice_mat"], dtype=np.float32)
    coords = np.asarray(struct["coords"], dtype=np.float64)
    symbols = [s.strip() for s in struct["elements"]]

    if struct.get("cartesian", False):
        atoms = Atoms(symbols=symbols, positions=coords, cell=cell, pbc=True)
        frac = atoms.get_scaled_positions(wrap=True).astype(np.float32)
    else:
        frac = (np.mod(coords, 1.0)).astype(np.float32)

    Z = np.array([ase_Z[s] for s in symbols], dtype=np.int64)
    return frac, cell, Z


class Stage3aDataset(Dataset):
    """Reads pairs.parquet; each sample has (narrative, atoms_struct).

    When `aux_target_kind` is set, also emits an `aux_target` field per sample for
    the auxiliary supervision head (see `alm/aux_heads.py`):
      - "composition": (100,) multi-hot over Z=1..100, computed once at __init__
      - "orbv3_mean":  (256,) precomputed mean OrbV3 feature, loaded by row_id
                       from `orbv3_means_path` (built by precompute_orbv3_means.py)
    """

    def __init__(self, parquet_path: str, tokenizer, max_num_tokens: int = 2048,
                 aux_target_kind: str | None = None,
                 orbv3_means_path: str | None = None,
                 num_output_atom_tokens: int = 8):
        import pyarrow.parquet as pq
        table = pq.read_table(parquet_path)
        self.rows = table.to_pydict()
        self.n = len(self.rows["row_id"])
        self.tokenizer = tokenizer
        self.max_num_tokens = max_num_tokens
        self.num_output_atom_tokens = num_output_atom_tokens
        self.output_atom_token_ids = tokenizer.convert_tokens_to_ids(
            [f"[atoms_{i}]" for i in range(num_output_atom_tokens)]
        )
        self.output_atom_str = "".join(f"[atoms_{i}]" for i in range(num_output_atom_tokens))

        # ── Aux target precomputation / cache loading ───────────────────────
        self.aux_target_kind = aux_target_kind
        self.aux_targets = None        # (n, target_dim) numpy array (composition path)
        self.orbv3_means = None        # memmap (n_means, 256) (orbv3_mean path)
        self.aux_target_indices = None # int64 (n,) row → orbv3_means index, -1 if missing

        if aux_target_kind == "composition":
            from composition_utils import composition_multihot, symbol_to_z
            sym2z = symbol_to_z()
            self.aux_targets = np.zeros((self.n, 100), dtype=np.float32)
            atoms_structs = self.rows["atoms_struct"]
            for i in range(self.n):
                struct = atoms_structs[i]
                if hasattr(struct, "as_py"):
                    struct = struct.as_py()
                self.aux_targets[i] = composition_multihot(struct["elements"], sym2z)
            if is_main_process():
                pos_per_row = self.aux_targets.sum(axis=1).mean()
                print(f"[stage3a] aux_target=composition: precomputed (n={self.n}, dim=100), "
                      f"avg ~{pos_per_row:.2f} elements/structure")

        elif aux_target_kind == "composition_count":
            from composition_utils import (
                composition_count_vec, MAX_COUNT, N_ELEMENTS, symbol_to_z,
            )
            sym2z = symbol_to_z()
            # Same (n, N_ELEMENTS) shape as the presence target — but holds
            # integer counts (clamped to MAX_COUNT) instead of {0, 1}. Stored
            # float32 for clean torch.stack alongside the rest of the batch.
            self.aux_targets = np.zeros((self.n, N_ELEMENTS), dtype=np.float32)
            n_clamped = 0
            atoms_structs = self.rows["atoms_struct"]
            for i in range(self.n):
                struct = atoms_structs[i]
                if hasattr(struct, "as_py"):
                    struct = struct.as_py()
                # Use int64 first to detect saturation, then cast to float32.
                v = composition_count_vec(struct["elements"], sym2z, dtype=np.int64)
                # Count rows where any slot hit MAX_COUNT before the clip
                # (composition_count_vec already clamped, so check raw element-list length).
                # Cheap heuristic: if any element symbol appears > MAX_COUNT times in the list.
                if v.max() == MAX_COUNT:
                    # Re-count without clamping to know if it actually saturated.
                    raw_max = max(
                        (struct["elements"].count(sym) for sym in set(struct["elements"])),
                        default=0,
                    )
                    if raw_max > MAX_COUNT:
                        n_clamped += 1
                self.aux_targets[i] = v.astype(np.float32, copy=False)
            if is_main_process():
                pos_per_row = (self.aux_targets > 0).sum(axis=1).mean()
                avg_count_when_pos = (
                    self.aux_targets[self.aux_targets > 0].mean()
                    if (self.aux_targets > 0).any() else 0.0
                )
                print(f"[stage3a] aux_target=composition_count: precomputed "
                      f"(n={self.n}, dim={N_ELEMENTS}, MAX_COUNT={MAX_COUNT}), "
                      f"avg ~{pos_per_row:.2f} elements/structure, "
                      f"avg count-when-present {avg_count_when_pos:.2f}, "
                      f"{n_clamped} rows had ≥1 element clamped to MAX_COUNT")

        elif aux_target_kind == "orbv3_mean":
            if orbv3_means_path is None:
                raise ValueError("--orbv3_means_path required for aux_target_kind=orbv3_mean")
            bin_path = Path(orbv3_means_path)
            idx_path = bin_path.with_suffix(".idx.json")
            with open(idx_path) as f:
                row_to_means_idx = json.load(f)  # row_id (str) → int index into means file
            self.orbv3_means = np.memmap(bin_path, dtype=np.float32, mode="r").reshape(-1, 256)
            row_ids = self.rows["row_id"]
            self.aux_target_indices = np.full(self.n, -1, dtype=np.int64)
            missing = 0
            for i, rid in enumerate(row_ids):
                idx = row_to_means_idx.get(rid)
                if idx is None:
                    missing += 1
                else:
                    self.aux_target_indices[i] = idx
            if is_main_process():
                print(f"[stage3a] aux_target=orbv3_mean: loaded {self.orbv3_means.shape[0]} means from "
                      f"{bin_path}, {missing}/{self.n} rows have no target (will be skipped)")
        elif aux_target_kind not in (None, "none", ""):
            raise ValueError(f"unknown aux_target_kind: {aux_target_kind!r}")

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        # All ChatML formatting (paraphrased user template + anchor) is pre-baked
        # into pairs.parquet at build time (see helper_scripts/build_stage3a_pairs.py).
        # Trainer just reads the literal columns and tokenizes.
        user_prompt = self.rows["user_prompt"][idx]
        assistant_anchor = self.rows["assistant_anchor"][idx]
        atoms_struct = self.rows["atoms_struct"][idx]
        if hasattr(atoms_struct, "as_py"):
            atoms_struct = atoms_struct.as_py()

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        prompt_ids = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            enable_thinking=False,
            truncation=True, max_length=self.max_num_tokens - 16,
        )
        assistant_content = assistant_anchor + self.output_atom_str
        full_ids = self.tokenizer.apply_chat_template(
            messages + [{"role": "assistant", "content": assistant_content}],
            tokenize=True, add_generation_prompt=False,
            enable_thinking=False, truncation=True, max_length=self.max_num_tokens,
        )
        # Verify all K [atoms_{i}] token IDs are present at end of full_ids
        # (K = self.num_output_atom_tokens, configurable via --num_output_atom_tokens)
        trailing = full_ids[-(len(self.output_atom_token_ids) + 2):]  # +2 for EOS/IM_END
        present = set(trailing) & set(self.output_atom_token_ids)
        if len(present) < len(self.output_atom_token_ids):
            # Truncation ate some [atoms_{i}] tokens — skip this sample by returning None;
            # collate will filter them out.
            return None

        input_ids = torch.tensor(full_ids, dtype=torch.long)
        attention_mask = torch.ones(len(full_ids), dtype=torch.long)
        labels = torch.tensor(
            [-100] * len(prompt_ids) + full_ids[len(prompt_ids):], dtype=torch.long
        )

        # Structure data for MatterGen ChemGraph
        frac, cell, Z = _atoms_struct_to_tensors(atoms_struct)
        sample = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "frac": torch.from_numpy(frac),      # (N_atoms, 3)
            "cell": torch.from_numpy(cell),       # (3, 3)
            "Z": torch.from_numpy(Z),             # (N_atoms,)
            "n_atoms": torch.tensor(len(Z), dtype=torch.long),
        }
        # Optional aux target for supervised mapper-output head.
        if self.aux_target_kind in ("composition", "composition_count"):
            sample["aux_target"] = torch.from_numpy(self.aux_targets[idx])
        elif self.aux_target_kind == "orbv3_mean":
            midx = int(self.aux_target_indices[idx])
            if midx < 0:
                # No precomputed mean for this row — skip the sample entirely
                return None
            sample["aux_target"] = torch.from_numpy(np.asarray(self.orbv3_means[midx]).copy())
        return sample


def stage3a_collate(samples):
    """Collate a list of samples from Stage3aDataset, filtering None entries."""
    samples = [s for s in samples if s is not None]
    if not samples:
        return None

    input_ids = [s["input_ids"] for s in samples]
    attention_mask = [s["attention_mask"] for s in samples]
    labels = [s["labels"] for s in samples]

    # Pad text tensors
    max_len = max(x.shape[0] for x in input_ids)
    input_ids_padded = torch.stack(
        [torch.nn.functional.pad(x, (0, max_len - x.shape[0]), value=0) for x in input_ids]
    )
    attn_mask_padded = torch.stack(
        [torch.nn.functional.pad(x, (0, max_len - x.shape[0]), value=0) for x in attention_mask]
    )
    labels_padded = torch.stack(
        [torch.nn.functional.pad(x, (0, max_len - x.shape[0]), value=-100) for x in labels]
    )

    # Structure arrays (keep as list — mg_collate handles batching via pyg)
    fracs = [s["frac"] for s in samples]
    cells = [s["cell"] for s in samples]
    Zs = [s["Z"] for s in samples]
    n_atoms = [s["n_atoms"] for s in samples]

    out = {
        "input_ids": input_ids_padded,         # (B, max_len)
        "attention_mask": attn_mask_padded,    # (B, max_len)
        "labels": labels_padded,               # (B, max_len)
        "fracs": fracs,
        "cells": cells,
        "Zs": Zs,
        "n_atoms": n_atoms,
    }
    if "aux_target" in samples[0]:
        out["aux_target"] = torch.stack([s["aux_target"] for s in samples])  # (B, target_dim)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# MatterGen loading
# ─────────────────────────────────────────────────────────────────────────────

def _make_adapter_cfg(pretrained_name: str, full_finetuning: bool,
                      hidden_dim: int = 4096, K: int = 8, mid_dim: int = 2048):
    """Build the OmegaConf DictConfig that init_adapter_lightningmodule_from_pretrained expects."""
    return OmegaConf.create({
        "pretrained_name": pretrained_name,
        "model_path": None,
        "load_epoch": "last",
        "full_finetuning": full_finetuning,
        "adapter": {
            "_target_": "mattergen.adapter.GemNetTAdapter",
            "property_embeddings_adapt": {
                "alm_embedding": {
                    "_target_": "mattergen.property_embeddings.PropertyEmbedding",
                    "name": "alm_embedding",
                    "unconditional_embedding_module": {
                        "_target_": "mattergen.property_embeddings.EmbeddingVector",
                        "hidden_dim": 512,
                    },
                    "conditional_embedding_module": {
                        "_target_": "atoms_mapper.AtomsMapper",
                        "hidden_dim": hidden_dim,
                        "mid_dim": mid_dim,
                        "out_dim": 512,
                        "K": K,
                    },
                    "scaler": {"_target_": "torch.nn.Identity"},
                }
            },
        },
    })


def _make_lightning_module_cfg(lr: float):
    return OmegaConf.create({
        "_target_": "mattergen.diffusion.lightning_module.DiffusionLightningModule",
        "optimizer_partial": {
            "_target_": "torch.optim.AdamW",
            "_partial_": True,
            "lr": lr,
            "weight_decay": 0.0,
            "amsgrad": True,
        },
        "scheduler_partials": [],
    })


def load_mattergen_adapter(pretrained_name: str, lr: float,
                           hidden_dim: int = 4096, K: int = 8, mid_dim: int = 2048):
    """Load MatterGen with AtomsMapper adapter (backbone frozen, adapter layers trainable)."""
    adapter_cfg = _make_adapter_cfg(
        pretrained_name, full_finetuning=False,
        hidden_dim=hidden_dim, K=K, mid_dim=mid_dim,
    )
    lm_cfg = _make_lightning_module_cfg(lr)
    pl_module, _ = init_adapter_lightningmodule_from_pretrained(adapter_cfg, lm_cfg)
    return pl_module


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint
# ─────────────────────────────────────────────────────────────────────────────

def save_checkpoint(step: int, alm, diffusion_module, optimizer, out_dir: Path,
                    include_optimizer: bool = True, aux_head=None, stage_3b: bool = False):
    """Save AtomsMapper, cond_adapt/mixin, optional aux head, and optimizer state.

    `diffusion_module` is the (possibly DDP-wrapped) DiffusionModule from
    `lightning_module.diffusion_module` — same handle the training loop uses.
    `aux_head` is the auxiliary supervision head (or None).
    When `stage_3b=True`, also writes the live LoRA adapter via PEFT's save_pretrained.

    include_optimizer=False skips the optimizer state (~2.5 GB) for a lightweight
    intermediate save. Use True for the final save and any save where you'd
    want to resume from.

    Layout:
      step=N/
        atoms_mapper.pt    — atoms_mapper + trainable_state (+ aux_head + optimizer)
        lora_adapter/      — only in Stage 3b (PEFT adapter dir; resumable via load_alm)
    """
    save_dir = out_dir / f"step={step}"
    save_dir.mkdir(parents=True, exist_ok=True)

    dm = diffusion_module.module if hasattr(diffusion_module, "module") else diffusion_module
    atoms_mapper = (
        dm.model
        .property_embeddings_adapt["alm_embedding"]
        .conditional_embedding_module
    )
    trainable_state = {
        k: v for k, v in dm.model.state_dict().items()
        if "property_embeddings_adapt" in k
        or "cond_adapt_layers" in k
        or "cond_mixin_layers" in k
    }
    payload = {
        "step": step,
        "atoms_mapper_state_dict": atoms_mapper.state_dict(),
        "trainable_state_dict": trainable_state,
    }
    if aux_head is not None:
        payload["aux_head_kind"] = aux_head.target_kind
        payload["aux_head_state_dict"] = aux_head.state_dict()
    if include_optimizer:
        payload["optimizer_state_dict"] = optimizer.state_dict()

    # Stage 3b: write the irreplaceable artifacts FIRST (lora_adapter, projector)
    # then the larger atoms_mapper.pt last. Order matters under disk pressure:
    # if the filesystem fills mid-save, we'd rather lose the (recomputable from
    # LoRA + same init) atoms_mapper checkpoint than the (irreplaceable) LoRA.
    # Run9 hit exactly this pathology: atoms_mapper.pt was written first
    # (1.45 GB w/ optimizer), then lora_adapter mkdir hit ENOSPC.
    #
    # The projector is frozen in Stage 3b, so its state is identical to the
    # Stage 2 source; we write only the projector_state_dict (no optimizer/sched).
    # save_embedding_layers=False keeps token embeds frozen at Stage 2 init.
    # ALM may be DDP-wrapped (Stage 2 pattern: outer-module wrap); unwrap once.
    if stage_3b:
        lora_dir = save_dir / "lora_adapter"
        alm_module = alm.module if hasattr(alm, "module") else alm
        alm_module.llm.save_pretrained(str(lora_dir), save_embedding_layers=False)
        torch.save(
            {"projector_state_dict": alm_module.projector.state_dict()},
            save_dir / "projector_and_state.pt",
        )

    torch.save(payload, save_dir / "atoms_mapper.pt")


def resume_checkpoint(ckpt_path: str, diffusion_module, optimizer, device, aux_head=None):
    """Restore AtomsMapper + cond_adapt layers + optional aux head + optimizer state.

    Note: LoRA adapter is loaded via load_alm(checkpoint=<step=N dir>, is_trainable=True)
    at startup — that resumes the LoRA. This function only handles the diffusion-side state.

    If `aux_head` is provided and the checkpoint contains a matching `aux_head_kind`,
    its state is restored. If the kinds differ (e.g., resume composition→orbv3_mean),
    a warning is printed and the aux head starts from scratch.
    """
    ckpt = torch.load(ckpt_path, map_location=device)
    dm = diffusion_module.module if hasattr(diffusion_module, "module") else diffusion_module

    atoms_mapper = (
        dm.model
        .property_embeddings_adapt["alm_embedding"]
        .conditional_embedding_module
    )
    atoms_mapper.load_state_dict(ckpt["atoms_mapper_state_dict"])

    if "trainable_state_dict" in ckpt:
        cur_sd = dm.model.state_dict()
        cur_sd.update(ckpt["trainable_state_dict"])
        dm.model.load_state_dict(cur_sd, strict=True)

    # Aux head: load if kinds match; otherwise warn and start fresh.
    if aux_head is not None:
        saved_kind = ckpt.get("aux_head_kind")
        if saved_kind == aux_head.target_kind and "aux_head_state_dict" in ckpt:
            aux_head.load_state_dict(ckpt["aux_head_state_dict"])
            print(f"[stage3a] resumed aux_head ({aux_head.target_kind}) state")
        else:
            print(f"[stage3a] aux_head kind mismatch (ckpt={saved_kind!r}, "
                  f"current={aux_head.target_kind!r}) — aux_head starts fresh")

    # Optimizer state is only saved every (save_every * save_optimizer_every) steps
    # to keep intermediate ckpts small. Weights-only resumes start with a fresh
    # optimizer (brief loss spike, quick recovery) — better than losing real steps.
    if "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        print(f"[stage3a] resumed optimizer state from step {ckpt.get('step', 0)}")
    else:
        print(f"[stage3a] no optimizer state in ckpt — starting with fresh optimizer "
              f"(step {ckpt.get('step', 0)} weights restored)")
    return ckpt.get("step", 0)


# ─────────────────────────────────────────────────────────────────────────────
# Build ChemGraph batch from raw structure tensors
# ─────────────────────────────────────────────────────────────────────────────

def build_chemgraph_batch(fracs, cells, Zs, n_atoms, device):
    """Build a batched ChemGraph from lists of per-structure tensors."""
    graphs = []
    for frac, cell, Z, na in zip(fracs, cells, Zs, n_atoms):
        g = ChemGraph(
            pos=frac.to(device, dtype=torch.float32),
            cell=cell.to(device, dtype=torch.float32).unsqueeze(0),  # (1, 3, 3)
            atomic_numbers=Z.to(device, dtype=torch.long),
            num_atoms=na.to(device),
        )
        g = symmetrize_lattice(g)
        g = set_chemical_system_string(g)
        graphs.append(g)
    return mg_collate(graphs)


# ─────────────────────────────────────────────────────────────────────────────
# DDP helpers
# ─────────────────────────────────────────────────────────────────────────────

def _group_grad_norm(params):
    """L2 norm of grads across a parameter group. Skips params with .grad is None."""
    sq = 0.0
    for p in params:
        if p.grad is not None:
            g = p.grad.detach()
            sq += g.float().pow(2).sum().item()
    return sq ** 0.5


def _group_weight_norm(params):
    """L2 norm of weights across a parameter group."""
    sq = 0.0
    for p in params:
        sq += p.detach().float().pow(2).sum().item()
    return sq ** 0.5


def is_main_process():
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0


def setup_ddp():
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        torch.cuda.set_device(local_rank)
        return local_rank
    return 0


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--alm_checkpoint", required=True,
                   help="Stage 2 checkpoint dir (lora_adapter/ + projector_and_state.pt)")
    p.add_argument("--pairs_parquet", required=True,
                   help="Output of build_stage3a_pairs.py")
    p.add_argument("--mattergen_pretrained", default="mattergen_base",
                   help="MatterGen pretrained name (HF hub) or path")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--total_steps", type=int, default=10000)
    p.add_argument("--batch_size", type=int, default=4,
                   help="Per-GPU batch size")
    p.add_argument("--lr", type=float, default=1e-4,
                   help="LR for AtomsMapper + cond_adapt/mixin layers + aux head.")
    p.add_argument("--lora_lr", type=float, default=0.0,
                   help="LR for Stage 2 LoRA params on Qwen3. Default 0 (Stage 3a, frozen LLM). "
                        "Set >0 to enter Stage 3b (unfreeze LoRA on Qwen3). Recommended: 1e-5 "
                        "to 5e-5 — much lower than Stage 2's 2e-4 since the diffusion+aux "
                        "gradient is noisier, and the run3 collapse showed LoRA at high lr "
                        "destroys Qwen3's hidden-state diversity. With aux composition loss "
                        "as anchor, low-lr LoRA can refine [atoms_i] hidden states without "
                        "drifting into the degenerate equilibrium.")
    p.add_argument("--contrastive_lambda", type=float, default=0.0,
                   help="Weight on the off-diag-cosine contrastive loss that decorrelates "
                        "AtomsMapper outputs across the batch. Default 0 (disabled). Note: "
                        "without aux loss, contrastive alone leads to composition-irrelevant "
                        "decorrelation (see run5). With MSE aux (orbv3_mean), absence of "
                        "contrastive can allow rank-collapse to the trivial mean-prediction "
                        "solution (see run7) — set to 0.05–0.1 in that case as a safety net.")
    p.add_argument("--aux_target_kind", type=str, default="composition",
                   choices=["composition", "composition_count", "orbv3_mean", "none"],
                   help="Auxiliary supervision target on AtomsMapper output. 'composition' "
                        "predicts a multi-hot Z=1..100 of the target structure (BCE). "
                        "'composition_count' adds a per-element CE on the exact integer count "
                        "(clamped to MAX_COUNT=20) on top of the same presence BCE — drop-in "
                        "replacement for 'composition' that pressures stoichiometry. "
                        "'orbv3_mean' regresses to the mean OrbV3 per-atom feature (MSE) — "
                        "richer but requires --orbv3_means_path from precompute_orbv3_means.py.")
    p.add_argument("--aux_lambda", type=float, default=1.0,
                   help="Weight on the auxiliary loss in total = L_diff + λ_aux * L_aux + "
                        "λ_contrastive * L_contrastive.")
    p.add_argument("--count_lambda", type=float, default=1.0,
                   help="Weight on the count-CE branch *inside* the composition_count head, "
                        "relative to the presence-BCE branch (= L_aux). 1.0 starts even.")
    p.add_argument("--orbv3_means_path", type=str, default=None,
                   help="Path to orbv3_means.bin (with sibling .idx.json). Required for "
                        "--aux_target_kind=orbv3_mean.")
    p.add_argument("--aux_warmup_steps", type=int, default=0,
                   help="Pre-train AtomsMapper on aux loss only for N steps (zero-out L_diff). "
                        "Useful if mapper output starts uninformative; lets the aux head "
                        "shape AtomsMapper before the noisy diffusion gradient comes online.")
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--save_every", type=int, default=500)
    p.add_argument("--save_optimizer_every", type=int, default=5,
                   help="Include optimizer state every Nth periodic save (else weights-only). "
                        "Final save always includes optimizer.")
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--max_num_tokens", type=int, default=2048)
    p.add_argument("--num_output_atom_tokens", type=int, default=8,
                   help="K — number of [atoms_{i}] output-side tokens emitted at the end "
                        "of the assistant turn. Default 8 (run9 setup). Reduce to 4 for the "
                        "K=4 ablation; increase to 16 to give Qwen3 more capacity for "
                        "spreading structural info across positions before AtomsMapper "
                        "pooling. Each change requires a fresh Stage 3b run from the Stage 2 "
                        "ckpt — load_alm migrates the saved [atoms_i] embedding rows to the "
                        "new K (truncating prefix on shrink, freshly init'ing extras on grow).")
    p.add_argument("--p_unconditional", type=float, default=0.2,
                   help="CFG dropout probability (fraction of samples that use ZerosEmbedding)")
    p.add_argument("--resume_atoms_mapper", default=None,
                   help="Path to atoms_mapper.pt checkpoint to resume from")
    p.add_argument("--disable_wandb", action="store_true")
    p.add_argument("--wandb_project", default="alm-stage3a")
    return p.parse_args()


def main():
    args = parse_args()
    local_rank = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    out_dir = Path(args.out_dir)

    # ── Load ALM (everything frozen — LoRA merged into base) ─────────────────
    # IMPORTANT: Stage 3a does NOT train LoRA. Earlier we tried co-training Stage 2
    # LoRA against the diffusion loss; the result was catastrophic representation
    # collapse — Qwen3 hidden states at [atoms_i] dropped from std=2.5 to 0.44 and
    # became identical across prompts (cosine 1.0). Diffusion loss provides too noisy
    # a gradient to safely co-train a 175M-param LoRA across the entire LLM. AtomsMapper
    # + cond_adapt/mixin (~12M trainable params) is sufficient capacity, and Stage 2
    # LoRA already gives prompt-differentiated [atoms_i] hidden states (cosine 0.96).
    # Stage 3a (default, lora_lr=0): merge_lora=True, all ALM frozen, ALM forward in no_grad.
    # Stage 3b (lora_lr > 0): merge_lora=False with PEFT live, only lora_A/B trainable,
    #   ALM forward NOT in no_grad so gradient reaches LoRA. Aux composition loss anchors
    #   [atoms_i] hidden states to a structurally meaningful direction, preventing the
    #   run3 collapse where diffusion-loss-only LoRA training drove Qwen3's outputs to
    #   zero/uniform. Memory: gradient checkpointing + enable_input_require_grads.
    stage_3b = args.lora_lr > 0
    if is_main_process():
        mode = "Stage 3b — LoRA UNFROZEN" if stage_3b else "Stage 3a — LoRA frozen+merged"
        print(f"[stage3a] Loading ALM from {args.alm_checkpoint} ({mode}) ...")
    alm, tokenizer = load_alm(
        checkpoint=args.alm_checkpoint,
        merge_lora=not stage_3b,
        is_trainable=stage_3b,
        use_cached_embeddings=True,
        device=device,
        num_output_atom_tokens=args.num_output_atom_tokens,
    )
    if stage_3b:
        # Freeze everything except LoRA A/B matrices.
        for name, p in alm.named_parameters():
            p.requires_grad_("lora_A" in name or "lora_B" in name)
        alm.llm.gradient_checkpointing_enable()
        alm.llm.enable_input_require_grads()
    else:
        for p in alm.parameters():
            p.requires_grad_(False)
    alm.eval()
    K = len(alm.output_atom_token_ids)

    # ── Load MatterGen adapter ───────────────────────────────────────────────
    if is_main_process():
        print(f"[stage3a] Loading MatterGen adapter from {args.mattergen_pretrained} ...")
    diffusion_pl = load_mattergen_adapter(
        pretrained_name=args.mattergen_pretrained,
        lr=args.lr,
        hidden_dim=alm.llm_hidden_dim,
        K=K,
        mid_dim=2048,
    )
    diffusion_pl = diffusion_pl.to(device)
    diffusion_module = diffusion_pl.diffusion_module

    # Override the pre_corruption_fn's p_unconditional if needed
    if hasattr(diffusion_module.pre_corruption_fn, "p_unconditional"):
        diffusion_module.pre_corruption_fn.p_unconditional = args.p_unconditional

    # ── Param groups + optimizer ─────────────────────────────────────────────
    mapper_params = [p for p in diffusion_module.parameters() if p.requires_grad]
    atoms_mapper_module = (
        diffusion_module.model
        .property_embeddings_adapt["alm_embedding"]
        .conditional_embedding_module
    )
    atoms_mapper_params = list(atoms_mapper_module.parameters())
    cond_adapt_params = list(
        diffusion_module.model.gemnet.cond_adapt_layers["alm_embedding"].parameters()
    )
    cond_mixin_params = list(
        diffusion_module.model.gemnet.cond_mixin_layers["alm_embedding"].parameters()
    )

    # Aux head (composition / orbv3_mean) — see aux_heads.py.
    aux_head = build_aux_head(args.aux_target_kind, in_dim=512,
                              count_lambda=args.count_lambda)
    aux_head_params: list = []
    if aux_head is not None:
        aux_head = aux_head.to(device)
        aux_head_params = list(aux_head.parameters())
        mapper_params = mapper_params + aux_head_params

    # Stage 3b: LoRA params get their own optimizer group at the lower lora_lr.
    lora_params = []
    if stage_3b:
        lora_params = [p for n, p in alm.named_parameters() if p.requires_grad]

    if is_main_process():
        n_mapper = sum(p.numel() for p in mapper_params)
        n_diff_total = sum(p.numel() for p in diffusion_module.parameters())
        n_am = sum(p.numel() for p in atoms_mapper_params)
        n_ca = sum(p.numel() for p in cond_adapt_params)
        n_cm = sum(p.numel() for p in cond_mixin_params)
        n_aux = sum(p.numel() for p in aux_head_params)
        n_lora = sum(p.numel() for p in lora_params)
        if stage_3b:
            print(f"[stage3a] LoRA: TRAINABLE (Stage 3b, lora_lr={args.lora_lr:.0e})")
            print(f"[stage3a] Trainable LoRA:         {n_lora/1e6:6.1f}M")
        else:
            print(f"[stage3a] LoRA: FROZEN (Stage 3a, merged into base)")
        print(f"[stage3a] Trainable AtomsMapper:  {n_am/1e6:6.1f}M")
        print(f"[stage3a] Trainable cond_adapt:   {n_ca/1e6:6.1f}M")
        print(f"[stage3a] Trainable cond_mixin:   {n_cm/1e6:6.1f}M (zero-init)")
        if aux_head is not None:
            print(f"[stage3a] Trainable aux_head:     {n_aux/1e6:6.3f}M ({aux_head.target_kind}, "
                  f"target_dim={aux_head.target_dim})")
        else:
            print(f"[stage3a] aux_head:               disabled (--aux_target_kind=none)")
        print(f"[stage3a] Trainable total:        {(n_mapper + n_lora)/1e6:6.1f}M")

    if stage_3b:
        optimizer = torch.optim.AdamW(
            [
                {"params": mapper_params, "lr": args.lr,      "name": "mapper"},
                {"params": lora_params,   "lr": args.lora_lr, "name": "lora"},
            ],
            weight_decay=0.0, betas=(0.9, 0.95),
        )
    else:
        optimizer = torch.optim.AdamW(
            mapper_params, lr=args.lr, weight_decay=0.0, betas=(0.9, 0.95),
        )

    # ── Resume ───────────────────────────────────────────────────────────────
    start_step = 0
    if args.resume_atoms_mapper is not None:
        start_step = resume_checkpoint(args.resume_atoms_mapper, diffusion_module,
                                       optimizer, device, aux_head=aux_head)
        if is_main_process():
            print(f"[stage3a] Resumed from step {start_step}")

    # ── DDP wrap ─────────────────────────────────────────────────────────────
    if world_size > 1:
        diffusion_module = DDP(
            diffusion_module,
            device_ids=[local_rank],
            find_unused_parameters=False,  # all mapper params see grad every step
        )
        if stage_3b:
            # Stage 2 pattern (train_stage2.py:299): wrap the OUTER ALM, not
            # alm.llm. Calling `alm(...)` then routes through DDP.forward → ALM.forward
            # so gradient sync hooks fire on backward, and submodule access goes
            # through `alm.module.<attr>` (no per-method DDP unwrap hacks).
            # find_unused_parameters=True matches Stage 2's setting — gradient
            # checkpointing on a frozen base + LoRA topology can leave some adapter
            # paths unused in a given microbatch, so DDP must tolerate that.
            alm = DDP(alm, device_ids=[local_rank], find_unused_parameters=True)

    # ── Dataset / DataLoader ─────────────────────────────────────────────────
    dataset = Stage3aDataset(
        args.pairs_parquet, tokenizer, max_num_tokens=args.max_num_tokens,
        aux_target_kind=args.aux_target_kind,
        orbv3_means_path=args.orbv3_means_path,
        num_output_atom_tokens=args.num_output_atom_tokens,
    )
    sampler = DistributedSampler(dataset, shuffle=True) if world_size > 1 else None
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        collate_fn=stage3a_collate,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )

    # ── W&B ──────────────────────────────────────────────────────────────────
    use_wandb = _WANDB and is_main_process() and not args.disable_wandb
    if use_wandb:
        wandb.init(project=args.wandb_project, config=vars(args))

    # ── Training loop ────────────────────────────────────────────────────────
    global_step = start_step
    diffusion_module.train()

    def _unwrap(m):
        return m.module if hasattr(m, "module") else m

    # Build infinite iterator over loader
    def _inf_loader():
        epoch = 0
        while True:
            if sampler is not None:
                sampler.set_epoch(epoch)
            for batch in loader:
                if batch is not None:
                    yield batch
            epoch += 1

    out_dir.mkdir(parents=True, exist_ok=True)

    for batch in _inf_loader():
        if global_step >= args.total_steps:
            break

        input_ids = batch["input_ids"].to(device)      # (B, max_len)
        attn_mask = batch["attention_mask"].to(device)  # (B, max_len)
        fracs = batch["fracs"]
        cells = batch["cells"]
        Zs = batch["Zs"]
        n_atoms_list = batch["n_atoms"]
        B = input_ids.shape[0]

        # ── ALM forward: text-only ──────────────────────────────────────────
        # Stage 3a: ALM fully frozen + merged → run in no_grad, detach output.
        # Stage 3b: LoRA trainable → no no_grad wrap, no detach. Gradient flows
        # back through Qwen3 (gradient_checkpointing on) into lora_A/B matrices.
        zero_atom_embeds = [torch.zeros(0, 256, device=device) for _ in range(B)]
        input_ids_list = [input_ids[b] for b in range(B)]
        attn_mask_list = [attn_mask[b] for b in range(B)]

        if stage_3b:
            # alm is DDP-wrapped — call it like a module so DDP.forward fires.
            hidden_states = alm(
                input_ids_list, attn_mask_list, labels=None,
                atom_embeds=zero_atom_embeds, output_atoms_hidden_states=True,
            )  # (B, K, hidden_dim) — autograd live, DDP grad-sync hooks armed
            alm_emb = hidden_states.flatten(1).float()  # gradient flows to LoRA
        else:
            # Stage 3a — alm is unwrapped and frozen; cheaper to call the alias.
            with torch.no_grad():
                hidden_states = alm.extract_atoms_hidden_states(
                    input_ids_list, attn_mask_list, atom_embeds=zero_atom_embeds
                )
            alm_emb = hidden_states.flatten(1).float().detach()

        # ── Build ChemGraph and attach alm_embedding ─────────────────────────
        chemgraph = build_chemgraph_batch(fracs, cells, Zs, n_atoms_list, device)
        chemgraph["alm_embedding"] = alm_emb

        # ── Diffusion loss ────────────────────────────────────────────────────
        dm_inner = _unwrap(diffusion_module)
        loss_diff, metrics = dm_inner.calc_loss(chemgraph)

        # Single AtomsMapper forward — used by both contrastive and aux losses.
        # (calc_loss internally runs AtomsMapper again via PropertyEmbedding;
        # forwarding here once more is cheap (~9.4M params) and gives a clean
        # handle for the aux/contrastive heads. Both forwards share parameters
        # so gradients accumulate correctly.)
        if args.contrastive_lambda > 0 or aux_head is not None:
            am_out = atoms_mapper_module(chemgraph["alm_embedding"])  # (B, out_dim)
        else:
            am_out = None

        # ── Contrastive regularization (decorrelate mapper outputs across batch) ──
        if args.contrastive_lambda > 0 and B > 1:
            am_norm = torch.nn.functional.normalize(am_out, dim=-1)
            sim = am_norm @ am_norm.T
            off_diag_mask = ~torch.eye(B, dtype=torch.bool, device=sim.device)
            off_diag = sim[off_diag_mask]
            loss_contrastive = off_diag.pow(2).mean()
            metrics["contrastive_offdiag_mean"] = off_diag.mean().detach()
            metrics["contrastive_offdiag_max"] = off_diag.max().detach()
            metrics["contrastive_loss"] = loss_contrastive.detach()
        else:
            loss_contrastive = torch.tensor(0.0, device=device)

        # ── Auxiliary supervised loss on AtomsMapper output ──────────────────
        # Aux head predicts a structural fingerprint (composition multi-hot or
        # OrbV3 mean) from `am_out`. Loss flows back to AtomsMapper, forcing
        # its output to encode prompt-specific structural info — the issue the
        # contrastive-only run5 didn't fix (linear probe recall = 0.001).
        if aux_head is not None:
            aux_target = batch["aux_target"].to(device)              # (B, target_dim)
            aux_pred = aux_head(am_out)                              # (B, target_dim)
            loss_aux = aux_head.loss(aux_pred, aux_target)
            metrics["aux/loss"] = loss_aux.detach()
            for k, v in aux_head.metrics(aux_pred, aux_target).items():
                metrics[f"aux/{k}"] = v.detach() if torch.is_tensor(v) else v
        else:
            loss_aux = torch.tensor(0.0, device=device)

        # ── Combined loss with optional warmup ────────────────────────────────
        # During aux warmup (global_step < args.aux_warmup_steps), the diffusion
        # loss is zeroed out so AtomsMapper learns the structural fingerprint
        # before the noisy diffusion gradient comes online. cond_adapt/mixin
        # don't get gradient during warmup (they only train via L_diff).
        diff_factor = 1.0
        if args.aux_warmup_steps > 0 and global_step < args.aux_warmup_steps and aux_head is not None:
            diff_factor = 0.0

        loss = (diff_factor * loss_diff
                + args.contrastive_lambda * loss_contrastive
                + args.aux_lambda * loss_aux)

        optimizer.zero_grad()
        loss.backward()

        # Capture per-group pre-clip gradient norms at log cadence.
        will_log = (global_step + 1) % args.log_every == 0 and is_main_process()
        if will_log:
            grad_norms = {
                "atoms_mapper": _group_grad_norm(atoms_mapper_params),
                "cond_adapt":   _group_grad_norm(cond_adapt_params),
                "cond_mixin":   _group_grad_norm(cond_mixin_params),
            }
            if aux_head_params:
                grad_norms["aux_head"] = _group_grad_norm(aux_head_params)
            if stage_3b and lora_params:
                grad_norms["lora"] = _group_grad_norm(lora_params)

        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(mapper_params, args.grad_clip)
            if stage_3b and lora_params:
                torch.nn.utils.clip_grad_norm_(lora_params, args.grad_clip)
        optimizer.step()
        global_step += 1

        # ── Logging ──────────────────────────────────────────────────────────
        if global_step % args.log_every == 0 and is_main_process():
            log = {"loss": loss.item(), "step": global_step}
            log.update({k: v.item() for k, v in metrics.items()})
            log.update({f"grad_norm/{k}": v for k, v in grad_norms.items()})
            log.update({
                "weight_norm/atoms_mapper": _group_weight_norm(atoms_mapper_params),
                "weight_norm/cond_adapt":   _group_weight_norm(cond_adapt_params),
                "weight_norm/cond_mixin":   _group_weight_norm(cond_mixin_params),
            })
            if aux_head_params:
                log["weight_norm/aux_head"] = _group_weight_norm(aux_head_params)
            if stage_3b and lora_params:
                log["weight_norm/lora"] = _group_weight_norm(lora_params)
            cd_mean = metrics.get("contrastive_offdiag_mean")
            cd_max = metrics.get("contrastive_offdiag_max")
            cd_str = (f"  cd_mean={cd_mean.item():+.3f} cd_max={cd_max.item():+.3f}"
                      if cd_mean is not None else "")
            aux_loss = metrics.get("aux/loss")
            aux_str = ""
            if aux_loss is not None:
                # CompositionHead exposes precision/recall; OrbV3MeanHead exposes cosine_sim.
                if "aux/recall" in metrics:
                    aux_str = (f"  aux={aux_loss.item():.3f}"
                               f" P={metrics['aux/precision'].item():.3f}"
                               f" R={metrics['aux/recall'].item():.3f}")
                elif "aux/cosine_sim" in metrics:
                    aux_str = (f"  aux={aux_loss.item():.3f}"
                               f" cos={metrics['aux/cosine_sim'].item():.3f}")
                else:
                    aux_str = f"  aux={aux_loss.item():.3f}"
            lora_str = ""
            if stage_3b and "lora" in grad_norms:
                lora_str = (f" lora={grad_norms['lora']:.2e}"
                            f"  |w_lora|={log['weight_norm/lora']:.2e}")
            print(f"[stage3a] step={global_step}/{args.total_steps}  loss={loss.item():.4f}  "
                  f"|g|: am={grad_norms['atoms_mapper']:.2e} "
                  f"ca={grad_norms['cond_adapt']:.2e} cm={grad_norms['cond_mixin']:.2e}"
                  f"{lora_str}  "
                  f"|w_cm|={log['weight_norm/cond_mixin']:.2e}{cd_str}{aux_str}")
            if use_wandb:
                wandb.log(log, step=global_step)

        # ── Checkpoint ───────────────────────────────────────────────────────
        # Periodic saves are weights-only (lightweight, ~700 MB). Every
        # save_optimizer_every periodic save *also* writes optimizer state
        # for resumability (~3 GB). Final save always includes optimizer.
        if global_step % args.save_every == 0 and is_main_process():
            include_opt = (global_step % (args.save_every * args.save_optimizer_every) == 0)
            save_checkpoint(global_step, alm, diffusion_module, optimizer, out_dir,
                            include_optimizer=include_opt, aux_head=aux_head,
                            stage_3b=stage_3b)
            print(f"[stage3a] Saved checkpoint at step {global_step}"
                  f" {'(with optimizer)' if include_opt else '(weights only)'}")

    # Final save (with optimizer) — but skip if the periodic save just covered
    # this exact step with optimizer too (avoids the redundant 1.5 GB rewrite
    # that hit run9's ENOSPC). The periodic save fires when global_step is a
    # multiple of save_every; it includes optimizer when global_step is also a
    # multiple of save_every * save_optimizer_every. If both, we already have
    # the file we'd be writing here — no-op.
    if is_main_process():
        periodic_covers_final = (
            global_step % args.save_every == 0
            and global_step % (args.save_every * args.save_optimizer_every) == 0
        )
        if periodic_covers_final:
            print(f"[stage3a] Final step={global_step} already saved by the "
                  f"periodic-with-optimizer save; skipping redundant final save.")
        else:
            save_checkpoint(global_step, alm, diffusion_module, optimizer, out_dir,
                            include_optimizer=True, aux_head=aux_head, stage_3b=stage_3b)
        print(f"[stage3a] Training complete. Final step: {global_step}")
        if use_wandb:
            wandb.finish()

    cleanup_ddp()


if __name__ == "__main__":
    main()
