#!/usr/bin/env python3
"""
Smoke test for ALM training + inference pipeline.

Uses Qwen3-8B and a tiny synthetic ASE database (5 supercells, 64–250 atoms
each) to verify the full forward/backward/inference path without the full
OQMD dataset.

Run from alm/:
    /home/sathya/micromamba/envs/llm/bin/python test_smoke.py
"""

import os
import sys
import tempfile
import traceback

import torch
from ase import Atoms
from ase.build import bulk
from ase.db import connect
import polars as pl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
LLM_PATH = "Qwen/Qwen3-8B"
ORB_MODEL = "orb_v3_direct_20_omat"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 2
N_TRAIN_STEPS = 3
MAX_NEW_TOKENS = 40

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Supercells sized 5–300 atoms each
STRUCTURES = [
    # NaCl rocksalt: 8-atom cubic cell × 2×2×2 = 64 atoms
    (bulk("NaCl", "rocksalt", a=5.64, cubic=True).repeat((2, 2, 2)),
     "Sodium chloride with rock-salt structure, an ionic compound."),
    # Fe BCC: 2-atom cell × 5×5×5 = 250 atoms
    (bulk("Fe", "bcc", a=2.87).repeat((5, 5, 5)),
     "Body-centered cubic iron, a ferromagnetic metal."),
    # MgO rocksalt: 8-atom cubic cell × 2×2×2 = 64 atoms
    (bulk("MgO", "rocksalt", a=4.2, cubic=True).repeat((2, 2, 2)),
     "Magnesium oxide with rock-salt structure, a ceramic oxide."),
    # Cu FCC: 4-atom cubic cell × 3×3×3 = 108 atoms
    (bulk("Cu", "fcc", a=3.61, cubic=True).repeat((3, 3, 3)),
     "Face-centered cubic copper, a ductile transition metal."),
    # Al FCC: 4-atom cubic cell × 3×3×3 = 108 atoms
    (bulk("Al", "fcc", a=4.05, cubic=True).repeat((3, 3, 3)),
     "Face-centered cubic aluminium, a lightweight structural metal."),
]


def make_fake_db_and_csv(tmpdir, n=5):
    db_path  = os.path.join(tmpdir, "test.db")
    csv_path = os.path.join(tmpdir, "test.csv")
    db = connect(db_path)
    oqmd_ids, descriptions = [], []
    for i, (atoms, desc) in enumerate(STRUCTURES[:n]):
        oqmd_id = i + 1
        db.write(atoms, data={"smiles": str(oqmd_id)})
        oqmd_ids.append(oqmd_id)
        descriptions.append(desc)
    pl.DataFrame({"oqmd_id": oqmd_ids, "description": descriptions}).write_csv(csv_path)
    return db_path, csv_path


def section(title):
    print(f"\n{'='*62}")
    print(f"  {title}")
    print('='*62)


def check(label, fn):
    print(f"\n  >> {label} ...", flush=True)
    try:
        result = fn()
        print(f"     PASS")
        return result, True
    except Exception as exc:
        print(f"     FAIL: {exc}")
        traceback.print_exc()
        return None, False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    section("Hardware")
    print(f"  Device : {DEVICE}")
    if DEVICE.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        print(f"  GPU    : {torch.cuda.get_device_name(0)}")
        print(f"  VRAM   : {props.total_memory / 1e9:.1f} GB")
        print(f"  CC     : {props.major}.{props.minor}")
        print(f"  PyTorch: {torch.__version__}")

    # Synthetic data
    tmpdir = tempfile.mkdtemp(prefix="alm_smoke_")
    db_path, csv_path = make_fake_db_and_csv(tmpdir)
    ckpt_path = os.path.join(tmpdir, "projector.pt")
    print(f"\n  Synthetic data → {tmpdir}")

    from alm import AtomisticLanguageModel
    from utils import AtomisticLanguageDataset, custom_collate_fn
    from torch.utils.data import DataLoader

    # ------------------------------------------------------------------
    # 1. Model instantiation — try flash_attn2, fall back to sdpa
    # ------------------------------------------------------------------
    section("1. Model instantiation")
    model = None
    for attn_impl in ("flash_attention_2", "sdpa"):
        print(f"\n  Trying attn_implementation={attn_impl!r} ...", flush=True)
        try:
            model = AtomisticLanguageModel(
                llm_name=LLM_PATH,
                atomistic_model_name=ORB_MODEL,
                device=DEVICE,
                attn_implementation=attn_impl,
            )
            print(f"  PASS  (using {attn_impl!r})")
            break
        except Exception as exc:
            print(f"  FAIL with {attn_impl!r}: {exc}")
            traceback.print_exc()

    if model is None:
        print("\nFatal: cannot instantiate model. Stopping.")
        return

    model = model.to(DEVICE)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"  Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    # ------------------------------------------------------------------
    # 2. Dataset + DataLoader
    # ------------------------------------------------------------------
    section("2. Dataset")
    dataset, ok = check("AtomisticLanguageDataset", lambda: AtomisticLanguageDataset(
        tokenizer=model.tokenizer,
        db_path=db_path,
        csv_path=csv_path,
        thinking=False,
        max_num_tokens=256,
    ))
    if not ok:
        return
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=custom_collate_fn)
    print(f"  Dataset size: {len(dataset)}, batches: {len(loader)}")

    # ------------------------------------------------------------------
    # 3. Single forward pass
    # ------------------------------------------------------------------
    section("3. Forward pass")
    batch_iter = iter(loader)

    def fwd():
        batch = next(batch_iter)
        row_batch    = batch["atom_rows"]
        input_ids    = [x.to(DEVICE) for x in batch["input_ids"]]
        labels       = [x.to(DEVICE) for x in batch["labels"]]
        attn_mask    = [x.to(DEVICE) for x in batch["attention_mask"]]
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            outputs = model(row_batch, input_ids, attn_mask, labels)
        print(f"  Initial loss: {outputs.loss.item():.4f}")
        return outputs

    fwd_out, ok = check("model.forward()", fwd)
    if not ok:
        return

    # ------------------------------------------------------------------
    # 4. Training steps
    # ------------------------------------------------------------------
    section(f"4. Training ({N_TRAIN_STEPS} steps)")

    def train_steps():
        optim = torch.optim.AdamW(model.projector.parameters(), lr=1e-3)
        model.train()
        model.llm.eval()
        model.atomistic_model.eval()
        train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=custom_collate_fn)
        losses = []
        for step, batch in enumerate(train_loader):
            if step >= N_TRAIN_STEPS:
                break
            row_batch = batch["atom_rows"]
            input_ids = [x.to(DEVICE) for x in batch["input_ids"]]
            labels    = [x.to(DEVICE) for x in batch["labels"]]
            attn_mask = [x.to(DEVICE) for x in batch["attention_mask"]]
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                out = model(row_batch, input_ids, attn_mask, labels)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.projector.parameters(), max_norm=1.0)
            optim.step()
            optim.zero_grad(set_to_none=True)
            losses.append(out.loss.item())
            print(f"    step {step+1}/{N_TRAIN_STEPS}  loss={out.loss.item():.4f}")
        return losses

    losses, ok = check(f"{N_TRAIN_STEPS} gradient updates", train_steps)

    # ------------------------------------------------------------------
    # 5. Checkpoint save / load
    # ------------------------------------------------------------------
    section("5. Checkpoint save / load")

    def save_load():
        torch.save({"projector_state_dict": model.projector.state_dict()}, ckpt_path)
        ck = torch.load(ckpt_path, map_location=DEVICE)
        model.projector.load_state_dict(ck["projector_state_dict"])
        print(f"  Saved → {ckpt_path}")

    check("save + reload projector", save_load)

    # ------------------------------------------------------------------
    # 6. Inference
    # ------------------------------------------------------------------
    section("6. Inference (generate_from_row)")

    def inference():
        from generate import generate_from_row
        model.eval()
        db  = connect(db_path)
        row = db.get(1)
        formula = row.toatoms().get_chemical_formula()
        text = generate_from_row(model, row, max_new_tokens=MAX_NEW_TOKENS, temperature=0.6)
        print(f"  Formula : {formula}")
        print(f"  Output  : {text[:300]!r}")
        return text

    check("generate_from_row", inference)

    # ------------------------------------------------------------------
    section("Summary")
    print("  Done. See PASS/FAIL above.")
    print(f"  Temp artefacts: {tmpdir}\n")


if __name__ == "__main__":
    main()
