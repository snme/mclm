# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**mclm** (Materials-Conditioned Language Modeling) implements an Atomistic Language Model (ALM) that takes crystal/molecular structures as input and generates natural language descriptions. It follows the LLaVA pattern: frozen domain encoder → trainable MLP projector → frozen LLM, using `inputs_embeds` injection.

This is **Stage 1** of a 5-stage multimodal training pipeline. Only the projector is trained; everything else is frozen.

## Environment

- **Conda env:** `llm` (Python 3.10)
- **Cluster:** MIT ORCD, SLURM partition `pg_tata`
- **Activate:** `conda activate llm`

## Running Training

Training runs via PyTorch DDP on 8× H200 GPUs. Submit via SLURM:

```bash
cd alm/
sbatch submit_train.sh
```

To run directly (single node, for debugging):
```bash
torchrun \
    --nnodes=1 \
    --nproc-per-node=8 \
    --rdzv-backend=c10d \
    --rdzv-endpoint=localhost:29507 \
    train.py \
    --train_csv_path /tmp/train.csv \
    --val_csv_path /tmp/validation.csv \
    --db_path /tmp/oqmd.db \
    --model_save_path /path/to/checkpoint.pt
```

Key `train.py` args: `--batch_size`, `--num_epochs`, `--learning_rate`, `--eval_every`, `--thinking` (enables Qwen3 thinking mode), `--resume_from_checkpoint`, `--checkpoint_save_path` (supports `{epoch}` format string), `--disable_wandb`.

Data is typically copied to `/tmp/` before training (see `submit_train.sh`). Source CSVs and DB live under `/home/sathyae/orcd/`.

## Inference / Evaluation

```bash
cd alm/
python generate-replace-with-my-own.py \
    --checkpoint /path/to/checkpoint.pt \
    --db_path /tmp/oqmd.db \
    --csv_path /tmp/validation.csv \
    --n_samples 10
```

## Embedding Caching

Pre-compute LLM embeddings (Qwen3-Embedding-8B via vLLM, 4× L40S GPUs):
```bash
sbatch submit_embeddings.sh
# or directly:
python cache_embeddings_llm.py --data /path/to/train.csv --emb_output /path/to/embs.pt --ids_output /path/to/ids.txt
```

Pre-compute OrbV3 atomistic embeddings:
```bash
python cache_embeddings_atomistic_orbv3.py \
    --model_name orb_v3_direct_20_omat \
    --data_path /path/to/oqmd.db \
    --dataset_name oqmd
```

## Architecture

### Models

- **LLM**: `Qwen/Qwen3-8B`, fully frozen. Hidden size 4096, 32 layers, GQA (32 query / 8 KV heads), flash-attention-2, bfloat16. Uses ChatML template (`<|im_start|>` / `<|im_end|>`). Thinking mode disabled for Stage 1 (`enable_thinking=False`).
- **Atomistic encoder**: `orb_v3_direct_20_omat` (OrbV3), fully frozen. Produces per-atom node features of dim 256. Variable-length output: one token per atom.
- **Trainable projector** (only trainable component, ~21M params): `Linear(256, 4096) → GELU → Linear(4096, 4096)`. The LLM `hidden_size` is read from `model.config` so swapping LLM backends only requires changing the model name string.

### `<atoms>` token injection

A special `<atoms>` token is added to the tokenizer as a placeholder in the user prompt. During `forward()`, `_merge_embeddings` replaces each `<atoms>` position in the text embedding sequence with the variable-length projected atomistic embeddings, then pads the batch to the max sequence length. The LLM receives only `inputs_embeds` — never raw `input_ids`. This pattern is compatible with any HuggingFace CausalLM.

### Key implementation decisions

1. **`inputs_embeds` injection**: `model.get_input_embeddings()` converts text tokens to embeddings; atomistic embeddings are spliced in at `<atoms>` positions before the LLM forward pass.

2. **DDP, not DeepSpeed**: Stage 1 has only ~21M trainable params (~168MB optimizer state), so plain DDP via `torchrun` is sufficient. DeepSpeed ZeRO becomes relevant in Stage 2+ when the LLM is unfrozen.

3. **Collate function returns lists**: `input_ids`, `labels`, and `attention_mask` are returned as lists of variable-length tensors. Padding happens inside `_merge_embeddings` (after atomistic tokens are spliced in), not in the collate function, to avoid double-padding.

4. **Label masking via double `apply_chat_template`**: Prompt tokens are identified by tokenizing user turn with `add_generation_prompt=True`, full conversation with `add_generation_prompt=False`. Labels are `-100` for all prompt positions; only assistant response tokens are supervised. Atom positions also get label `-100`.

5. **Token budget**: `max_num_tokens` accounts for the expansion from the single `<atoms>` placeholder to per-atom embeddings: `effective_budget = max_num_tokens - n_atoms + 1`.

6. **Per-item device transfer**: Since the collate function returns lists, device transfer is done per-item in the training loop: `[ids.to(device) for ids in batch["input_ids"]]`.

### `alm/alm.py` — `AtomisticLanguageModel`
- `encode_atoms(row_batch)` — ASE `AtomsRow` objects → OrbV3 graphs → node features (256-d) → projected embeddings; returns `(features, n_atoms_tuple)`.
- `forward(row_batch, input_ids, attention_mask, labels)` — encodes atoms, gets text embeddings, stitches via `_merge_embeddings`, forwards through frozen LLM, returns `outputs` (loss at `outputs.loss`).
- `_merge_embeddings(...)` — handles multiple `<atoms>` positions per sample (loops over all positions), pads batch to max sequence length after splicing.

### `alm/utils.py` — `AtomisticLanguageDataset`
- Reads structures from an ASE SQLite DB and descriptions from a Polars CSV.
- `custom_collate_fn` returns `{"atom_rows": list, "input_ids": list[Tensor], "labels": list[Tensor], "attention_mask": list[Tensor]}`.

### `alm/train.py`
- AdamW on `model.module.projector.parameters()` only; lr=1e-3, weight_decay=0, betas=(0.9, 0.999).
- Cosine LR schedule with warmup (3% of total steps, max 2000 warmup steps).
- Gradient clipping at max_norm=1.0.
- LLM and atomistic model kept in `.eval()` throughout; only projector in `.train()`.
- Checkpoints save projector + optimizer + scheduler state. W&B project: `alm-pretrain`.
- Expected loss: rapid drop in first ~300 steps, plateaus around 0.1–0.2.

## Data Format

- **ASE DB** (`oqmd.db`): 1-indexed (`db.get(idx + 1)`); rows have `.toatoms()` → ASE `Atoms` object.
- **CSV**: must have a `description` column; row 0 in CSV aligns with DB entry 1.
- Source data: OQMD (Open Quantum Materials Database). Split files: `train.csv`, `validation.csv`.

## Roadmap (Stages 2–5, not yet implemented)

- **Stage 2**: Unfreeze LLM with LoRA (rank 64–128), instruction-tune on diverse tasks. Switch DDP → DeepSpeed ZeRO-2.
- **Stage 3**: Joint multimodal pre-training with mixed atomistic + text-only data.
- **Stage 4**: Instruction tuning with full fine-tuning or high-rank LoRA.
- **Stage 5**: Reasoning post-training via GRPO; monitor Language Shortcut Ratio to address the "Thinking Over Seeing" problem (text-bias mitigation).
