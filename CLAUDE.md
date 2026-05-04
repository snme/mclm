# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**mclm** (Materials-Conditioned Language Modeling) implements an Atomistic Language Model (ALM) that takes crystal/molecular structures as input and produces natural-language descriptions, property predictions, and applications reasoning. It follows the LLaVA pattern: frozen domain encoder → trainable MLP projector → LLM, using `inputs_embeds` injection.

**Current stage: Stage 2** — instruction fine-tuning. The projector is continued from Stage 1, and LoRA adapters are attached to every linear in every Qwen3 decoder block. Base LLM weights and the OrbV3 encoder remain frozen. Stage 1 code (projector-only pretraining) is still in the repo (`alm/train.py`) and used for reproducing / resuming Stage 1 checkpoints.

### Stage 2 training mixture — 5 buckets, weighted by **task type** (not dataset)

| Bucket | Weight | Content |
|---|---|---|
| `describe` | 0.40 | LLM4Mat `DESCRIBE_TASK` + GPT-Narratives `NARRATE_TASK` (captioning: structure → prose) |
| `property_apps` | 0.40 | LLM4Mat property-prediction + GPT-Narratives `EXPLAIN_TASK` + narrative property-prediction (VQA: structure + instruction → property/applications) |
| `arxiv` | 0.14 | JARVIS arXiv abstracts, ChatML instruction-tuning ("title + categories → write the abstract"; assistant turn supervised, prompt masked). Earlier raw-LM framing partially undid Qwen3-8B's instruction-tuning and surfaced base-LM web priors at eval time — moved inside ChatML to keep scientific-text exposure without that fallback. |
| `camel` | 0.04 | CAMEL-AI chem + physics role-play Q&A, ChatML |
| `mascqa` | 0.02 | MaScQA 650-question benchmark, ChatML |

The two 40% buckets draw from the same underlying LLM4Mat-Bench (10 subdatasets) + GPT-Narratives (4 parquets: dft_3d, mp_3d_2020, aflow2, oqmd) rows — the split is over prompt/target shape, not dataset source. This is LLaVA-style (captioning + VQA + text instruction), not a Voxtral-style same-input pattern ablation. The 40/40 split inside the structure buckets is **unvalidated** — treat it as a starting point, plan an ablation (20/60, 60/20).

Training duration is parameterized by **total optimizer steps** (`--total_optim_steps`, default 12000, LLaVA-1.5 convention) — not by epochs sized to the largest bucket. Rank-0 prints a per-bucket coverage table at startup (`visits = total_optim_steps × effective_batch × weight`, and `visits / bucket_size` — watch for ratios ≫ 1 = overfitting, ratios < 1 = under-coverage).

## Environment

- **Conda env:** `llm` (Python 3.10). Activate: `conda activate llm`.
- **Cluster:** MIT ORCD, SLURM partition `pg_tata`, 8× H200 per node.
- Data staged under `/home/sathyae/orcd/pool/` and copied to `/tmp/` before a run.

## Running Training

### Stage 2 — `alm/train_stage2.py`

Stage all data first (LLM4Mat + cached embs + GPT-Narratives + MaScQA + preprocessed arXiv/CAMEL):
```bash
bash helper_scripts/copy_ds_to_tmp_stage2.sh
```

```bash
torchrun --nnodes=1 --nproc-per-node=8 --rdzv-backend=c10d \
    --rdzv-endpoint=$(hostname):29507 train_stage2.py \
    --resume_from_stage1 /home/sathyae/mclm/alm/checkpoint_epoch=1_step=23000.pt \
    --total_optim_steps 12000
```
Defaults point at `/tmp/` staging locations for every dataset.

Resume: `--resume_from_stage2 <save_dir>/step=NNN` (dir with `lora_adapter/` + `projector_and_state.pt`).

Key args: `--lora_rank` (64), `--lora_alpha` (128), `--lora_lr` (2e-4), `--projector_lr` (2e-5), `--batch_size` (4), `--grad_accum_steps` (8 → effective 32/rank, 256 global on 8 GPUs), `--max_num_tokens` (2048), `--total_optim_steps` (12000), `--bucket_weights` (`"0.40,0.40,0.14,0.04,0.02"` — describe, property_apps, arxiv, camel, mascqa), `--eval_every` (500 optim steps), `--val_subset_fraction` (0.01). W&B project: `alm-stage2`.

### Stage 3a / 3b — `alm/train_stage3a.py` (joint training, GILL/DreamLLM pattern)

Single trainer covers both modes; `--lora_lr` selects which:

- **Stage 3a (default, `--lora_lr 0`)** — frozen-LLM mode. ALM is loaded with
  `merge_lora=True`, all params frozen, ALM forward in `torch.no_grad()` and the
  hidden states detached. Only AtomsMapper (~9.4M) and the GemNetTCtrl
  cond_adapt/mixin layers (~4.1M) train. Run3 collapse note: when LoRA was
  trained against diffusion loss alone, Qwen3 hidden states at `[atoms_i]`
  collapsed (cosine 1.0 across prompts, std 2.5 → 0.44). Stage 3a freezes LoRA
  to avoid that.
- **Stage 3b (`--lora_lr > 0`, e.g. `2e-5`)** — LoRA unfrozen. ALM is loaded
  with `merge_lora=False, is_trainable=True`; only `lora_A`/`lora_B` matrices
  are unfrozen, and `gradient_checkpointing_enable()` + `enable_input_require_grads()`
  let gradient flow from L_diff back through Qwen3 into LoRA. Requires the
  composition aux loss (`--aux_target_kind composition --aux_lambda 1.0`) as
  an anchor — it pins `[atoms_i]` hidden states to a structurally meaningful
  direction and prevents the run3 collapse equilibrium. AdamW gets two groups
  (`mapper` at `--lr`, `lora` at `--lora_lr`); both are clipped at `--grad_clip`.
  DDP wraps `alm.llm` separately with `find_unused_parameters=True`.
  `save_checkpoint` writes `lora_adapter/` next to `atoms_mapper.pt` so resume
  re-loads via `--alm_checkpoint <step=N dir>` (load_alm finds `lora_adapter/`).

Frozen in both modes: Qwen3-8B base weights, the Stage 1 projector, OrbV3
encoder, all token embedding rows (including the K=8 `[atoms_i]` rows — LoRA
on q/k/v/o/MLP is the LLM-side adaptation mechanism), and the MatterGen
pretrained backbone.

Gradient path (Stage 3b): `L_diff → cond_adapt/mixin → AtomsMapper → LLM
hidden states at [atoms_i] → gradient-checkpointed Qwen3 → LoRA A/B`. Stage 3a
truncates the path at AtomsMapper (no gradient enters the LLM).

Auxiliary supervision on `am_out` (selected via `--aux_target_kind`):
- `composition` (default) — 100-d multi-hot over Z=1..100, BCE with
  `pos_weight=32` to escape the trivial all-negative minimum. Cheap; computed
  at dataset init from `atoms_struct.elements`. Best frozen-LLM result so far
  (run6b, val recall 0.124).
- `orbv3_mean` — 256-d MSE regression to the mean OrbV3 per-atom feature.
  Requires one-shot precompute via `helper_scripts/precompute_orbv3_means.py`
  → `orbv3_means.bin` + `.idx.json`. Run7 found this collapses to rank-2
  without contrastive (`--contrastive_lambda 0.05` is a safety net there).
- `none` — disables the aux head; relies on diffusion + optional contrastive only.

K=8 output tokens, in_dim = 8 × 4096 = 32768, out_dim = 512 (MatterGen hidden_dim).
ALM forward is in `alm.eval()` mode (deterministic LayerNorm/dropout) but autograd
tracks the graph regardless when `lora_lr > 0`.

The MatterGen fork lives at `external/mattergen/` (submodule of mclm). Our edits are
tracked as `external/mattergen_alm_stage3a.patch` and replayed via
`bash external/setup_mattergen.sh`. For co-install in the `llm` env (torch 2.9+cu128),
run `bash external/mattergen/install_for_h200.sh` then
`bash external/mattergen/build_pyg_for_torch29.sh` on a GPU compute node.

One-time submodule setup (required before training):

```bash
bash external/setup_mattergen.sh   # init submodule + apply patch + chmod +x
# then install mattergen into the llm env (already done; see install_for_h200.sh)
```

End-to-end pipeline:

```bash
# 1. Build (text-prompt, structure) pairs from the four GPT-Narratives parquets,
#    filtered to <=20 atoms (MatterGen Alex-MP-20 distribution).
python helper_scripts/build_stage3a_pairs.py \
    --out_path /home/sathyae/orcd/pool/stage3a/pairs.parquet

# 2a. Stage 3a (frozen-LLM) joint training: ALM frozen → no_grad [atoms_{i}] hidden states
#     → AtomsMapper → MatterGen diffusion loss + composition aux loss. Reads pairs.parquet directly.
#
# IMPORTANT: export the allocator env var (inline `VAR=val python ...` form is fragile
# under multi-line backslash continuation; explicit export is bulletproof). Without
# expandable_segments=True, fragmentation steadily eats GPU memory and triggers OOM
# at unpredictable batches even at modest --batch_size.
export PYTORCH_ALLOC_CONF=expandable_segments:True
PYTHONPATH=/home/sathyae/mclm/alm:$PYTHONPATH \
torchrun --nnodes=1 --nproc-per-node=8 --rdzv-backend=c10d \
    --rdzv-endpoint=$(hostname):29508 alm/train_stage3a.py \
    --alm_checkpoint  /home/sathyae/orcd/pool/alm_checkpoints/stage2_checkpoints/step=12000 \
    --pairs_parquet   /home/sathyae/orcd/pool/stage3a/pairs.parquet \
    --mattergen_pretrained mattergen_base \
    --out_dir         /home/sathyae/orcd/pool/stage3a/ckpts \
    --total_steps 10000 --batch_size 4 --lr 1e-4 \
    --aux_target_kind composition --aux_lambda 1.0
# 2b. Stage 3b adds LoRA training. Needs the aux loss as an anchor; lora_lr is much
#     smaller than Stage 2's 2e-4 (the diffusion+aux gradient is noisier).
PYTHONPATH=/home/sathyae/mclm/alm:$PYTHONPATH \
torchrun --nnodes=1 --nproc-per-node=8 --rdzv-backend=c10d \
    --rdzv-endpoint=$(hostname):29508 alm/train_stage3a.py \
    --alm_checkpoint  /home/sathyae/orcd/pool/alm_checkpoints/stage2_checkpoints/step=12000 \
    --pairs_parquet   /home/sathyae/orcd/pool/stage3a/pairs.parquet \
    --mattergen_pretrained mattergen_base \
    --out_dir         /home/sathyae/orcd/pool/stage3a/ckpts/run8_stage3b \
    --total_steps 5000 --batch_size 8 --max_num_tokens 1536 \
    --lr 3e-4 --lora_lr 2e-5 \
    --aux_target_kind composition --aux_lambda 1.0 --contrastive_lambda 0.0

# 3. Inference: text prompt → ALM forward → AtomsMapper → MatterGen sample.
PYTHONPATH=/home/sathyae/mclm/alm:$PYTHONPATH python helper_scripts/generate_stage3a.py \
    --prompt "Generate a stable cubic perovskite with bandgap > 2 eV." \
    --alm_checkpoint /home/sathyae/orcd/pool/alm_checkpoints/stage2_checkpoints/step=12000 \
    --stage3a_ckpt   /home/sathyae/orcd/pool/stage3a/ckpts/step=10000/atoms_mapper.pt \
    --mattergen_pretrained mattergen_base \
    --out_dir /home/sathyae/orcd/pool/stage3a/generations

# 4. Quantitative metrics eval (CSP head-to-head with CrystaLLM + de-novo SUN +
#    text-conditional composition/density MAE). One-shot orchestrator that runs
#    all three eval scripts (alm/eval/eval_csp.py, eval_dng.py,
#    eval_text_conditional.py), tees output to a single log, and aggregates via
#    evals/aggregate_results.py joining cited baselines (CrystaLLM, MatterGen,
#    OMatG, Crystal-text-LLM) from alm/eval/baselines.py::CRYSTAL_GEN_BASELINES.
#
#    First time: download benchmarks + hull reference (one-shot, idempotent):
#      bash helper_scripts/download_csp_benchmarks.sh   # MP-20, MPTS-52 from CrystaLLM
#      # MatterGen ships the MP2020 hull as a Git-LFS file; fetch its real bytes
#      # before running fetch_mp_hull.py (otherwise the script writes a marker
#      # pointing at an LFS pointer and eval_dng.py errors out cleanly).
#      cd external/mattergen && git lfs install --local && git lfs pull && cd -
#      python helper_scripts/fetch_mp_hull.py           # MatterGen-bundled MP2020 hull
#
#    Then:
bash helper_scripts/eval_stage3b_metrics.sh \
    /home/sathyae/orcd/pool/stage3a/ckpts/run9_stage3b_2node/step=4500 \
    /home/sathyae/orcd/pool/stage3a/eval_metrics/run9_step4500 \
    7 smoke    # ~2-3h on 1 GPU; replace `smoke` with `full` for the publishable run

# Five priority metrics (set in alm/.../structure_metrics.py):
#   1. Match rate + RMSE on MP-20 / MPTS-52 (OrderedStructureMatcher,
#      ltol=0.3 stol=0.5 angle_tol=10 — CDVAE/CrystaLLM defaults).
#   2. Geometric + smact-charge validity.
#   3. % Metastable via MatterSim relaxation + MP convex hull (E_hull < 0.1).
#   4. S.U.N. = Stable & Unique & Novel.
#   5. Density MAE + (optional) formation-energy MAE for text-conditioned gens.
# Targets to claim "beats CrystaLLM-large":
#   MP-20 n=20:  match-rate ≥ 0.74,  RMSE ≤ 0.0349
#   MPTS-52 n=20: match-rate ≥ 0.34, RMSE ≤ 0.106
```

The fork's edits (all in `external/mattergen_alm_stage3a.patch`, replayed by `setup_mattergen.sh`):

- `mattergen/common/utils/globals.py` — append `"alm_embedding"` to `PROPERTY_SOURCE_IDS`.
- `mattergen/conf/.../property_embeddings/alm_embedding.yaml` — wires `atoms_mapper.AtomsMapper`
  as the `conditional_embedding_module`. K=8, in_dim=32768. `unconditional_embedding_module`
  is auto-rewritten to `ZerosEmbedding` by `GemNetTAdapter.__init__`.
- `pyproject.toml` — torch loosened to `>=2.5,<2.10`; pytorch-lightning `>=2.4,<2.6`;
  numpy `>=1.26,<3`; mattersim `>=1.2`. PyG uv sources target cu128.
- `install_for_h200.sh` + `build_pyg_for_torch29.sh` — co-install helpers for the `llm` env.

### Stage 1 — `alm/train.py`

```bash
cd alm/
sbatch submit_train.sh   # 8× H200 DDP on full LLM4Mat-Bench with cached OrbV3 features
```

Direct invocation accepts either `--data_parent_path` (multi-dataset `FullAtomisticLanguageDataset`) or the legacy `--train_csv_path` + `--db_path` + `--val_csv_path` (single-dataset `AtomisticLanguageDataset`). Cached features: pass `--cached_embs_parent_path /tmp/cached_embs`; pass empty string `''` to force live OrbV3 encoding. W&B project: `alm-pretrain`.

Other args: `--batch_size`, `--num_epochs`, `--learning_rate` (1e-3), `--eval_every`, `--thinking`, `--resume_from_checkpoint`, `--checkpoint_save_path` (supports `{epoch}`), `--start_step` (skip first N batches of resume epoch), `--val_subset_fraction`, `--disable_wandb`, `--num_workers`.

## Inference / Evaluation

**Stage 1 quick-look:** `alm/generate.py` (projector-only checkpoint, describe prompts).

**Stage 2 quick-look:** `helper_scripts/generate_stage2.py` (uses `alm.eval.loader.load_alm`; per-bucket sampling, prints prompt + ground-truth + model output).

**Stage 2 paper-eval harness:** `alm/eval/` — one script per benchmark. See `alm/eval/README.md`. Each accepts `--checkpoint <stage2 step=N/ dir>` and writes `metrics.json` + `predictions.jsonl` to `$ALM_EVAL_RESULTS_ROOT/{benchmark}/{step=N}/` (default root: `/home/sathyae/orcd/pool/eval_results/`).

| Script | Benchmark |
|---|---|
| `eval_llm4mat.py` | LLM4Mat-Bench (9 staged configs; per-property MAE + MAD:MAE) |
| `eval_matterchat.py` | MatterChat 9-task MP (5 of 9 today; remaining 4 need their Zenodo CSV) |
| `eval_mattext.py` | MatText perovskites / KVRH / GVRH (live OrbV3 from CIF) |
| `eval_gnome_fe.py` | GNoME formation energy MAE |
| `eval_mat2props.py` | Park et al. Mat2Props on `mp_3d_2020` narrative held-out |
| `eval_mat2mcq.py` | Park et al. Mat2MCQ (synthesized element MCQ; `--mcq_jsonl` for Park's exact set) |
| `eval_language_retention.py` | MMLU + GSM8K + GPQA-Diamond chemistry (run with `--model alm` and `--model base` for the Voxtral-Figure-6 analog) |
| `eval_mascqa.py` | MaScQA 131-Q held-out (stratified, from `MaScQADataset(split="validation")`) |

Aggregate: `python evals/aggregate_results.py --run_id step=N` → `evals/headline_table.{csv,tex}`. Static cited baselines live in `alm/eval/baselines.py` and join the run's metrics in the LaTeX table.

Stage data via `helper_scripts/copy_eval_data_to_pool.sh` (Zenodo download for MatterChat, git clone of Park et al., HF cache for MatText / MMLU / GSM8K / GPQA). LLM4Mat-Bench + MaScQA + GPT-Narratives are already on disk via the Stage 2 staging scripts.

## Embedding Caching

**LLM4Mat-Bench OrbV3 (per-dataset, per-split):**
```bash
python helper_scripts/cache_embeddings_atomistic_orbv3.py \
    --model_name orb_v3_direct_20_omat \
    --data_path /path/to/oqmd.db \
    --dataset_name oqmd
# Then flatten per-row .pt shards into the mmap format the loader uses:
python helper_scripts/flatten_cached_embs.py ...
```

**GPT-Narratives OrbV3** (new for Stage 2; reads parquet directly, no separate ASE DB needed):
```bash
python helper_scripts/cache_embeddings_narratives_orbv3.py \
    --parquet_path $DATA_HOME/GPT-Narratives-for-Materials/oqmd_gpt_narratives.parquet \
    --out_dir      $DATA_HOME/cached_embs_narratives/oqmd/embeddings
```
Writes `{model}_atom.flat.bin` + `{model}_atom.flat.idx.json` keyed by parquet row index (string), plus a companion `atoms.db` + `atoms.id_index.json` so the same structures are accessible through ASE.

**Qwen3 text embeddings (vLLM, 4× L40S)** — used for offline analysis, not training:
```bash
sbatch submit_embeddings.sh
```

## Architecture

### Models

- **LLM**: `Qwen/Qwen3-8B`. Hidden 4096, 32 layers, GQA (32 Q / 8 KV), flash-attention-2, bfloat16. ChatML template. `enable_thinking=False` throughout.
  - Stage 1: fully frozen.
  - Stage 2: base frozen; LoRA adapters (rank 64, α 128, dropout 0.05) on `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`. Gradient checkpointing on; `enable_input_require_grads()` so grads flow through the frozen base.
  - Stage 3a: same as Stage 2 (LoRA + projector inherited via the Stage 2 ckpt) and the Stage 2 LoRA stays trainable. Base weights, projector, and `[atoms_i]` embedding rows are frozen — LoRA on q/k/v/o/MLP is the LLM-side adaptation. Joint training in `alm/train_stage3a.py` (DDP), with the AtomsMapper + cond_adapt/mixin layers in MatterGen's adapter framework.
- **Atomistic encoder**: `orb_v3_direct_20_omat` (OrbV3), 256-d per-atom node features, variable-length output. Frozen in both stages. Instantiation is skipped when `use_cached_embeddings=True` to save ~7 GB GPU RAM + per-step graph build.
- **Projector** (always trainable): `Linear(256 → 4096) → GELU → Linear(4096 → 4096)`, ~21M params. `llm_hidden_dim` read from `self.llm.config.hidden_size`.

### `<atoms>` token injection (input side)

A special `<atoms>` token is added to the tokenizer and placed in the user prompt. `AtomisticLanguageModel._merge_embeddings` splices projected atomistic embeddings (variable length, one token per atom) into every `<atoms>` position of every sample, then pads the batch to max sequence length. The LLM receives only `inputs_embeds`. Works with any HF CausalLM. If a sample contains zero `<atoms>` tokens (e.g. MaScQA), the text embeds pass through unchanged and the (empty or present) atom embedding is ignored — enabling a unified forward path for mixed multimodal + text-only batches.

### `[atoms_{i}]` output tokens (Stage 3a)

K=8 special tokens `[atoms_0]` … `[atoms_7]` are added at construction time alongside `<atoms>`, with deterministic random init (seed 42 around `resize_token_embeddings`) for reproducibility across training runs. They are emitted contiguously at the end of the assistant turn. In Stage 3a, their input embedding rows stay frozen — LoRA on q/k/v/o/MLP adapts how Qwen3 processes any token (including `[atoms_i]`) to produce useful hidden states. Gradient from L_diff flows through AtomsMapper into the LoRA adapters via gradient-checkpointed Qwen3. `extract_atoms_hidden_states` has `@torch.no_grad()` removed for this reason.

### Key implementation decisions

1. **`inputs_embeds` injection** via `model.get_input_embeddings()`; atomistic features spliced in before the LLM forward.
2. **Two atom-encoding paths on `AtomisticLanguageModel`**:
   - `encode_atoms(row_batch)` — live OrbV3 from ASE rows/`Atoms`.
   - `encode_cached_atoms(atom_embeds)` — skip OrbV3; project pre-cached `(N_i, 256)` tensors directly. Picked in `forward()` based on which of `row_batch` / `atom_embeds` is non-None. Cached mode is the default for Stage 2.
3. **Atom count cap (`max_atoms`)** is enforced in both the dataset (`prepare_sample`) and the model (`encode_atoms`) so live and cached paths produce the same number of spliced tokens. Default cap: `max_num_tokens - 256` (reserving ≥256 for prompt + target).
4. **DDP, not DeepSpeed**. Stage 2 is ~2.5% trainable (LoRA ~188M + projector ~21M), DDP is still fine. Pass `find_unused_parameters=True` in Stage 2 — pure-text MaScQA batches have zero-element projector outputs and the check fires otherwise.
5. **Grad accumulation with `model.no_sync()`** on all but the last microbatch of each accumulation window (Stage 2 only).
6. **Collate returns lists** of per-sample variable-length tensors (`input_ids`, `labels`, `attention_mask`). Padding happens inside `_merge_embeddings` after atomistic tokens are spliced in.
7. **Label masking via double `apply_chat_template`**: prompt-only (with `add_generation_prompt=True`) vs. full conversation; `labels[:len(prompt_ids)] = -100`. Atom positions also get `-100`.
8. **Token budget**: `text_budget = max_num_tokens - n_atoms + 1` so the expanded sequence (atoms spliced in for the single `<atoms>` placeholder) still fits in `max_num_tokens`.
9. **Per-item device transfer** in the training loop: `[t.to(device) for t in batch["input_ids"]]` (etc.), because collate returns lists.
10. **Checkpoint format**: Stage 1 saves a dict `{"projector_state_dict", "optimizer_state_dict", "scheduler_state_dict", "epoch", "step", "global_step"}` — loaders also accept a raw projector state_dict for backward compat. Stage 2 saves a directory per optim step: `{save_dir}/step=N/lora_adapter/` (HF `save_pretrained`) + `projector_and_state.pt` (projector + optim + sched + epoch/step).

### `alm/alm.py` — `AtomisticLanguageModel`
- `encode_atoms(row_batch)` — ASE row/`Atoms` list → OrbV3 graphs → 256-d node features → projected. Returns `(features_concat, n_atoms_tuple)`.
- `encode_cached_atoms(atom_embeds)` — list of `(N_i, 256)` tensors → projected. Same return contract.
- `forward(input_ids, attention_mask, labels, row_batch=None, atom_embeds=None)` — picks live vs. cached, stitches via `_merge_embeddings`, forwards LLM. Returns `outputs` (loss at `outputs.loss`).
- `_merge_embeddings(...)` — iterates all `<atoms>` positions per sample, pads batch to max seq length. Handles samples with zero `<atoms>` (pass-through).
- `extract_atoms_hidden_states(input_ids, attention_mask, row_batch=None, atom_embeds=None)` — Stage 3a: returns `(B, K, hidden_dim)` at the K=8 output-side `[atoms_{i}]` positions. `@torch.no_grad()` was removed so gradient flows back through Qwen3 into the LoRA adapters during joint training. For text-only prompts (Stage 3a), pass `atom_embeds=[torch.zeros(0, 256)] * B` to skip the input-side splice.

### `alm/atoms_mapper.py` — `AtomsMapper`
- Two-Linear-with-GELU MLP: `in_dim=K*4096=32768` (K=8) → `hidden=4096` → `out_dim=512` (MatterGen `hidden_dim`). ~70M params. Accepts both `(B, K, 4096)` and `(B, in_dim)` inputs (auto-flatten).
- Imported in the MatterGen fork via `_target_: atoms_mapper.AtomsMapper` in `alm_embedding.yaml`. PYTHONPATH=/home/sathyae/mclm/alm must be set (alm/ is not a package). In Stage 3a joint training, the ALM hidden states are passed live (no detach) as `chemgraph["alm_embedding"]` so gradient from L_diff reaches the Stage 2 LoRA adapters via Qwen3.

### `alm/train_stage3a.py` — Stage 3a joint trainer
- Loads ALM via `load_alm(checkpoint=..., merge_lora=False, is_trainable=True)` so PEFT keeps LoRA live with `requires_grad=True` on the lora_A/B params. Loads MatterGen adapter via `init_adapter_lightningmodule_from_pretrained` with `full_finetuning=False` so only AtomsMapper + cond_adapt/mixin are trainable on the diffusion side.
- Selective freeze: only `lora_A`/`lora_B` rows of `alm` are trainable; Qwen3 base, projector, encoder, and all token embeddings (including the K=8 `[atoms_i]` rows) stay frozen. Gradient checkpointing + `enable_input_require_grads()` lets gradient flow from `inputs_embeds` through frozen base into LoRA.
- Two optimizer param groups: `lora` at `--lora_lr` (default 2e-4) and `mapper` (AtomsMapper + cond_adapt/mixin) at `--lr` (default 1e-4). When `step < --unfreeze_lora_at_step`, the LoRA group's LR is forced to 0 (AtomsMapper warmup phase).
- Per step: tokenize narrative → text-only ALM forward (no `<atoms>`) → ALM hidden states (live, no detach) → build ChemGraph → `diffusion_module.calc_loss` → backward (gradient flows back through both LLM-LoRA and AtomsMapper) → optimizer step.
- `save_checkpoint` writes `step=N/atoms_mapper.pt` (AtomsMapper + cond_adapt/mixin + optimizer state) and `step=N/lora_adapter/` (PEFT `save_pretrained`). `resume_checkpoint` only restores the diffusion-side state; LoRA is restored at startup via `load_alm(checkpoint=<step=N dir>, is_trainable=True)`.
- DDP wraps both `alm.llm` and `diffusion_module` separately, both with `find_unused_parameters=True`. W&B logging, `--p_unconditional 0.2` CFG dropout (handled by MatterGen's `SetEmbeddingType` inside `calc_loss`).

### `alm/utils.py`
- **Task registry**: `DESCRIBE_TASK`, `NARRATE_TASK`, `EXPLAIN_TASK`, `_property_task(prop)`, `_desc_to_property_task(prop)` (eval-only). Every task dict carries a `"bucket"` key (`describe`, `property_apps`, `eval_only`) so Stage 2's 5-bucket sampler can group them.
- **Dataset dispatchers**: `tasks_for_dataset(name)` / `narrative_tasks_for(name)` return the full per-dataset task set (used for eval). Bucketed variants feed the 5-bucket Stage 2 mixer: `describe_tasks_for_dataset`, `property_tasks_for_dataset`, `describe_tasks_for_narrative`, `applications_tasks_for_narrative`. `eval_tasks_description_input(name)` is the paper-parity description→property eval family.
- **`AtomisticLanguageDataset`** — LLM4Mat-Bench format: ASE DB + CSV. Two modes: `cached_embs_path` set → read flat.bin via np.memmap (shared page cache across DDP ranks on the same node); else live OrbV3. Filters requested `tasks` to those whose columns all exist in the CSV. `_pick_task(idx)` per-call, skipping tasks whose target is null for that row.
- **`FullAtomisticLanguageDataset`** — concatenates all subfolder datasets under a `parent_folder`. Accepts `tasks=None | list | callable(dataset_name) -> list`. Skips subdatasets without a matching cache file when `cached_embs_parent_path` is set.
- **`GPTNarrativeDataset`** — parquet-backed equivalent. `atoms` struct column is decoded to ASE on the fly in live mode; cached mode keys by parquet row index (string). `_atoms_struct_to_ase` handles the cartesian/scaled-positions branch.
- **`MaScQADataset`** — text-only Q&A from MaScQA (650 questions). ChatML, `-100` masked prompt. Zero-length `atom_embed` so the shared forward skips the splice.
- **`CamelAIDataset`** — 40k CAMEL-AI chem+physics role-play Q&A from a single JSONL (built by `helper_scripts/preprocess_text_datasets.py`). ChatML, `-100` masked prompt. Zero-length `atom_embed`.
- **`ArxivAbstractDataset`** — 1.8M JARVIS arXiv abstracts from a lean parquet (`id, title, categories, abstract`). **ChatML instruction-tuning** (system: "scientific writing assistant", user: `Title: {title}\nCategories: {cats}\n\nAbstract:`, assistant: the abstract). Prompt masked with `-100`; only the assistant turn (the abstract) is supervised. Zero-length `atom_embed`. Earlier framing was raw continued-pretraining (no ChatML, every token supervised) — that pulled Qwen3-8B back toward base-LM web priors (markdown image embeds, imgur / materialsproject.org URLs at eval time, especially under uncertainty). Switching to instruction-tuning kept the scientific-text exposure but inside the follow-the-user discipline.
- **`custom_collate_fn`** — returns `{input_ids, labels, attention_mask, id}` as lists, plus either `atom_embeds` (list of tensors) or `atom_rows` (list of ASE rows), depending on the first sample's keys. Stage 2 runs all-cached (including zero-length embeds for text-only), so all 5 buckets share the atom_embeds path and mix freely within a batch.

### `alm/train.py` (Stage 1)
- AdamW on projector only. lr=1e-3, wd=0, betas=(0.9, 0.999). Cosine schedule with warmup (3% of total steps, capped 2000). Grad clip max_norm=1.0. Fast-forward of scheduler on resume when scheduler state missing. Checkpoints saved on eval cadence (`--eval_every`) and per-epoch when `--checkpoint_save_path` is set. Expected loss: sharp drop <300 steps, plateau ~0.1–0.2.

### `alm/train_stage2.py` (Stage 2)
- `build_stage2_datasets` builds five bucket `ConcatDataset`s (describe, property_apps, arxiv, camel, mascqa) wrapped in an outer `ConcatDataset`; returns it plus `bucket_offsets` + `bucket_lengths` for the sampler.
- `alm/samplers.py::BucketedDistributedSampler` streams a fixed `num_microbatches = total_optim_steps × grad_accum_steps × world_size` of indices. Per step: rank-local `multinomial(weights)` picks a bucket, then advances a seeded permutation cursor (reshuffles with a bumped seed on exhaustion, so repeat passes aren't in the same order). Each rank gets a deterministic disjoint stream via `(seed, epoch, rank, bucket, cycle)`.
- Validation: every bucket evaluates on held-out rows. `describe` / `property_apps` use the LLM4Mat-Bench val split. `arxiv` / `camel` / `mascqa` datasets each accept `split={"train","validation"}` and partition deterministically on `split_seed=42` — arXiv & CAMEL hold out 500 rows, MaScQA holds out 20% stratified by topic (131/650). Train buckets get `split="train"` implicitly; val instances are built fresh in `train_stage2.py::_text_val_loader`.
- Single-pass training loop: the sampler's `__len__` drives the DataLoader; no outer epoch iteration. Resume fast-forwards via `resume_micro = global_opt_step × grad_accum_steps`.
- Two param groups in AdamW: LoRA at `--lora_lr` (2e-4) and projector at `--projector_lr` (2e-5), weight_decay=0.01, betas=(0.9, 0.95). Cosine warmup (min(2000, 3% of total_optim_steps)) → cosine decay across `total_optim_steps`.
- `save_checkpoint` writes LoRA adapter via `PeftModel.save_pretrained` plus a projector+optim+sched+`global_opt_step` blob. Resume reattaches via `PeftModel.from_pretrained(..., is_trainable=True)`.
- Startup audit: rank-0 prints per-bucket size, weight, expected visits, and visits/size so over/under-coverage is visible before GPU time is spent.

## Data Formats

- **LLM4Mat-Bench ASE DB**: 1-indexed (`db.get(idx + 1)` when no index file). Companion `{db}.id_index.json` maps dataset sample_id → db row id; built on-demand if missing (slow — avoid).
- **LLM4Mat-Bench CSV**: has a `*_id` column plus `description` plus per-dataset property columns (see `_DATASET_PROPERTIES`). Missing property values are null and filtered per-row at task-pick time.
- **GPT-Narratives parquet**: columns include `atoms` (struct: `elements`, `coords`, `lattice_mat`, `cartesian`), `gpt_text`, `gpt_explanation`, and per-parquet property columns (see `_NARRATIVE_PROPERTIES`). Key is parquet row index as string.
- **MaScQA**: `mascqa-eval.json` (topic → `{qids, questions}`) joined with `scoresheets/all_questions.xlsx` (`Question Info`, `Correct Answer`, `Question Type`, `TOPIC`).

## Helper Scripts (`helper_scripts/`)
- `cache_embeddings_atomistic_orbv3.py` — OrbV3 caching for an ASE DB (LLM4Mat-Bench path).
- `cache_embeddings_narratives_orbv3.py` — OrbV3 caching for a GPT-Narratives parquet.
- `cache_embeddings_llm.py` + `submit_embeddings.sh` — Qwen3-Embedding-8B caching via vLLM.
- `flatten_cached_embs.py` — collapses per-row `.pt` shards into the `.flat.bin` + `.flat.idx.json` layout the training loader mmaps.
- `build_id_indices.py` — builds `{db}.id_index.json` for an ASE DB.
- `csv_to_ase.py` / `csv_to_ase.ipynb` — build an ASE DB from a CSV.
- `copy_ds_to_tmp.sh` / `copy_tmp_ds_to_pool.sh` — Stage 1 cluster data staging.
- `copy_ds_to_tmp_stage2.sh` — full Stage 2 data staging (LLM4Mat + cached embs + GPT-Narratives + MaScQA + preprocessed arXiv/CAMEL). Idempotent via `rsync --update`.
- `preprocess_text_datasets.py` — streams JARVIS `arXivdataset.json` (2.8 GB) → lean `jarvis_arxiv.parquet` (`id, title, categories, abstract`); collapses 40k per-file CAMEL-AI JSONs into `camel_ai.jsonl`. Called by `copy_ds_to_tmp_stage2.sh`; mtime-idempotent.
- **Stage 3a scripts**: `build_stage3a_pairs.py` (GPT-Narratives → pairs.parquet, ≤20 atoms filter, K=8 tokens) → `generate_stage3a.py` (inference: text prompt → ALM forward → AtomsMapper → `mattergen-generate`). Training is done directly via `alm/train_stage3a.py` which reads pairs.parquet — no pre-caching of hidden states needed. `prep_stage3a_dataset.py` converts pairs → structure-only numpy arrays for inspection/unconditional eval (not needed for joint training).

## Roadmap
- **Stage 2** *(done)*: LoRA instruction-tune on atomistic + text-only mixture described above. DDP.
- **Stage 3a** *(current)*: K=8 output-side `[atoms_{i}]` tokens + AtomsMapper + GemNetTCtrl cond_adapt/mixin + Stage 2 LoRA all trained against MatterGen's diffusion loss via joint training (`alm/train_stage3a.py`). Qwen3 base + projector + MatterGen backbone frozen; LoRA is the LLM-side adaptation mechanism. Reads pairs.parquet directly; no pre-caching.
- **Stage 3b**: Token-count ablation (4 / 8 / 16) and free placement of `[atoms_i]` in the assistant turn. (LoRA-unfreezing previously slated for 3b moved into 3a.)
- **Stage 4**: Full fine-tune or high-rank LoRA instruction tuning.
- **Stage 5**: Reasoning post-training via GRPO; track Language Shortcut Ratio for "Thinking Over Seeing" / text-bias mitigation.
