# mclm
Materials-Conditioned Language Modeling

An Atomistic Language Model (ALM) that takes crystal/molecular structures as input and generates natural language descriptions. Frozen OrbV3 atomistic encoder → trainable MLP projector → frozen Qwen3-8B, using `inputs_embeds` injection at a special `<atoms>` token position.

## Repo structure

```
alm/                        # Stage 1 training pipeline
  alm.py                    # AtomisticLanguageModel — encoder + projector + LLM
  utils.py                  # AtomisticLanguageDataset, collate fn, DDP helpers
  train.py                  # DDP training loop (torchrun, 8× H200)
  generate-replace-with-my-own.py  # Inference / qualitative evaluation
  submit_train.sh           # SLURM job script (pg_tata partition)

cache_embeddings_llm.py     # Pre-compute Qwen3-Embedding-8B embeddings (vLLM, 4× L40S)
cache_embeddings_atomistic_orbv3.py  # Pre-compute OrbV3 node embeddings
submit_embeddings.sh        # SLURM job script for embedding caching

exploration/                # Exploratory experiments (Colab notebooks)
  Atomistic and Language Fusion and Alignment.ipynb
  Contrastive learning and prompting.ipynb
```

## Exploration

**Atomistic and Language Fusion and Alignment** — embeds ~770k OQMD crystal descriptions with Qwen3-Embedding-8B and ~770k structures with OrbV3, then filters to ~95k aligned pairs. Benchmarks early, late, tensor, and LMF fusion on the AV-MNIST audio-image digit classification task for reference, then applies the same fusion approaches to the 256-d atomistic / 4096-d language embedding pairs. Trains a CLIP-style contrastive model on the 95k pairs: loss drops from 7.4 → 0.27 over 20 epochs, and nearest-neighbor retrieval (atomistic query → language embedding) recovers the correct match with similarity 0.95. t-SNE shows clear alignment between the two modalities after contrastive training.

**Contrastive learning and prompting** — renders all OQMD structures as ASE visualizations (~42k images) and fine-tunes Qwen2.5-VL-3B-Instruct on the image→description task using LoRA (1.8M / 3.76B trainable params). Zero-shot baseline generates plausible-sounding but incorrect descriptions (wrong compound identity). Prompt engineering experiments (crystallographer persona, JSON output constraints, few-shot) improve structure but not factual accuracy. After LoRA fine-tuning, the model correctly identifies compound identity and structure type on held-out images.

## Data

Source: OQMD (Open Quantum Materials Database). Structures in an ASE SQLite DB (`oqmd.db`), descriptions in CSVs with an `oqmd_id` column. Splits: `train.csv`, `validation.csv`.

## Setup

```bash
conda activate llm  # Python 3.10, MIT ORCD cluster
cd alm/
sbatch submit_train.sh
```
