# ALM Stage 2 evaluation harness

One script per benchmark. Each accepts `--checkpoint <stage2 step=N/ dir>`,
writes `metrics.json` + `predictions.jsonl` to
`$POOL/eval_results/{benchmark}/{step=N}/`, and (for benchmarks with cited
baselines) is rolled up by `evals/aggregate_results.py` into the paper headline
table.

Set the output root with `ALM_EVAL_RESULTS_ROOT=...` if you want to write
elsewhere; default is `/home/sathyae/orcd/pool/eval_results/`.

## Shared modules
- `loader.py` — `load_alm(checkpoint, …)` (LoRA + projector, optionally merge_and_unload) and `load_base_only()` (Qwen3-8B for the language-retention reference run).
- `generate.py` — batched greedy `inputs_embeds` generation; `atomistic=True/False` switch.
- `parsers.py` — `extract_number`, `extract_choice`. Run `python alm/eval/parsers.py` for the unit-test smoke.
- `metrics.py` — `mae`, `rmse`, `mad_mae_ratio`, `accuracy`, `weighted_f1`.
- `baselines.py` — static cited numbers from each benchmark paper. Update as you
  fill the table.
- `io.py` — writes the per-run output and resolves the run dir.

## Per-benchmark scripts

| Script | Source | Metric |
|---|---|---|
| `eval_llm4mat.py` | LLM4Mat-Bench held-out (9 staged configs; `_DATASET_PROPERTIES` in `alm/utils.py`) | per-config × per-property MAE + MAD:MAE + validity_rate |
| `eval_matterchat.py` | MP test split (LLM4Mat-Bench mp/test as the staged proxy until MatterChat's Zenodo CSV is wired); 5 of 9 tasks today | per-task MAE/RMSE (reg) or accuracy/weighted_f1 (cls) |
| `eval_mattext.py` | HF `n0w0f/MatText` test configs (perovskites, kvrh, gvrh) — live OrbV3 from CIF | MAE per task |
| `eval_gnome_fe.py` | LLM4Mat-Bench `gnome` split, `Formation_Energy_Per_Atom` | MAE + RMSE + MAD:MAE |
| `eval_mat2props.py` | GPT-Narratives parquet (default `mp_3d_2020`); last 10% as held-out unless `--id_list` given | per-property MAE |
| `eval_mat2mcq.py` | Synthesized 4-way element-MCQ from the GPT-Narratives parquet's `atoms` struct (deterministic per `split_seed`); pass `--mcq_jsonl` once Park et al.'s exact MCQs are staged | accuracy |
| `eval_language_retention.py` | HF `cais/mmlu`, `openai/gsm8k`, `Idavidrein/gpqa` (gated; needs HF auth) | accuracy per task |
| `eval_mascqa.py` | `MaScQADataset(split="validation")` (131 stratified-by-topic Qs from the 650-Q benchmark) | mcq_accuracy + numerical_mae |

## One-shot examples

```
# LLM4Mat MP val smoke (5 props × 1000 samples each):
python -m alm.eval.eval_llm4mat \
    --checkpoint /home/sathyae/orcd/pool/alm_checkpoints/stage2_checkpoints/step=12000 \
    --configs mp --split validation --max_samples 1000

# MatText (live OrbV3 path):
python -m alm.eval.eval_mattext \
    --checkpoint .../step=12000 --tasks perovskites,kvrh,gvrh --max_samples 1000

# Language retention — ALM:
python -m alm.eval.eval_language_retention \
    --checkpoint .../step=12000 --task all --max_samples 200

# Language retention — Qwen3-8B base reference:
python -m alm.eval.eval_language_retention --model base --task all --max_samples 200

# MaScQA (held-out 131 Qs):
python -m alm.eval.eval_mascqa --checkpoint .../step=12000

# Aggregate:
python evals/aggregate_results.py --run_id step=12000
```

## Data prep

`helper_scripts/copy_eval_data_to_pool.sh` stages MatterChat (Zenodo), MatText
(HF cache), Park et al.'s repo (git clone into `$POOL/GPT-Narratives-for-Materials/code/`),
and the HF-cached MMLU / GSM8K / GPQA splits. Idempotent. LLM4Mat-Bench and
MaScQA are already on disk and need no staging.

`alex_mp_20` LLM4Mat-Bench config is not staged — eval_llm4mat skips it. Stage
it via `cache_embeddings_atomistic_orbv3.py` if you need the 10th config in the
table.

LLM4Mat-Bench `test` split has CSVs but no `*.db` and no `*_test_atom.flat.bin`
cache — pass `--split validation` until you cache test-split embeddings.
