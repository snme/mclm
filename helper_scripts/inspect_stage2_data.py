"""Decode a few samples from a given Stage 2 bucket for eyeball inspection.

Run after staging data with copy_ds_to_tmp_stage2.sh (or just the preprocessor
for text-only buckets). Defaults all point at /tmp.

Examples:
  python helper_scripts/inspect_stage2_data.py --bucket arxiv --n 2
  python helper_scripts/inspect_stage2_data.py --bucket mascqa
  python helper_scripts/inspect_stage2_data.py --bucket describe --n 2
  python helper_scripts/inspect_stage2_data.py --bucket narrate --narrative_name dft_3d
"""
import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "alm"))
from utils import (
    ArxivAbstractDataset, CamelAIDataset, MaScQADataset,
    FullAtomisticLanguageDataset, GPTNarrativeDataset, AtomisticLanguageDataset,
    describe_tasks_for_dataset, property_tasks_for_dataset,
    describe_tasks_for_narrative, applications_tasks_for_narrative,
)


def show(sample, tok, idx, name):
    ids = sample["input_ids"].squeeze(0).tolist()
    labs = sample["labels"].squeeze(0).tolist()
    masked = sum(1 for l in labs if l == -100)
    prompt_ids = [i for i, l in zip(ids, labs) if l == -100]
    target_ids = [i for i, l in zip(ids, labs) if l != -100]
    atom_shape = tuple(sample["atom_embed"].shape) if "atom_embed" in sample else "ROWS"
    print(f"\n{'='*70}\n{name}[{idx}]  id={sample.get('id')}  "
          f"total_tokens={len(ids)}  masked={masked}  atom_embed={atom_shape}\n{'='*70}")
    print("-- PROMPT (masked, not supervised) --")
    print(tok.decode(prompt_ids) if prompt_ids else "(none — raw LM sample)")
    print("-- TARGET (supervised) --")
    print(tok.decode(target_ids))


def build(args, tok):
    b = args.bucket
    if b == "arxiv":
        return ArxivAbstractDataset(tok, args.arxiv_parquet, args.max_num_tokens)
    if b == "camel":
        return CamelAIDataset(tok, args.camel_jsonl, max_num_tokens=args.max_num_tokens)
    if b == "mascqa":
        return MaScQADataset(tok, args.mascqa_json, args.mascqa_xlsx,
                             max_num_tokens=args.max_num_tokens)
    if b in ("describe", "property"):
        tasks_fn = describe_tasks_for_dataset if b == "describe" else property_tasks_for_dataset
        return FullAtomisticLanguageDataset(
            tokenizer=tok, split="train", parent_folder=args.data_parent_path,
            max_num_tokens=args.max_num_tokens,
            cached_embs_parent_path=args.cached_embs_parent_path,
            tasks=tasks_fn,
        )
    if b == "matterchat":
        return AtomisticLanguageDataset(
            tokenizer=tok, db_path=None, csv_path=args.matterchat_csv,
            thinking=False, max_num_tokens=args.max_num_tokens,
            dataset_name="matterchat_mp", cached_embs_path=args.matterchat_cache,
            tasks=property_tasks_for_dataset("matterchat_mp"),
        )
    if b in ("narrate", "explain"):
        name = args.narrative_name
        task_fn = describe_tasks_for_narrative if b == "narrate" else applications_tasks_for_narrative
        cache = Path(f"{args.narrative_cache_dir}/{name}/embeddings/orb_v3_direct_20_omat_atom.flat.bin")
        # Live mode (cache=None) is fine for text-only inspection — decodes atoms struct on demand.
        return GPTNarrativeDataset(
            tokenizer=tok,
            parquet_path=f"{args.narrative_parquet_dir}/{name}_gpt_narratives.parquet",
            cached_embs_path=str(cache) if cache.exists() else None,
            max_num_tokens=args.max_num_tokens, dataset_name=name, tasks=task_fn(name),
        )
    raise ValueError(f"unknown bucket {b}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--bucket", required=True,
                   choices=["arxiv", "camel", "mascqa", "describe", "property",
                            "narrate", "explain", "matterchat"])
    p.add_argument("--narrative_name", default="dft_3d",
                   choices=["dft_3d", "mp_3d_2020", "aflow2", "oqmd"])
    p.add_argument("--llm", default="Qwen/Qwen3-8B")
    p.add_argument("--n", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max_num_tokens", type=int, default=2048)
    p.add_argument("--arxiv_parquet", default="/tmp/jarvis_arxiv.parquet")
    p.add_argument("--camel_jsonl",   default="/tmp/camel_ai.jsonl")
    p.add_argument("--mascqa_json",   default="/tmp/MaScQA/mascqa-eval.json")
    p.add_argument("--mascqa_xlsx",   default="/tmp/MaScQA/scoresheets/all_questions.xlsx")
    p.add_argument("--data_parent_path",        default="/tmp/LLM4Mat-Bench")
    p.add_argument("--cached_embs_parent_path", default="/tmp/cached_embs")
    p.add_argument("--narrative_parquet_dir",   default="/tmp/GPT-Narratives-for-Materials")
    p.add_argument("--narrative_cache_dir",     default="/tmp/cached_embs_narratives")
    p.add_argument("--matterchat_csv",
                   default="/home/sathyae/orcd/pool/eval_data/Dataset_MatterChat/dataset/Material_data_postprocess1_out_correct_train.csv")
    p.add_argument("--matterchat_cache",
                   default="/home/sathyae/orcd/pool/cached_embs/matterchat_mp/embeddings/orb_v3_direct_20_omat_train_atom.flat.bin")
    args = p.parse_args()

    tok = AutoTokenizer.from_pretrained(args.llm)
    tok.add_tokens(["<atoms>"])   # structure buckets embed this token literally in prompts

    ds = build(args, tok)
    print(f"[{args.bucket}] dataset size = {len(ds):,}")
    g = torch.Generator().manual_seed(args.seed)
    for i in torch.randperm(len(ds), generator=g)[:args.n].tolist():
        show(ds[i], tok, i, args.bucket)
