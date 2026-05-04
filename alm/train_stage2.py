"""Stage 2: LoRA on Qwen3-8B + continued projector training.

5-bucket task-type mixture (weights over task type, not dataset source):
  describe      0.40  LLM4Mat DESCRIBE_TASK + GPT-Narratives NARRATE_TASK
  property_apps 0.40  LLM4Mat property tasks + GPT-Narratives EXPLAIN + narrative property tasks
  arxiv         0.14  JARVIS arXiv abstracts, raw continued-pretraining
  camel         0.04  CAMEL-AI chem + physics Q&A, ChatML
  mascqa        0.02  MaScQA, ChatML

Training is parameterized by total optimizer steps (~12k, LLaVA-1.5 convention).

Run:
  torchrun --nnodes=1 --nproc-per-node=8 --rdzv-backend=c10d \\
      --rdzv-endpoint=$(hostname):29507 train_stage2.py \\
      --resume_from_stage1 /home/.../stage1_projector.pt \\
      --cached_embs_parent_path /tmp/cached_embs \\
      --narrative_cache_dir $DATA_HOME/cached_embs_narratives \\
      --mascqa_json  $DATA_HOME/MaScQA/mascqa-eval.json \\
      --mascqa_xlsx  $DATA_HOME/MaScQA/scoresheets/all_questions.xlsx \\
      --arxiv_parquet /tmp/jarvis_arxiv.parquet \\
      --camel_jsonl  /tmp/camel_ai.jsonl
"""
import argparse
import os
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import ConcatDataset, DataLoader, Subset, DistributedSampler
from transformers import get_cosine_schedule_with_warmup

from peft import LoraConfig, PeftModel, get_peft_model

from alm import AtomisticLanguageModel
from samplers import BucketedDistributedSampler
from utils import (
    ArxivAbstractDataset,
    AtomisticLanguageDataset,
    CamelAIDataset,
    FullAtomisticLanguageDataset,
    GPTNarrativeDataset,
    MaScQADataset,
    applications_tasks_for_narrative,
    custom_collate_fn,
    describe_tasks_for_dataset,
    describe_tasks_for_narrative,
    is_main_process,
    property_tasks_for_dataset,
    tasks_for_dataset,
)

import wandb


NARRATIVE_PARQUET_NAMES = ["dft_3d", "mp_3d_2020", "aflow2", "oqmd"]
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"]
BUCKET_NAMES = ["describe", "property_apps", "arxiv", "camel", "mascqa"]


def _narrative_subsets(tokenizer, parquet_dir, cache_dir, max_num_tokens, task_fn):
    """Build GPTNarrativeDataset instances for each parquet that has a cache on disk."""
    out = []
    for name in NARRATIVE_PARQUET_NAMES:
        parquet = parquet_dir / f"{name}_gpt_narratives.parquet"
        cache = cache_dir / name / "embeddings" / "orb_v3_direct_20_omat_atom.flat.bin"
        if not parquet.exists() or not cache.exists():
            if is_main_process():
                print(f"[stage2] skip {name}: parquet={parquet.exists()} cache={cache.exists()}")
            continue
        out.append(GPTNarrativeDataset(
            tokenizer=tokenizer, parquet_path=parquet, cached_embs_path=cache,
            thinking=False, max_num_tokens=max_num_tokens,
            dataset_name=name, tasks=task_fn(name),
        ))
    return out


def build_stage2_datasets(args, tokenizer):
    parquet_dir = Path(args.narrative_parquet_dir)
    cache_dir = Path(args.narrative_cache_dir)

    describe_bucket = ConcatDataset([
        FullAtomisticLanguageDataset(
            tokenizer=tokenizer, split="train", parent_folder=args.data_parent_path,
            thinking=False, max_num_tokens=args.max_num_tokens,
            cached_embs_parent_path=args.cached_embs_parent_path,
            tasks=describe_tasks_for_dataset,
        ),
        *_narrative_subsets(tokenizer, parquet_dir, cache_dir, args.max_num_tokens,
                            describe_tasks_for_narrative),
    ])
    property_subsets = [
        FullAtomisticLanguageDataset(
            tokenizer=tokenizer, split="train", parent_folder=args.data_parent_path,
            thinking=False, max_num_tokens=args.max_num_tokens,
            cached_embs_parent_path=args.cached_embs_parent_path,
            tasks=property_tasks_for_dataset,
        ),
        *_narrative_subsets(tokenizer, parquet_dir, cache_dir, args.max_num_tokens,
                            applications_tasks_for_narrative),
    ]
    # MatterChat-style natural-language tasks on the same MP rows as LLM4Mat/mp,
    # different prompt+target shape (matches eval_matterchat.py exactly). Skipped
    # at runtime when its train cache hasn't been built yet.
    mc_csv = Path(args.matterchat_train_csv)
    mc_cache = Path(args.matterchat_train_cache)
    if mc_csv.exists() and mc_cache.exists():
        property_subsets.append(AtomisticLanguageDataset(
            tokenizer=tokenizer, db_path=None, csv_path=str(mc_csv),
            thinking=False, max_num_tokens=args.max_num_tokens,
            dataset_name="matterchat_mp", cached_embs_path=str(mc_cache),
            tasks=property_tasks_for_dataset("matterchat_mp"),
        ))
    elif is_main_process():
        print(f"[stage2] skip matterchat_mp: csv={mc_csv.exists()} cache={mc_cache.exists()}")
    property_bucket = ConcatDataset(property_subsets)
    arxiv_bucket  = ArxivAbstractDataset(tokenizer, args.arxiv_parquet, args.max_num_tokens)
    camel_bucket  = CamelAIDataset(tokenizer, args.camel_jsonl, thinking=False,
                                   max_num_tokens=args.max_num_tokens)
    mascqa_bucket = MaScQADataset(tokenizer, args.mascqa_json, args.mascqa_xlsx,
                                  thinking=False, max_num_tokens=args.max_num_tokens)

    bucket_map = {
        "describe": describe_bucket, "property_apps": property_bucket,
        "arxiv": arxiv_bucket, "camel": camel_bucket, "mascqa": mascqa_bucket,
    }
    buckets = list(bucket_map.values())
    lengths = [len(b) for b in buckets]
    offsets, off = [], 0
    for n in lengths:
        offsets.append(off); off += n
    return ConcatDataset(buckets), offsets, lengths, bucket_map


def _print_coverage(lengths, weights, total_optim_steps, effective_batch):
    total = total_optim_steps * effective_batch
    print(f"[stage2] total_optim_steps={total_optim_steps} effective_batch={effective_batch} "
          f"-> {total:,} total samples seen over the run")
    print(f"{'bucket':<16}{'size':>12}{'weight':>10}{'visits':>14}{'visits/size':>14}")
    for name, n, w in zip(BUCKET_NAMES, lengths, weights):
        v = int(total * w)
        print(f"{name:<16}{n:>12,}{w:>10.3f}{v:>14,}{v/max(1,n):>14.2f}x")


def _log_mem(label, main_process):
    """Log rank-0 process RSS + cgroup usage/limit at EVERY level up the hierarchy."""
    if not main_process:
        return
    try:
        with open(f"/proc/{os.getpid()}/status") as f:
            rss_kb = next(int(l.split()[1]) for l in f if l.startswith("VmRSS:"))
        cg = next(l.split(":")[2].strip() for l in open(f"/proc/{os.getpid()}/cgroup")
                  if "memory" in l or l.split(":")[1] == "")
        print(f"[mem] {label}: rss={rss_kb/1e6:.2f}GB", flush=True)
        # Walk up — print every ancestor's limit so the binding one is visible.
        path = cg
        while path and path != "/":
            for usage_p, limit_p in [
                (f"/sys/fs/cgroup/memory{path}/memory.usage_in_bytes",
                 f"/sys/fs/cgroup/memory{path}/memory.limit_in_bytes"),
                (f"/sys/fs/cgroup{path}/memory.current",
                 f"/sys/fs/cgroup{path}/memory.max"),
            ]:
                if os.path.exists(usage_p) and os.path.exists(limit_p):
                    u = int(open(usage_p).read())
                    raw = open(limit_p).read().strip()
                    lim = "unlimited" if raw == "max" or int(raw) > 1 << 60 else f"{int(raw)/1e9:.2f}GB"
                    print(f"[mem]   {path}: usage={u/1e9:.2f}GB limit={lim}", flush=True)
                    break
            path = os.path.dirname(path)
    except Exception as e:
        print(f"[mem] {label}: error ({e})", flush=True)


class _MultiOpt(torch.optim.Optimizer):
    """Two optimizers stepped in lockstep with a shared LR scheduler. Inherits
    Optimizer so LambdaLR's isinstance check passes; bootstraps with a sentinel
    parameter we never step, then exposes the real inner param_groups so the
    scheduler walks all of them.

    Used only for --optimizer muon (Muon for LoRA 2D matrices, AdamW for
    everything else, per Keller Jordan's recommended recipe)."""
    def __init__(self, optimizers):
        super().__init__([torch.nn.Parameter(torch.zeros(1))], {})
        self.optimizers = optimizers
        self.param_groups = [g for o in optimizers for g in o.param_groups]
    def zero_grad(self, set_to_none=True):
        for o in self.optimizers:
            o.zero_grad(set_to_none=set_to_none)
    def step(self, closure=None):
        for o in self.optimizers:
            o.step()
    def state_dict(self):
        return {"opts": [o.state_dict() for o in self.optimizers]}
    def load_state_dict(self, sd):
        for o, s in zip(self.optimizers, sd["opts"]):
            o.load_state_dict(s)


def _build_optimizer(args, lora_params, projector_params):
    if args.optimizer == "adamw":
        return torch.optim.AdamW(
            [{"params": lora_params,      "lr": args.lora_lr},
             {"params": projector_params, "lr": args.projector_lr}],
            betas=(0.9, 0.95), weight_decay=0.01,
        )
    if args.optimizer == "muon":
        from muon import Muon
        muon_p  = [p for p in lora_params if p.ndim >= 2]
        adam_p  = [p for p in lora_params if p.ndim < 2] + projector_params
        return _MultiOpt([
            Muon(muon_p, lr=args.lora_lr, momentum=0.95),
            torch.optim.AdamW(
                [{"params": adam_p, "lr": args.projector_lr}],
                betas=(0.9, 0.95), weight_decay=0.01,
            ),
        ])
    raise ValueError(f"unknown optimizer: {args.optimizer}")


def train(args):
    # Read torchrun-injected env first so we can pin NCCL to the LOCAL device
    # before init. Without device_id, ProcessGroupNCCL guesses device from
    # GLOBAL rank — wrong on multi-node (e.g. rank 8 on node4301 with 7 GPUs
    # would bind to "GPU 8", which doesn't exist) and silently hangs at the
    # first collective. Reproduces as: NCCL warning "Guessing device ID based
    # on global rank" followed by indefinite freeze at DDP init or first barrier.
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    dist.init_process_group(backend="nccl", init_method="env://", device_id=device)
    main_process = is_main_process()
    _log_mem("startup", main_process)
    use_wandb = main_process and not args.disable_wandb

    weights = [float(x) for x in args.bucket_weights.split(",")]
    assert len(weights) == len(BUCKET_NAMES), f"expected {len(BUCKET_NAMES)} weights, got {weights}"

    model = AtomisticLanguageModel(
        llm_name="Qwen/Qwen3-8B",
        atomistic_model_name="orb_v3_direct_20_omat",
        device=device,
        use_cached_embeddings=True,   # Stage 2 is cached-only across all atomistic buckets
        max_atoms=max(1, args.max_num_tokens - 256),
    )

    # Always wrap fresh; on resume, overwrite LoRA weights from the saved file.
    # (PeftModel.from_pretrained hits a PEFT/transformers version mismatch importing
    # EmbeddingParallel for TP-sharding we don't use, so we bypass it.)
    lora_cfg = LoraConfig(
        r=args.lora_rank, lora_alpha=args.lora_alpha, lora_dropout=0.05,
        bias="none", task_type="CAUSAL_LM", target_modules=LORA_TARGET_MODULES,
    )
    model.llm = get_peft_model(model.llm, lora_cfg)
    if args.resume_from_stage2:
        from safetensors.torch import load_file
        adapter_dir = Path(args.resume_from_stage2) / "lora_adapter"
        sd = load_file(str(adapter_dir / "adapter_model.safetensors"))
        # PEFT's save_pretrained strips the adapter name ("default") from keys; re-insert it.
        sd = {k.replace(".lora_A.weight", ".lora_A.default.weight")
               .replace(".lora_B.weight", ".lora_B.default.weight"): v
              for k, v in sd.items()}
        # Vocab-resize migration: when alm.py grew (e.g. added Stage 3a output tokens
        # after this checkpoint was saved), embed_tokens / lm_head in the adapter are
        # smaller than the current model. Copy the old rows into the matching prefix;
        # new rows keep their fresh resize_token_embeddings init.
        cur_sd = model.llm.state_dict()
        for k in list(sd.keys()):
            if k in cur_sd and sd[k].shape != cur_sd[k].shape:
                old, cur = sd[k], cur_sd[k]
                if old.ndim == cur.ndim and all(o <= c for o, c in zip(old.shape, cur.shape)):
                    new = cur.clone()
                    new[tuple(slice(0, s) for s in old.shape)] = old.to(new.dtype)
                    sd[k] = new
                    if main_process:
                        print(f"  resized {k}: {tuple(old.shape)} → {tuple(new.shape)}")
        _, unexpected = model.llm.load_state_dict(sd, strict=False)
        assert not unexpected, f"unexpected keys after rename: {unexpected[:5]}..."
        if main_process:
            print(f"Loaded Stage 2 LoRA weights from {adapter_dir}")

    # Non-reentrant grad checkpointing: required for training stability under DDP
    # with find_unused_parameters=True + LoRA on a frozen base. Disabling this OOMs
    # on worst-case long-sequence batches even with batch=6 on H200 140GB.
    model.llm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.llm.enable_input_require_grads()

    if args.resume_from_stage1:
        ckpt = torch.load(args.resume_from_stage1, map_location=device)
        proj_state = ckpt.get("projector_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        model.projector.load_state_dict(proj_state)
        if main_process:
            print(f"Loaded Stage 1 projector from {args.resume_from_stage1}")
    elif args.resume_from_stage2:
        state = torch.load(Path(args.resume_from_stage2) / "projector_and_state.pt",
                           map_location=device)
        model.projector.load_state_dict(state["projector_state_dict"])

    model = model.to(device)
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    if main_process:
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"Trainable: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")
    _log_mem("after model+lora+ddp", main_process)

    # Data
    train_dataset, bucket_offsets, bucket_lengths, bucket_map = build_stage2_datasets(args, model.module.tokenizer)
    _log_mem("after train datasets", main_process)
    effective_batch = args.batch_size * args.grad_accum_steps * world_size
    if main_process:
        _print_coverage(bucket_lengths, weights, args.total_optim_steps, effective_batch)

    # Sampler yields per-sample indices. DataLoader groups them into microbatches of
    # batch_size, which the training loop accumulates grad_accum_steps times per
    # optimizer step. So total sample indices needed globally is:
    num_samples_total = (args.total_optim_steps * args.grad_accum_steps
                         * args.batch_size * world_size)
    train_sampler = BucketedDistributedSampler(
        bucket_lengths=bucket_lengths, bucket_offsets=bucket_offsets, weights=weights,
        num_microbatches=num_samples_total, num_replicas=world_size, rank=rank, seed=42,
    )
    train_sampler.set_epoch(0)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, sampler=train_sampler,
        num_workers=args.num_workers, persistent_workers=args.num_workers > 0,
        pin_memory=True, collate_fn=custom_collate_fn,
    )

    # Per-bucket val loaders: separate LLM4Mat-Bench val streams for describe vs property
    # so we can see which capability is degrading. Text-only buckets have no canonical val.
    def _make_val_loader(task_fn):
        if not Path(args.data_parent_path).exists():
            return None
        ds = FullAtomisticLanguageDataset(
            tokenizer=model.module.tokenizer, split="validation",
            parent_folder=args.data_parent_path,
            thinking=False, max_num_tokens=args.max_num_tokens,
            cached_embs_parent_path=args.cached_embs_parent_path,
            tasks=task_fn,
        )
        if args.val_subset_fraction and args.val_subset_fraction < 1.0:
            n = max(1, int(args.val_subset_fraction * len(ds)))
            g = torch.Generator().manual_seed(42)
            ds = Subset(ds, torch.randperm(len(ds), generator=g)[:n].tolist())
        sampler = DistributedSampler(ds, shuffle=True, drop_last=True, seed=42)
        return DataLoader(
            ds, batch_size=args.batch_size, sampler=sampler,
            num_workers=args.num_workers, persistent_workers=args.num_workers > 0,
            pin_memory=True, collate_fn=custom_collate_fn,
        )

    # Text-only val: each dataset class deterministically holds out a slice
    # (split_seed=42) — train buckets get split="train" implicitly, here we build
    # split="validation" instances. Held-out rows, never seen during training.
    def _text_val_loader(name):
        tok = model.module.tokenizer
        if name == "arxiv":
            ds = ArxivAbstractDataset(tok, args.arxiv_parquet, args.max_num_tokens,
                                      split="validation")
        elif name == "camel":
            ds = CamelAIDataset(tok, args.camel_jsonl, thinking=False,
                                max_num_tokens=args.max_num_tokens, split="validation")
        elif name == "mascqa":
            ds = MaScQADataset(tok, args.mascqa_json, args.mascqa_xlsx,
                               thinking=False, max_num_tokens=args.max_num_tokens,
                               split="validation")
        else:
            raise ValueError(name)
        sampler = DistributedSampler(ds, shuffle=False, drop_last=True, seed=42)
        return DataLoader(
            ds, batch_size=args.batch_size, sampler=sampler,
            num_workers=args.num_workers, persistent_workers=args.num_workers > 0,
            pin_memory=True, collate_fn=custom_collate_fn,
        )

    # Separate MatterChat val: LLM4Mat val uses JSON property prompts, MatterChat
    # uses natural-language A/B/C-letter prompts — different distributions, different
    # floors. Tracking them separately disambiguates "LLM4Mat val rising = LoRA
    # capacity drifting toward MatterChat format" from real property-prediction regression.
    def _matterchat_val_loader():
        mc_csv = Path(args.matterchat_val_csv)
        mc_cache = Path(args.matterchat_val_cache)
        if not (mc_csv.exists() and mc_cache.exists()):
            if is_main_process():
                print(f"[stage2] skip matterchat val: csv={mc_csv.exists()} cache={mc_cache.exists()}")
            return None
        ds = AtomisticLanguageDataset(
            tokenizer=model.module.tokenizer, db_path=None, csv_path=str(mc_csv),
            thinking=False, max_num_tokens=args.max_num_tokens,
            dataset_name="matterchat_mp", cached_embs_path=str(mc_cache),
            tasks=property_tasks_for_dataset("matterchat_mp"),
        )
        if args.val_subset_fraction and args.val_subset_fraction < 1.0:
            n = max(1, int(args.val_subset_fraction * len(ds)))
            g = torch.Generator().manual_seed(42)
            ds = Subset(ds, torch.randperm(len(ds), generator=g)[:n].tolist())
        sampler = DistributedSampler(ds, shuffle=True, drop_last=True, seed=42)
        return DataLoader(
            ds, batch_size=args.batch_size, sampler=sampler,
            num_workers=args.num_workers, persistent_workers=args.num_workers > 0,
            pin_memory=True, collate_fn=custom_collate_fn,
        )

    val_loaders = {
        "describe":      _make_val_loader(describe_tasks_for_dataset),
        "property_apps": _make_val_loader(property_tasks_for_dataset),
        "matterchat":    _matterchat_val_loader(),
        "arxiv":  _text_val_loader("arxiv"),
        "camel":  _text_val_loader("camel"),
        "mascqa": _text_val_loader("mascqa"),
    }
    val_loaders = {k: v for k, v in val_loaders.items() if v is not None}
    _log_mem("after val loaders", main_process)

    # Optim + scheduler driven by total_optim_steps.
    lora_params, projector_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "projector" in name:
            projector_params.append(p)
        else:
            lora_params.append(p)
    optim = _build_optimizer(args, lora_params, projector_params)
    warmup_steps = min(2000, int(0.03 * args.total_optim_steps))
    scheduler = get_cosine_schedule_with_warmup(
        optim, num_warmup_steps=warmup_steps, num_training_steps=args.total_optim_steps,
    )

    global_opt_step = 0
    if args.resume_from_stage2:
        state = torch.load(Path(args.resume_from_stage2) / "projector_and_state.pt",
                           map_location=device)
        if "optimizer_state_dict" in state:
            optim.load_state_dict(state["optimizer_state_dict"])
        if "scheduler_state_dict" in state:
            scheduler.load_state_dict(state["scheduler_state_dict"])
        global_opt_step = int(state.get("global_opt_step", 0))
        dist.barrier()
    _log_mem("after optim+resume", main_process)

    if use_wandb:
        # Resume an existing wandb run by ID so curves stay on one plot across
        # node-count / batch-size changes. `step=global_opt_step` is passed on
        # every log call below so the x-axis stays consistent regardless of how
        # many wandb auto-increments happened in the prior run.
        init_kwargs = {"project": args.wandb_project, "config": vars(args)}
        if args.wandb_run_id:
            init_kwargs["id"] = args.wandb_run_id
            init_kwargs["resume"] = "allow"
        wandb.init(**init_kwargs)

    save_root = Path(args.save_dir)
    save_root.mkdir(parents=True, exist_ok=True)

    # Single-pass training loop: the sampler is sized to deliver exactly
    # total_optim_steps × grad_accum_steps microbatches per rank.
    model.train()
    model.module.llm.train()
    if model.module.atomistic_model is not None:
        model.module.atomistic_model.eval()
    optim.zero_grad(set_to_none=True)

    micro_i = 0
    resume_micro = global_opt_step * args.grad_accum_steps
    for batch in train_loader:
        if micro_i < resume_micro:
            micro_i += 1
            continue
        row_batch = batch.get("atom_rows")
        atom_embeds = batch.get("atom_embeds")
        input_ids = [t.to(device) for t in batch["input_ids"]]
        labels = [t.to(device) for t in batch["labels"]]
        attention_mask = [t.to(device) for t in batch["attention_mask"]]

        is_accum = ((micro_i + 1) % args.grad_accum_steps) != 0
        sync_ctx = model.no_sync() if is_accum else nullcontext()
        with sync_ctx, torch.cuda.amp.autocast(dtype=torch.bfloat16):
            outputs = model(input_ids, attention_mask, labels,
                            row_batch=row_batch, atom_embeds=atom_embeds)
            loss = outputs.loss / args.grad_accum_steps
        loss.backward()

        micro_i += 1
        if is_accum:
            continue

        total_norm = torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], max_norm=1.0,
        )
        # All-reduce a "all ranks finite" flag so DDP stays in sync on skip decisions.
        # One bf16 outlier batch otherwise NaNs grads → poisons AdamW state → entire run dies.
        is_finite = torch.isfinite(total_norm).to(torch.float32)
        dist.all_reduce(is_finite, op=dist.ReduceOp.MIN)
        if is_finite.item() == 1.0:
            optim.step()
            scheduler.step()
        elif main_process:
            print(f"[skip] nan/inf grads at opt_step {global_opt_step + 1}; "
                  f"local total_norm={total_norm.item():.3e}")
        optim.zero_grad(set_to_none=True)
        global_opt_step += 1

        if main_process and global_opt_step % 10 == 0:
            print(f"opt_step {global_opt_step}/{args.total_optim_steps} "
                  f"loss={loss.item() * args.grad_accum_steps:.4f} "
                  f"lr_lora={optim.param_groups[0]['lr']:.2e} "
                  f"lr_proj={optim.param_groups[1]['lr']:.2e}")
        if use_wandb and global_opt_step % args.log_every == 0:
            # step=global_opt_step pins wandb's cursor to opt_step. Every log call
            # in train_stage2.py uses this convention so wandb_step == global_opt_step
            # by construction across resumes / node-count changes. Required for
            # resumed runs to plot cleanly without the step-axis scale-mixing bug.
            wandb.log({
                "train/loss": loss.item() * args.grad_accum_steps,
                "train/lr_lora": optim.param_groups[0]["lr"],
                "train/lr_projector": optim.param_groups[1]["lr"],
                "global_opt_step": global_opt_step,
            }, step=global_opt_step)

        if val_loaders and global_opt_step % args.eval_every == 0:
            run_validation(model, val_loaders, device, global_opt_step, main_process, use_wandb)
            save_checkpoint(model, optim, scheduler, global_opt_step, save_root, main_process)

        if global_opt_step >= args.total_optim_steps:
            break

    save_checkpoint(model, optim, scheduler, global_opt_step, save_root, main_process)
    if use_wandb:
        wandb.finish()
    dist.destroy_process_group()


def run_validation(model, val_loaders, device, global_opt_step, main_process, use_wandb):
    model.eval()
    per_bucket = {}
    for name, loader in val_loaders.items():
        total_loss, n_batches = 0.0, 0
        for batch in loader:
            row_batch = batch.get("atom_rows")
            atom_embeds = batch.get("atom_embeds")
            input_ids = [t.to(device) for t in batch["input_ids"]]
            labels = [t.to(device) for t in batch["labels"]]
            attention_mask = [t.to(device) for t in batch["attention_mask"]]
            with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
                out = model(input_ids, attention_mask, labels,
                            row_batch=row_batch, atom_embeds=atom_embeds)
                total_loss += out.loss.item()
                n_batches += 1
        per_bucket[name] = total_loss / max(1, n_batches)
    model.train()
    model.module.llm.train()
    if model.module.atomistic_model is not None:
        model.module.atomistic_model.eval()
    if main_process:
        for name, v in per_bucket.items():
            print(f"val_loss/{name} (opt_step {global_opt_step}): {v:.4f}")
    if use_wandb:
        log = {f"val/loss_{n}": v for n, v in per_bucket.items()}
        log["val/loss"] = sum(per_bucket.values()) / max(1, len(per_bucket))
        log["global_opt_step"] = global_opt_step
        wandb.log(log, step=global_opt_step)


def save_checkpoint(model, optim, scheduler, global_opt_step, save_root, main_process):
    if not main_process:
        return
    ckpt_dir = save_root / f"step={global_opt_step}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    model.module.llm.save_pretrained(ckpt_dir / "lora_adapter")
    torch.save(
        {
            "projector_state_dict": model.module.projector.state_dict(),
            "optimizer_state_dict": optim.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "global_opt_step": global_opt_step,
        },
        ckpt_dir / "projector_and_state.pt",
    )
    print(f"Saved Stage 2 checkpoint → {ckpt_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    # Data
    p.add_argument("--data_parent_path", type=str, default="/tmp/LLM4Mat-Bench")
    p.add_argument("--cached_embs_parent_path", type=str, default="/tmp/cached_embs")
    p.add_argument("--narrative_parquet_dir", type=str, default="/tmp/GPT-Narratives-for-Materials")
    p.add_argument("--narrative_cache_dir", type=str, default="/tmp/cached_embs_narratives")
    p.add_argument("--matterchat_train_csv", type=str,
                   default="/home/sathyae/orcd/pool/eval_data/Dataset_MatterChat/dataset/Material_data_postprocess1_out_correct_train.csv",
                   help="MatterChat MP train CSV (128k rows). Skipped if missing.")
    p.add_argument("--matterchat_train_cache", type=str,
                   default="/home/sathyae/orcd/pool/cached_embs/matterchat_mp/embeddings/orb_v3_direct_20_omat_train_atom.flat.bin",
                   help="OrbV3 cache for matterchat MP train. Build via cache_embeddings_atomistic_orbv3.py.")
    p.add_argument("--matterchat_val_csv", type=str,
                   default="/home/sathyae/orcd/pool/eval_data/Dataset_MatterChat/dataset/Material_data_postprocess1_out_correct_val.csv",
                   help="MatterChat MP val CSV (~14k rows). Held-out, used only for val/loss_matterchat.")
    p.add_argument("--matterchat_val_cache", type=str,
                   default="/home/sathyae/orcd/pool/cached_embs/matterchat_mp/embeddings/orb_v3_direct_20_omat_validation_atom.flat.bin",
                   help="OrbV3 cache for matterchat MP val.")
    p.add_argument("--mascqa_json", type=str, default="/tmp/MaScQA/mascqa-eval.json")
    p.add_argument("--mascqa_xlsx", type=str, default="/tmp/MaScQA/scoresheets/all_questions.xlsx")
    p.add_argument("--arxiv_parquet", type=str, default="/tmp/jarvis_arxiv.parquet")
    p.add_argument("--camel_jsonl",   type=str, default="/tmp/camel_ai.jsonl")
    p.add_argument("--max_num_tokens", type=int, default=2048)
    # Mixture
    p.add_argument("--bucket_weights", type=str, default="0.408,0.408,0.14,0.04,0",
                   help="order: describe, property_apps, arxiv, camel, mascqa (0 = skip)")
    # Resume
    p.add_argument("--resume_from_stage1", type=str, default=None)
    p.add_argument("--resume_from_stage2", type=str, default=None)
    # LoRA
    p.add_argument("--lora_rank", type=int, default=64)
    p.add_argument("--lora_alpha", type=int, default=128)
    # Optim
    p.add_argument("--optimizer", choices=["adamw", "muon"], default="adamw",
                   help="muon: Newton-Schulz orthogonalized SGD on LoRA 2D mats, AdamW for the rest")
    p.add_argument("--lora_lr", type=float, default=2e-4)
    p.add_argument("--projector_lr", type=float, default=2e-5)
    p.add_argument("--batch_size", type=int, default=12)
    p.add_argument("--grad_accum_steps", type=int, default=4)
    p.add_argument("--total_optim_steps", type=int, default=12000,
                   help="LLaVA-1.5 style total optimizer-step budget (~1-3 logical passes of the mixture).")
    # Eval / logging
    p.add_argument("--eval_every", type=int, default=500)
    p.add_argument("--val_subset_fraction", type=float, default=0.01)
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--disable_wandb", action="store_true")
    p.add_argument("--wandb_project", type=str, default="alm-stage2")
    p.add_argument("--wandb_run_id", type=str, default=None,
                   help="Resume a specific wandb run by id (e.g. 'vc5cfy32' to keep "
                        "curves on one plot when changing node count or batch size).")
    # IO
    p.add_argument("--save_dir", type=str, default="/home/sathyae/orcd/pool/alm_checkpoints/stage2_checkpoints")
    p.add_argument("--num_workers", type=int, default=4)
    args = p.parse_args()
    train(args)
