"""Batched greedy generation on inputs_embeds for ALM eval.

The training collate_fn returns lists of variable-length tensors; we splice atomistic
features into <atoms> positions via the model's _merge_embeddings, then call
llm.generate(inputs_embeds=...) once per batch. Greedy decoding (do_sample=False).

Two paths:
  atomistic=True  (default): batch must contain atom_embeds (cached) or atom_rows (live).
  atomistic=False:           text-only; batch only needs input_ids/labels/attention_mask.

Decode-time leak guard (OFF by default — opt in per-script via
`block_leak_tokens=True` or the eval scripts' `--block_leak_tokens` flag):
Qwen3-8B base falls back to web-style markdown image embeds
(`![](https://i.imgur.com/...png)`) on uncertain prompts because our 14%
arxiv-bucket continued-pretraining partially undoes Qwen3 instruction-tuning.
The guard hard-blocks the markdown-image opener tokens at generate time so
the model cannot start the leak path. We default to OFF so the saved
predictions reflect the model's actual behavior — turn it on for ablation
runs comparing pre/post-leak-fix decoding. Token IDs are computed once per
(model, tokenizer) and cached.
"""
import torch

# Tokens we never want the model to emit — the prefix of a markdown image embed
# and any URL-opener. We pass these as bad_words_ids to generate(); HF blocks
# any of these sequences from appearing. Each entry is a string; we tokenize at
# first call and cache the result on the tokenizer object.
_LEAK_PREFIXES = ["![", "![]", "![](", "https://", "http://"]
_LEAK_CACHE_ATTR = "_alm_eval_bad_words_ids"


def _leak_bad_words_ids(tokenizer):
    cached = getattr(tokenizer, _LEAK_CACHE_ATTR, None)
    if cached is not None:
        return cached
    ids = []
    for s in _LEAK_PREFIXES:
        toks = tokenizer(s, add_special_tokens=False)["input_ids"]
        if toks:
            ids.append(toks)
    setattr(tokenizer, _LEAK_CACHE_ATTR, ids)
    return ids


@torch.no_grad()
def generate_batch(model, batch, max_new_tokens=512, atomistic=True,
                   block_leak_tokens=False):
    """batch is a dict from custom_collate_fn (lists of per-sample tensors).

    Returns: list[str] of generated text per sample (newly generated tokens only).
    """
    device = model.device
    tokenizer = model.tokenizer

    input_ids = [t.squeeze(0).to(device) for t in batch["input_ids"]]
    labels    = [t.squeeze(0).to(device) for t in batch["labels"]]
    attn_mask = [t.squeeze(0).to(device) for t in batch["attention_mask"]]

    # Per-sample prompt = positions where labels == -100. Falls back to the leading
    # 50 tokens for raw-LM samples (arXiv) that have no masked prompt portion.
    prompt_ids_list = []
    for ids, labs in zip(input_ids, labels):
        mask = labs == -100
        if mask.any():
            n_prompt = int(mask.sum().item())
            prompt = ids[:n_prompt]
        else:
            prompt = ids[:min(50, len(ids))]
        prompt_ids_list.append(prompt)

    embed_layer = model.llm.get_input_embeddings()
    text_embeds = [embed_layer(p) for p in prompt_ids_list]
    dummy_labels = [torch.full((p.shape[0],), -100, dtype=torch.long, device=device)
                    for p in prompt_ids_list]
    prompt_attn = [torch.ones(p.shape[0], dtype=torch.long, device=device)
                   for p in prompt_ids_list]

    if atomistic:
        if "atom_embeds" in batch:
            atom_features, n_atoms = model.encode_cached_atoms(batch["atom_embeds"])
        else:
            atom_features, n_atoms = model.encode_atoms(batch["atom_rows"])
        atom_features = torch.split(atom_features, n_atoms)
    else:
        # No <atoms> tokens in prompt → splice path is a no-op; pass empty per-sample
        # tensors with the right shape so _merge_embeddings's zero-atoms branch fires.
        embed_dim = text_embeds[0].shape[-1]
        atom_features = [torch.zeros(0, embed_dim, dtype=text_embeds[0].dtype, device=device)
                         for _ in prompt_ids_list]

    inputs_embeds, _, attention_mask = model._merge_embeddings(
        text_embeds, atom_features, prompt_ids_list, dummy_labels, prompt_attn,
    )

    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_id

    bad_words_ids = _leak_bad_words_ids(tokenizer) if block_leak_tokens else None

    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        out_ids = model.llm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=eos_id,
            pad_token_id=pad_id,
            bad_words_ids=bad_words_ids,
        )
    # When called with inputs_embeds, generate returns ONLY the newly generated
    # tokens (no prompt prefix to strip).
    return [tokenizer.decode(seq.tolist(), skip_special_tokens=True) for seq in out_ids]
