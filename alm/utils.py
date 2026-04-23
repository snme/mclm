import torch
import torch.distributed as dist
from torch.utils.data import Dataset
from ase.db import connect
import polars as pl
from tqdm import tqdm
from pathlib import Path
import numpy as np
import json
import os

class AtomisticLanguageDataset(Dataset):
    def __init__(self, tokenizer, db_path=None, csv_path=None, thinking=False, max_num_tokens=1024,
                 dataset_name=None, cached_embs_path=None):
        super().__init__()
        self.tokenizer = tokenizer
        self.thinking = thinking
        self.max_num_tokens = max_num_tokens
        self.dataset_name = dataset_name
        self.cached_embs_path = cached_embs_path

        # Only load the columns we actually use: id (first column) and description.
        # Reading the full CSV is multi-GB per dataset and × 8 ranks blows host RAM.
        header_cols = pl.read_csv(csv_path, n_rows=0).columns
        self.id_name = [col for col in header_cols if col.endswith('_id')][0]
        df = pl.read_csv(csv_path, columns=[self.id_name, "description"])
        ids = df[self.id_name].to_list()
        descriptions = df["description"].to_list()
        del df

        if cached_embs_path is not None:
            # Cached-embedding mode: skip DB entirely. The .bin is a flat float32
            # (total_atoms, 256) array; the adjacent .idx.json maps id → [offset, n_atoms].
            # np.memmap gives every DDP rank on a node a shared page cache, so physical
            # RAM use for embeddings is ~one-copy-per-node instead of one-per-rank.
            self.db = None
            self.dataset_id_to_db_idx = None
            bin_path = Path(cached_embs_path)
            idx_path = bin_path.with_suffix(".idx.json")
            with open(idx_path) as f:
                self.cached_index = {k: tuple(v) for k, v in json.load(f).items()}
            self.cached_embs = np.memmap(bin_path, dtype=np.float32, mode="r").reshape(-1, 256)
            before = len(ids)
            kept = [
                (sid, desc) for sid, desc in zip(ids, descriptions)
                if str(sid) in self.cached_index
            ]
            self._ids = [sid for sid, _ in kept]
            self._descriptions = [desc for _, desc in kept]
            if is_main_process() and len(self._ids) != before:
                print(
                    f"[{dataset_name or 'dataset'}] filtered {before - len(self._ids)} / {before} "
                    f"samples with no cached embedding."
                )
        else:
            self._ids = ids
            self._descriptions = descriptions
            self.cached_embs = None
            self.db = connect(db_path)

            # lookup between dataset and db id
            id_index_path = str(db_path).replace(".db", ".id_index.json")
            if os.path.exists(id_index_path):
                with open(id_index_path, 'r') as f:
                    self.dataset_id_to_db_idx = json.load(f)
            else:
                # not recommended, extremely slow
                self.dataset_id_to_db_idx = {}
                for row in tqdm(self.db.select(), total=len(self.db), desc="Building index for dataset"):
                    dataset_id = row.data['smiles']
                    self.dataset_id_to_db_idx[str(dataset_id)] = row.id

    def __len__(self):
        return len(self._ids)

    def __getitem__(self, idx):
        return self.prepare_sample(idx)

    def prepare_sample(self, idx):

        # process single atom
        description = self._descriptions[idx]
        sample_id = self._ids[idx]

        # Cap atoms so the spliced sequence (n_atoms + text) never exceeds max_num_tokens
        # and so text_budget below stays positive. hmof/gnome structures can have 10k+ atoms,
        # which otherwise OOMs the LLM forward and triggers the HF tokenizer "indexing errors"
        # warning when max_num_tokens - n_atoms + 1 goes negative.
        MIN_TEXT_TOKENS = 256
        max_atoms = max(1, self.max_num_tokens - MIN_TEXT_TOKENS)

        if self.cached_embs is not None:
            offset, full_n_atoms = self.cached_index[str(sample_id)]
            n_atoms = min(full_n_atoms, max_atoms)
            # Copy out of the mmap into an owned tensor so downstream pin_memory /
            # worker-process handoff doesn't hold a file-backed view.
            atom_embed = torch.from_numpy(
                np.array(self.cached_embs[offset : offset + n_atoms], dtype=np.float32)
            )
            row = None
        else:
            row = self.db.get(self.dataset_id_to_db_idx[str(sample_id)])
            atom_embed = None
            n_atoms = min(len(row.toatoms()), max_atoms)

        # let's start with a simple prompt.
        messages = [
            {
                "role": "system",
                "content": "You are an expert at materials science and atomistic structure."
            },
            {
                "role": "user",
                "content": "<atoms>\nDescribe the structure of this material."
            }
        ]
        text_budget = self.max_num_tokens - n_atoms + 1
        prompt_ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=self.thinking,
            truncation=True,
            max_length=text_budget,
        )

        full_ids = self.tokenizer.apply_chat_template(
            messages + [
                {
                    "role": "assistant",
                    "content": description
                }
            ],
            add_generation_prompt=False,
            enable_thinking=self.thinking,
            tokenize=True,
            truncation=True,
            max_length=text_budget,
        )

        assistant_full_ids = full_ids[len(prompt_ids):]

        # build labels
        input_ids = torch.tensor([full_ids], dtype=torch.long)
        labels = torch.tensor([[-100] * len(prompt_ids) + assistant_full_ids], dtype=torch.long)

        max_num_tokens = text_budget  # redundant with tokenizer truncation; kept for safety

        sample = {
            "input_ids": input_ids[:, :max_num_tokens],
            "labels": labels[:, :max_num_tokens],
            "attention_mask": torch.ones_like(input_ids[:, :max_num_tokens]),
            "id": sample_id,
        }
        if atom_embed is not None:
            sample["atom_embed"] = atom_embed
        else:
            sample["atom_rows"] = [row]
        return sample


class FullAtomisticLanguageDataset(Dataset):
    def __init__(self, tokenizer, split, parent_folder, thinking=False, max_num_tokens=1024,
                 cached_embs_parent_path=None, atomistic_model_name="orb_v3_direct_20_omat"):
        # split should be 'train' or 'validation'
        super().__init__()
        self.parent_folder = Path(parent_folder)
        self.datasets = {}
        self.lengths = {}
        cached_parent = Path(cached_embs_parent_path) if cached_embs_parent_path else None
        folders = sorted(self.parent_folder.iterdir())
        for folder in folders:
            if not folder.is_dir():
                continue
            dataset_name = folder.name
            cached_bin = None
            if cached_parent is not None:
                candidate = (
                    cached_parent / dataset_name / "embeddings"
                    / f"{atomistic_model_name}_{split}_atom.flat.bin"
                )
                if candidate.exists():
                    cached_bin = candidate
                else:
                    if is_main_process():
                        print(f"[FullAtomisticLanguageDataset] skip {dataset_name}/{split}: no cache at {candidate}")
                    continue
            dataset = AtomisticLanguageDataset(
                tokenizer=tokenizer,
                db_path=folder / f'{split}.db' if cached_bin is None else None,
                csv_path=folder / f'{split}.csv',
                thinking=thinking,
                max_num_tokens=max_num_tokens,
                dataset_name=dataset_name,
                cached_embs_path=cached_bin,
            )
            self.datasets[dataset_name] = dataset
            self.lengths[dataset_name] = len(dataset)
        self.cum_lengths = np.cumsum(list(self.lengths.values()))
                
    def __len__(self):
        return self.cum_lengths[-1]

    def __getitem__(self, idx):
        dataset_ind = np.searchsorted(self.cum_lengths, idx, side="right")
        dataset = self.datasets[list(self.datasets.keys())[dataset_ind]]
        start = 0 if dataset_ind == 0 else self.cum_lengths[dataset_ind - 1].item()
        return dataset[idx - start]


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()

def is_main_process():
    return get_rank() == 0
        
def custom_collate_fn(batch):
    out = {
        "input_ids": [b["input_ids"].squeeze(0) for b in batch],
        "labels": [b["labels"].squeeze(0) for b in batch],
        "attention_mask": [b["attention_mask"].squeeze(0) for b in batch],
        "id": [b["id"] for b in batch],
    }
    if "atom_embed" in batch[0]:
        out["atom_embeds"] = [b["atom_embed"] for b in batch]
    else:
        out["atom_rows"] = [b["atom_rows"][0] for b in batch]
    return out