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
    def __init__(self, tokenizer, db_path, csv_path, thinking=False, max_num_tokens=1024, dataset_name=None):
        super().__init__()
        self.db = connect(db_path) # ordered the same way as the df. # this is somehow not true anymore? whoops.
        self.tokenizer = tokenizer
        self.thinking = thinking
        self.max_num_tokens = max_num_tokens
        self.dataset_name = dataset_name

        # Only load the columns we actually use: id (first column) and description.
        # Reading the full CSV is multi-GB per dataset and × 8 ranks blows host RAM.
        header_cols = pl.read_csv(csv_path, n_rows=0).columns
        self.id_name = [col for col in header_cols if col.endswith('_id')][0]
        df = pl.read_csv(csv_path, columns=[self.id_name, "description"])
        # Materialize as plain Python lists once; cheaper random access than df[idx][col][0].
        self._ids = df[self.id_name].to_list()
        self._descriptions = df["description"].to_list()
        del df

        # lookup betwen dataset and db id
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
        row = self.db.get(self.dataset_id_to_db_idx[str(sample_id)])
        
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
        prompt_ids = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=True, 
            add_generation_prompt=True,
            enable_thinking=self.thinking,
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
            tokenize=True
        )

        assistant_full_ids = full_ids[len(prompt_ids):]

        # build labels
        input_ids = torch.tensor([full_ids], dtype=torch.long)
        labels = torch.tensor([[-100] * len(prompt_ids) + assistant_full_ids], dtype=torch.long)

        max_num_tokens = self.max_num_tokens - len(row.toatoms()) + 1 # 1 atom per token

        return {
            "input_ids": input_ids[:, :max_num_tokens],
            "labels": labels[:,:max_num_tokens],
            "attention_mask": torch.ones_like(input_ids[:,:max_num_tokens]),
            "atom_rows" : [row], 
            "id" : sample_id
        }


class FullAtomisticLanguageDataset(Dataset):
    def __init__(self, tokenizer, split, parent_folder, thinking=False, max_num_tokens=1024):
        # split should be 'train' or 'validation'
        super().__init__()
        self.parent_folder = Path(parent_folder)
        self.datasets = {}
        self.lengths = {}
        folders = sorted(self.parent_folder.iterdir())
        for folder in folders:
            if folder.is_dir():
                dataset_name = str(folder).split('/')[-1]
                dataset = AtomisticLanguageDataset(
                    tokenizer=tokenizer,
                    db_path=folder / f'{split}.db',
                    csv_path=folder / f'{split}.csv',
                    thinking=thinking,
                    max_num_tokens=max_num_tokens,
                    dataset_name=dataset_name,
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
    return {
        "atom_rows": [b["atom_rows"][0] for b in batch],
        "input_ids": [b["input_ids"].squeeze(0) for b in batch],
        "labels": [b["labels"].squeeze(0) for b in batch],
        "attention_mask": [b["attention_mask"].squeeze(0) for b in batch],
        "id": [b["id"] for b in batch],
    }