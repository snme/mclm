import torch
import torch.distributed as dist
from torch.utils.data import Dataset
from ase.db import connect
import polars as pl
from tqdm import tqdm
from pathlib import Path
import numpy as np

class AtomisticLanguageDataset(Dataset):
    def __init__(self, tokenizer, db_path, csv_path, thinking=False, max_num_tokens=1024, dataset_name=None):
        super().__init__()
        self.db = connect(db_path) # ordered the same way as the df. # this is somehow not true anymore? whoops.
        self.tokenizer = tokenizer
        self.thinking = thinking
        self.df = pl.read_csv(csv_path)
        self.max_num_tokens = max_num_tokens
        self.dataset_name = dataset_name
        # self.id_name = [column for column in self.df.columns if column.endswith('_id')][0]
        self.id_name = self.df.columns[0]

        # lookup betwen dataset and db id
        self.dataset_id_to_db_idx = {}
        for row in tqdm(self.db.select(), total=len(self.db), desc="Building index for dataset"):
            dataset_id = row.data['smiles']
            self.dataset_id_to_db_idx[str(dataset_id)] = row.id

        # lookup between dataset and df index
        self.dataset_id_to_df_idx = {}
        for row_idx in range(len(self.df)):
            dataset_id = self.df[row_idx][self.id_name][0]
            self.dataset_id_to_df_idx[str(dataset_id)] = row_idx

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return self.prepare_sample(idx)

    def prepare_sample(self, idx):

        # process single atom
        description = self.df[idx]['description'][0]
        row = self.db.get(self.dataset_id_to_db_idx[str(self.df[idx][self.id_name][0])])
        
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
            "id" : self.df[idx][self.id_name][0]
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
        print(dataset_ind, start, idx, list(self.datasets.keys())[dataset_ind])
        
        print(dataset_ind, start, idx, list(self.datasets.keys())[dataset_ind])
        
        print(dataset_ind, start, idx, list(self.datasets.keys())[dataset_ind])
        
        print(self.cum_lengths)
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