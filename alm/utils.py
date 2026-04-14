import torch
import torch.distributed as dist
from torch.utils.data import Dataset
from ase.db import connect
import polars as pl
from tqdm import tqdm

class AtomisticLanguageDataset(Dataset):
    def __init__(self, tokenizer, db_path, csv_path, thinking=False, max_num_tokens=1024):
        super().__init__()
        self.db = connect(db_path) # ordered the same way as the df. # this is somehow not true anymore? whoops.
        self.tokenizer = tokenizer
        self.thinking = thinking
        self.df = pl.read_csv(csv_path)
        self.max_num_tokens = max_num_tokens

        # lookup betwen oqmd and db id
        self.oqmd_id_to_db_idx = {}
        for row in tqdm(self.db.select(), total=len(self.db), desc="Building index for dataset"):
            oqmd_id = int(row.data['smiles'])
            self.oqmd_id_to_db_idx[oqmd_id] = row.id

        # lookup between oqmd and df index
        self.oqmd_id_to_df_idx = {}
        for row_idx in range(len(self.df)):
            oqmd_id = self.df[row_idx]['oqmd_id'][0]
            self.oqmd_id_to_df_idx[oqmd_id] = row_idx

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return self.prepare_sample(idx)

    def prepare_sample(self, idx):

        # process single atom
        description = self.df[idx]['description'][0]
        row = self.db.get(self.oqmd_id_to_db_idx[self.df[idx]['oqmd_id'][0]])
        
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
            "oqmd_id" : self.df[idx]['oqmd_id'][0]
        }

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
        "oqmd_ids": [b["oqmd_id"] for b in batch],
    }