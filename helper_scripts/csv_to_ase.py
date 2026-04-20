# %%
from pathlib import Path
import polars as pl
from ase.db import connect
from tqdm import tqdm
import os
from ase.io import read
from io import StringIO

# %%
data_dir = Path('/home/sathya/Data/LLM4Mat-Bench/data')
for dataset_path in data_dir.iterdir():
    if dataset_path.is_dir():
        df = pl.read_csv(dataset_path / 'train.csv')
        df = df.drop("Unnamed: 0", strict=False)
        id_name = [column for column in df.columns if column.endswith('_id')][0]
        db = connect(dataset_path / 'train.db')
        for row in tqdm(range(len(df)), total=len(df), desc=f'Processing train data for {dataset_path}'):
            ase_atoms = read(StringIO(df[row]['cif_structure'][0]), format='cif')
            data = {k: df[row][k][0] for k in df[row].columns}
            data['smiles'] = df[row][id_name][0]  # id column
            db.write(
                ase_atoms,
                data=data,
                # key_value_pairs=data,
            )
        df = pl.read_csv(dataset_path / 'validation.csv')
        db = connect(dataset_path / 'validation.db')
        for row in tqdm(range(len(df)), total=len(df), desc=f'Processing validation data for {dataset_path}'):
            ase_atoms = read(StringIO(df[row]['cif_structure'][0]), format='cif')
            data = {k: df[row][k][0] for k in df[row].columns}
            data['smiles'] = df[row][id_name][0]  # id column
            db.write(
                ase_atoms,
                data=data,
                # key_value_pairs=data,
            )

