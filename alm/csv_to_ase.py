from ase.db import connect
from ase.io import read
from io import StringIO
import polars as pl
import shutil
import argparse
from tqdm import tqdm
import osdd

def csv_to_ase(folder_path):
    for folder in os.listdir(folder_path):
        shutil.copy(os.path.join(folder_path, folder, 'train.csv'), os.path.join('/tmp', f'{folder}_train.csv'))
        shutil.copy(os.path.join(folder_path, folder, 'validation.csv'), os.path.join('/tmp', f'{folder}_validation.csv'))
        df = pl.read_csv(os.path.join(folder_path, folder, 'train.csv'))
        db = connect(os.path.join('/tmp', f'{folder}_train.db'))
        for row in tqdm(range(len(df)), total=len(df), desc=f'Processing train data for {folder}'):
            ase_atoms = read(StringIO(df[row]['cif_structure'][0]))
            data = {k: df[row][k][0] for k in df[row].columns}
            data['smiles'] = df[row][df[row].columns[0]][0] # id column
            db.write(
                ase_atoms,
                data=data,
                key_value_data=data,
            )
        df = pl.read_csv(os.path.join(folder_path, folder, 'validation.csv'))
        db = connect(os.path.join('/tmp', f'{folder}_validation.db'))
        for row in tqdm(range(len(df)), total=len(df), desc=f'Processing validation data for {folder}'):
            ase_atoms = read(StringIO(df[row]['cif_structure'][0]))
            data = {k: df[row][k][0] for k in df[row].columns}
            data['smiles'] = df[row][df[row].columns[0]][0] # id column
            db.write(
                ase_atoms,
                data=data,
                key_value_data=data,
            )
        shutil.copy(f'/tmp/{folder}_train.db', os.path.join(folder_path, folder, 'train.db'))
        shutil.copy(f'/tmp/{folder}_validation.db', os.path.join(folder_path, folder, 'validation.db'))
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder_path", type=str, required=True)
    args = parser.parse_args()
    csv_to_ase(args.folder_path)