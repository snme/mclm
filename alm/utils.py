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
import random


# Task registry for multi-task instruction tuning. Each task draws the same
# atomistic input (the <atoms> token) but asks a different question and pulls
# its target from a different CSV column. Stage 1 uses DEFAULT_TASKS (describe
# only) so training stays identical; Stage 2+ can pass `tasks=ALL_TASKS`.
DESCRIBE_TASK = {
    "name": "describe",
    "system": "You are an expert at materials science and atomistic structure.",
    "user": "<atoms>\nDescribe the structure of this material.",
    "target_column": "description",
    "target_format": None,   # passthrough string
    "bucket": "describe",
}

_PROPERTY_PREDICTION_SYSTEM = (
    "You are a material scientist. "
    "Look at the atomistic structure of the given crystalline material and predict its property. "
    'The output must be in JSON format, e.g. {"property_name": predicted_value}. '
    "Answer as precisely and concisely as possible."
)


def _property_task(prop_name):
    return {
        "name": f"predict_{prop_name}",
        "system": _PROPERTY_PREDICTION_SYSTEM,
        "user": f"<atoms>\nProperty name: {prop_name}.",
        "target_column": prop_name,
        "target_format": lambda v, n=prop_name: f'{{"{n}": {v}}}',
        "bucket": "property_apps",
    }


# Eval-only: text-input property prediction, mirroring LLM4Mat-Bench's
# `{prop}_description_zero_shot` task family. No <atoms> — pure LLM capability
# probe on the same benchmark scenes for direct comparison with the paper.
_DESC_PROPERTY_PREDICTION_SYSTEM = (
    "You are a material scientist. "
    "Look at the structure description of the given crystalline material and predict its property. "
    'The output must be in JSON format, e.g. {"property_name": predicted_value}. '
    "Answer as precisely and concisely as possible."
)


def _desc_to_property_task(prop_name):
    # Eval-only. Not part of any training bucket.
    return {
        "name": f"predict_{prop_name}_from_description",
        "system": _DESC_PROPERTY_PREDICTION_SYSTEM,
        "user_template": "Structure description: {description}\nProperty name: " + prop_name + ".",
        "input_columns": ["description"],
        "target_column": prop_name,
        "target_format": lambda v, n=prop_name: f'{{"{n}": {v}}}',
        "bucket": "eval_only",
    }


DEFAULT_TASKS = [DESCRIBE_TASK]
ALL_TASKS = [
    DESCRIBE_TASK,
    _property_task("bandgap"),
    _property_task("e_form"),
]


# Per-dataset "headline" numeric properties benchmarked in LLM4Mat-Bench. Each subdataset
# has its own column naming + property set. Missing values (null/NaN) for a given row are
# filtered out at sample time in prepare_sample, so a task with partial coverage is fine.
_DATASET_PROPERTIES = {
    "cantor_hea":  ["Ef_per_atom", "e_above_hull", "volume_per_atom"],
    "gnome":       ["Formation_Energy_Per_Atom", "Bandgap", "Decomposition_Energy_Per_Atom", "Density"],
    "hmof":        ["max_co2_adsp", "void_fraction", "surface_area_m2g", "lcd", "pld"],
    "jarvis_dft":  ["formation_energy_peratom", "optb88vdw_bandgap", "mbj_bandgap", "ehull"],
    "jarvis_qetb": ["energy_per_atom", "indir_gap", "f_enp"],
    "mp":          ["formation_energy_per_atom", "band_gap", "energy_above_hull", "density"],
    "omdb":        ["bandgap"],
    "oqmd":        ["bandgap", "e_form"],
    "qmof":        ["bandgap", "lcd", "pld", "energy_total"],
    "snumat":      ["Band_gap_HSE", "Band_gap_GGA"],
}


# MatterChat-style natural-language tasks. Same MP rows as LLM4Mat's `mp` config,
# different prompt/target shape — matches eval_matterchat.py's TASK_DEFS so train
# and eval agree. 3 reg + 4 binary + 2 multi-class = 9 tasks per row.
_MATTERCHAT_PROPERTIES = {
    "matterchat_mp": {
        "reg": [
            ("formation_energy",  "Predict the formation energy per atom (eV/atom).", "eV/atom"),
            ("energy_above_hull", "Predict the energy above the convex hull (eV/atom).", "eV/atom"),
            ("bandgap",           "Predict the band gap (eV).", "eV"),
        ],
        "binary": [
            ("is_metal",        "Is this material a metal? Answer A) Yes or B) No."),
            ("is_magnetic",     "Is this material magnetic? Answer A) Yes or B) No."),
            ("direct_bandgap",  "Is the band gap direct? Answer A) Yes or B) No."),
            ("stable",          "Is this material thermodynamically stable? Answer A) Yes or B) No."),
        ],
        "multiclass": [
            ("magnetic_order",
             "Classify the magnetic ordering. A) NM (non-magnetic) B) FM (ferromagnetic) "
             "C) AFM (antiferromagnetic) D) FiM (ferrimagnetic).",
             {"NM": "A", "FM": "B", "AFM": "C", "FiM": "D"}),
            ("crystal_system",
             "Which crystal system does this material belong to? "
             "A) Cubic B) Tetragonal C) Orthorhombic D) Hexagonal "
             "E) Trigonal F) Monoclinic G) Triclinic.",
             {"Cubic": "A", "Tetragonal": "B", "Orthorhombic": "C", "Hexagonal": "D",
              "Trigonal": "E", "Monoclinic": "F", "Triclinic": "G"}),
        ],
    },
}

_MATTERCHAT_SYSTEM = (
    "You are a material scientist. "
    "Look at the structure of the given crystalline material and predict its property."
)


def _natural_reg_task(col, question, unit):
    """Natural-language regression: 'X.XXXX <unit>' instead of JSON."""
    return {
        "name": f"predict_{col}_natural",
        "system": _MATTERCHAT_SYSTEM,
        "user": f"<atoms>\n{question}",
        "target_column": col,
        "target_format": lambda v, u=unit: f"{float(v):.4f} {u}",
        "bucket": "property_apps",
    }


def _binary_yn_task(col, question):
    """Binary cls with 'A) Yes' / 'B) No' targets so extract_choice parses cleanly."""
    return {
        "name": f"predict_{col}_yn",
        "system": _MATTERCHAT_SYSTEM,
        "user": f"<atoms>\n{question}",
        "target_column": col,
        "target_format": lambda v: "A) Yes" if bool(v) else "B) No",
        "bucket": "property_apps",
    }


def _multi_class_task(col, question, label_map):
    """Multi-class cls. Target is '<letter>) <raw label>' (e.g. 'C) AFM')."""
    valid = frozenset(label_map.keys())
    return {
        "name": f"predict_{col}_mc",
        "system": _MATTERCHAT_SYSTEM,
        "user": f"<atoms>\n{question}",
        "target_column": col,
        "target_format": lambda v, m=label_map: f"{m[str(v)]}) {v}",
        "valid_values": valid,
        "bucket": "property_apps",
    }


def _matterchat_tasks(dataset_name):
    spec = _MATTERCHAT_PROPERTIES.get(dataset_name)
    if not spec:
        return []
    out = []
    for col, q, unit in spec["reg"]:
        out.append(_natural_reg_task(col, q, unit))
    for col, q in spec["binary"]:
        out.append(_binary_yn_task(col, q))
    for col, q, lm in spec["multiclass"]:
        out.append(_multi_class_task(col, q, lm))
    return out


def tasks_for_dataset(dataset_name):
    """Return describe + per-property prediction tasks for one LLM4Mat-Bench subdataset.

    Returns just [DESCRIBE_TASK] for unknown datasets so training degrades gracefully
    rather than erroring.
    """
    props = _DATASET_PROPERTIES.get(dataset_name, [])
    return [DESCRIBE_TASK] + [_property_task(p) for p in props]


def eval_tasks_description_input(dataset_name):
    """Eval-only: text-input property prediction (description → property value).

    Matches LLM4Mat-Bench's `{prop}_description_zero_shot` prompt family but in
    Qwen3 ChatML. Intended for evaluation parity with the paper, not training.
    """
    props = _DATASET_PROPERTIES.get(dataset_name, [])
    return [_desc_to_property_task(p) for p in props]


# GPT-Narratives-for-Materials — richer GPT-written narrative + application explanation.
NARRATE_TASK = {
    "name": "narrate",
    "system": "You are an expert at materials science and atomistic structure.",
    "user": "<atoms>\nProvide a detailed narrative description of this material and its properties.",
    "target_column": "gpt_text",
    "target_format": None,
    "bucket": "describe",
}
EXPLAIN_TASK = {
    "name": "explain_applications",
    "system": "You are a material scientist reasoning about applications and use cases.",
    "user": "<atoms>\nBased on this material's properties, list plausible application areas with reasoning.",
    "target_column": "gpt_explanation",
    "target_format": None,
    "bucket": "property_apps",
}

_NARRATIVE_PROPERTIES = {
    "aflow2":     ["band gap (eV)", "density (g/cm³)", "energy above hull (eV/atom)",
                   "volume (Å³)", "energy per atom (eV/atom)", "formation energy per atom (eV/atom)",
                   "enthalpy per atom (eV/atom)"],
    "dft_3d":     ["formation energy per atom (eV/atom)", "band gap (eV)", "total energy per atom (eV/atom)",
                   "energy above hull (eV/atom)", "density (g/cm³)", "volume (Å³)",
                   "total magnetization (μB/f.u.)", "enthalpy per atom (eV/atom)"],
    "mp_3d_2020": ["energy per atom (eV/atom)", "volume (Å³)", "formation energy per atom (eV/atom)",
                   "energy above hull (eV/atom)", "band gap (eV)", "density (g/cm³)",
                   "total magnetization (μB/f.u.)", "enthalpy per atom (eV/atom)"],
    "oqmd":       ["_oqmd_band_gap", "_oqmd_delta_e", "_oqmd_stability",
                   "Enthalpy per atom (eV/atom)", "density (g/cm3)"],
}


def narrative_tasks_for(dataset_name):
    """describe + explain + per-property prediction for one GPT-Narratives parquet."""
    props = _NARRATIVE_PROPERTIES.get(dataset_name, [])
    return [NARRATE_TASK, EXPLAIN_TASK] + [_property_task(p) for p in props]


# Bucketed views of the same task registry — used by Stage 2's 5-bucket sampler.
def describe_tasks_for_dataset(dataset_name):   return [DESCRIBE_TASK]
def property_tasks_for_dataset(dataset_name):
    if dataset_name in _MATTERCHAT_PROPERTIES:
        return _matterchat_tasks(dataset_name)
    return [_property_task(p) for p in _DATASET_PROPERTIES.get(dataset_name, [])]
def describe_tasks_for_narrative(dataset_name): return [NARRATE_TASK]
def applications_tasks_for_narrative(dataset_name):
    return [EXPLAIN_TASK] + [_property_task(p) for p in _NARRATIVE_PROPERTIES.get(dataset_name, [])]


class AtomisticLanguageDataset(Dataset):
    def __init__(self, tokenizer, db_path=None, csv_path=None, thinking=False, max_num_tokens=1024,
                 dataset_name=None, cached_embs_path=None, tasks=None):
        super().__init__()
        self.tokenizer = tokenizer
        self.thinking = thinking
        self.max_num_tokens = max_num_tokens
        self.dataset_name = dataset_name
        self.cached_embs_path = cached_embs_path

        # Filter requested tasks to those whose target column AND input columns
        # (if any — used by description-input eval tasks) are all in this CSV.
        header_cols = pl.read_csv(csv_path, n_rows=0).columns
        requested_tasks = DEFAULT_TASKS if tasks is None else tasks

        def _task_ok(t):
            needed = [t["target_column"]] + list(t.get("input_columns", []))
            return all(c in header_cols for c in needed)

        self.tasks = [t for t in requested_tasks if _task_ok(t)]
        if not self.tasks:
            raise ValueError(
                f"No requested tasks have all required columns in {csv_path}. "
                f"Requested: {[t['name'] for t in requested_tasks]}. "
                f"Available columns: {header_cols}"
            )

        self.id_name = [col for col in header_cols if col.endswith('_id')][0]
        needed_cols = set()
        for t in self.tasks:
            needed_cols.add(t["target_column"])
            needed_cols.update(t.get("input_columns", []))
        target_cols = sorted(needed_cols)
        df = pl.read_csv(csv_path, columns=[self.id_name] + target_cols)
        ids = df[self.id_name].to_list()
        column_data = {c: df[c].to_list() for c in target_cols}
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
            keep_mask = [str(sid) in self.cached_index for sid in ids]
            self._ids = [s for s, k in zip(ids, keep_mask) if k]
            self._column_data = {c: [v for v, k in zip(vals, keep_mask) if k]
                                 for c, vals in column_data.items()}
            if is_main_process() and len(self._ids) != before:
                print(
                    f"[{dataset_name or 'dataset'}] filtered {before - len(self._ids)} / {before} "
                    f"samples with no cached embedding."
                )
        else:
            self._ids = ids
            self._column_data = column_data
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

        # Back-compat alias: generate.py and older callers read _descriptions directly.
        self._descriptions = self._column_data.get("description")

    def __len__(self):
        return len(self._ids)

    def __getitem__(self, idx):
        return self.prepare_sample(idx)

    def _pick_task(self, idx):
        # Randomly choose a task whose target for this row is actually present.
        # Describe is always available (description is never null in LLM4Mat-Bench);
        # property tasks can have null bandgap/e_form for some rows and get skipped.
        # Multi-class tasks carry a `valid_values` set; skip rows where the raw value
        # isn't in the label map (e.g. "Unknown" in magnetic_order).
        def _is_valid(t):
            v = self._column_data[t["target_column"]][idx]
            if v is None:
                return False
            vv = t.get("valid_values")
            return vv is None or str(v) in vv
        candidates = [t for t in self.tasks if _is_valid(t)]
        if not candidates:
            candidates = self.tasks  # last resort; will surface as None target below
        return random.choice(candidates)

    def prepare_sample(self, idx):

        # process single atom
        sample_id = self._ids[idx]
        task = self._pick_task(idx)
        target_raw = self._column_data[task["target_column"]][idx]
        target = (task["target_format"](target_raw)
                  if task["target_format"] is not None else str(target_raw))

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

        # Description-input eval tasks (text-only) set a user_template that
        # substitutes row columns; normal atoms-input tasks just use `user`.
        if "user_template" in task:
            user_msg = task["user_template"].format(
                **{c: self._column_data[c][idx] for c in task.get("input_columns", [])}
            )
        else:
            user_msg = task["user"]
        messages = [
            {"role": "system", "content": task["system"]},
            {"role": "user", "content": user_msg},
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
            messages + [{"role": "assistant", "content": target}],
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
                 cached_embs_parent_path=None, atomistic_model_name="orb_v3_direct_20_omat",
                 tasks=None):
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
            # `tasks` can be None (describe-only), a list (same tasks for every subdataset),
            # or a callable(dataset_name) -> list (per-dataset tasks, typical for LLM4Mat-Bench
            # where properties differ between oqmd/gnome/hmof/jarvis_dft/...).
            dataset_tasks = tasks(dataset_name) if callable(tasks) else tasks
            dataset = AtomisticLanguageDataset(
                tokenizer=tokenizer,
                db_path=folder / f'{split}.db' if cached_bin is None else None,
                csv_path=folder / f'{split}.csv',
                thinking=thinking,
                max_num_tokens=max_num_tokens,
                dataset_name=dataset_name,
                cached_embs_path=cached_bin,
                tasks=dataset_tasks,
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


def _atoms_struct_to_ase(a):
    """Convert a GPT-Narratives `atoms` struct row to an ASE Atoms (no CIF needed)."""
    from ase import Atoms
    cell = a["lattice_mat"]
    coords = a["coords"]
    if a["cartesian"]:
        return Atoms(symbols=a["elements"], positions=coords, cell=cell, pbc=True)
    return Atoms(symbols=a["elements"], scaled_positions=coords, cell=cell, pbc=True)


class GPTNarrativeDataset(Dataset):
    """GPT-Narratives-for-Materials parquet → (ASE Atoms, prompt, target) samples.

    Mirrors AtomisticLanguageDataset's two-mode interface: live mode builds Atoms from
    the parquet's `atoms` struct column on demand; cached mode reads OrbV3 features
    from a flat mmap keyed by parquet row index (string).
    """
    def __init__(self, tokenizer, parquet_path, cached_embs_path=None,
                 thinking=False, max_num_tokens=1024, dataset_name=None, tasks=None):
        super().__init__()
        self.tokenizer = tokenizer
        self.thinking = thinking
        self.max_num_tokens = max_num_tokens
        self.dataset_name = dataset_name

        header = pl.read_parquet(parquet_path, n_rows=0).columns
        requested = [NARRATE_TASK] if tasks is None else tasks
        self.tasks = [t for t in requested if t["target_column"] in header]
        if not self.tasks:
            raise ValueError(
                f"No requested tasks have target columns in {parquet_path}. "
                f"Requested: {[t['name'] for t in requested]}."
            )

        target_cols = sorted({t["target_column"] for t in self.tasks})
        need_atoms = cached_embs_path is None
        df = pl.read_parquet(parquet_path, columns=(["atoms"] if need_atoms else []) + target_cols)
        column_data = {c: df[c].to_list() for c in target_cols}
        atoms_list = df["atoms"].to_list() if need_atoms else None
        n_rows = len(column_data[target_cols[0]])
        del df

        if cached_embs_path is not None:
            bin_path = Path(cached_embs_path)
            idx_path = bin_path.with_suffix(".idx.json")
            with open(idx_path) as f:
                self.cached_index = {k: tuple(v) for k, v in json.load(f).items()}
            self.cached_embs = np.memmap(bin_path, dtype=np.float32, mode="r").reshape(-1, 256)
            keep = [i for i in range(n_rows) if str(i) in self.cached_index]
            self._column_data = {c: [vals[i] for i in keep] for c, vals in column_data.items()}
            self._atoms = None
            self._ids = [str(i) for i in keep]
            if is_main_process() and len(keep) != n_rows:
                print(f"[{dataset_name or 'narrative'}] filtered {n_rows - len(keep)} / {n_rows} "
                      f"samples with no cached embedding.")
        else:
            self.cached_embs = None
            self.cached_index = None
            self._column_data = column_data
            self._atoms = atoms_list
            self._ids = [str(i) for i in range(n_rows)]

        # Back-compat alias matching AtomisticLanguageDataset._descriptions so generate.py works.
        self._descriptions = self._column_data.get("gpt_text")

    def __len__(self):
        return len(self._ids)

    def __getitem__(self, idx):
        return self.prepare_sample(idx)

    def _pick_task(self, idx):
        cand = [t for t in self.tasks if self._column_data[t["target_column"]][idx] is not None]
        return random.choice(cand or self.tasks)

    def prepare_sample(self, idx):
        sample_id = self._ids[idx]
        task = self._pick_task(idx)
        target_raw = self._column_data[task["target_column"]][idx]
        target = (task["target_format"](target_raw)
                  if task["target_format"] is not None else str(target_raw))

        MIN_TEXT_TOKENS = 256
        max_atoms = max(1, self.max_num_tokens - MIN_TEXT_TOKENS)

        if self.cached_embs is not None:
            offset, full_n_atoms = self.cached_index[sample_id]
            n_atoms = min(full_n_atoms, max_atoms)
            atom_embed = torch.from_numpy(
                np.array(self.cached_embs[offset : offset + n_atoms], dtype=np.float32)
            )
            atoms_obj = None
        else:
            atoms_obj = _atoms_struct_to_ase(self._atoms[idx])
            atom_embed = None
            n_atoms = min(len(atoms_obj), max_atoms)

        messages = [
            {"role": "system", "content": task["system"]},
            {"role": "user", "content": task["user"]},
        ]
        text_budget = self.max_num_tokens - n_atoms + 1
        prompt_ids = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            enable_thinking=self.thinking, truncation=True, max_length=text_budget,
        )
        full_ids = self.tokenizer.apply_chat_template(
            messages + [{"role": "assistant", "content": target}],
            tokenize=True, add_generation_prompt=False,
            enable_thinking=self.thinking, truncation=True, max_length=text_budget,
        )

        input_ids = torch.tensor([full_ids], dtype=torch.long)
        labels = torch.tensor([[-100] * len(prompt_ids) + full_ids[len(prompt_ids):]], dtype=torch.long)
        cap = text_budget
        sample = {
            "input_ids": input_ids[:, :cap],
            "labels": labels[:, :cap],
            "attention_mask": torch.ones_like(input_ids[:, :cap]),
            "id": sample_id,
        }
        if atom_embed is not None:
            sample["atom_embed"] = atom_embed
        else:
            sample["atom_rows"] = [atoms_obj]
        return sample


class MaScQADataset(Dataset):
    """Text-only materials-science Q&A from MaScQA (650 questions across 14 topics).

    Joins the topic-grouped `mascqa-eval.json` (questions) with
    `scoresheets/all_questions.xlsx` (ground-truth answers and question types).
    Returns samples with an empty atom_embed so the shared LLM forward pass
    skips the <atoms> splice (since no <atoms> token is in the prompt) and
    trains purely on next-token loss.
    """
    _SYSTEM = (
        "You are a materials science expert answering questions from the MaScQA benchmark. "
        "For multiple-choice / matching questions, respond with only the letter of the correct option. "
        "For numerical questions, respond with the numeric value or range as given."
    )

    def __init__(self, tokenizer, questions_json, scoresheet_xlsx,
                 thinking=False, max_num_tokens=1024,
                 split="train", val_frac=0.2, split_seed=42):
        import pandas as pd
        from collections import defaultdict
        super().__init__()
        self.tokenizer = tokenizer
        self.thinking = thinking
        self.max_num_tokens = max_num_tokens

        with open(questions_json) as f:
            grouped = json.load(f)
        qid_to_question = {}
        for topic, d in grouped.items():
            for qid, q in zip(d["qids"], d["questions"]):
                qid_to_question[qid] = q

        df = pd.read_excel(scoresheet_xlsx)
        self._qids, self._questions, self._answers, self._qtypes, self._topics = [], [], [], [], []
        for _, row in df.iterrows():
            qid = row["Question Info"]
            if qid not in qid_to_question:
                continue
            self._qids.append(qid)
            self._questions.append(qid_to_question[qid])
            self._answers.append(str(row["Correct Answer"]).strip())
            self._qtypes.append(row["Question Type"])
            self._topics.append(row["TOPIC"])

        # Stratified train/val partition by topic. Deterministic per (split_seed, topic).
        # MaScQA is a published benchmark (650 Qs); held-out val keeps it usable as one.
        idx_by_topic = defaultdict(list)
        for i, t in enumerate(self._topics):
            idx_by_topic[t].append(i)
        rng = random.Random(split_seed)
        val_set = set()
        for topic in sorted(idx_by_topic):
            idxs = list(idx_by_topic[topic])
            rng.shuffle(idxs)
            n_val = max(1, int(round(val_frac * len(idxs))))
            val_set.update(idxs[:n_val])
        keep = [i for i in range(len(self._qids)) if (i in val_set) == (split == "validation")]
        self._qids      = [self._qids[i]      for i in keep]
        self._questions = [self._questions[i] for i in keep]
        self._answers   = [self._answers[i]   for i in keep]
        self._qtypes    = [self._qtypes[i]    for i in keep]
        self._topics    = [self._topics[i]    for i in keep]

    def __len__(self):
        return len(self._qids)

    def __getitem__(self, idx):
        return self.prepare_sample(idx)

    def prepare_sample(self, idx):
        messages = [
            {"role": "system", "content": self._SYSTEM},
            {"role": "user", "content": self._questions[idx]},
        ]
        prompt_ids = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            enable_thinking=self.thinking, truncation=True, max_length=self.max_num_tokens,
        )
        full_ids = self.tokenizer.apply_chat_template(
            messages + [{"role": "assistant", "content": self._answers[idx]}],
            tokenize=True, add_generation_prompt=False,
            enable_thinking=self.thinking, truncation=True, max_length=self.max_num_tokens,
        )
        input_ids = torch.tensor([full_ids], dtype=torch.long)
        labels = torch.tensor([[-100] * len(prompt_ids) + full_ids[len(prompt_ids):]], dtype=torch.long)
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": torch.ones_like(input_ids),
            "id": self._qids[idx],
            # Empty atom_embed: _merge_embeddings sees no <atoms> token in the prompt
            # and passes text_embeds through unchanged. Keeps the collate + forward
            # signatures identical to the atomistic datasets.
            "atom_embed": torch.zeros(0, 256, dtype=torch.float32),
        }


class CamelAIDataset(Dataset):
    """CAMEL-AI chem + physics role-play Q&A as text-only ChatML samples."""
    _SYSTEM = "You are a helpful science assistant."

    def __init__(self, tokenizer, jsonl_path, thinking=False, max_num_tokens=1024,
                 split="train", val_size=500, split_seed=42):
        super().__init__()
        self.tokenizer = tokenizer
        self.thinking = thinking
        self.max_num_tokens = max_num_tokens
        with open(jsonl_path) as f:
            rows = [json.loads(line) for line in f]
        rng = random.Random(split_seed)
        perm = list(range(len(rows)))
        rng.shuffle(perm)
        val_set = set(perm[:min(val_size, len(rows))])
        keep = [i for i in range(len(rows)) if (i in val_set) == (split == "validation")]
        self._rows = [rows[i] for i in keep]

    def __len__(self):
        return len(self._rows)

    def __getitem__(self, idx):
        r = self._rows[idx]
        sys_msg = self._SYSTEM
        if r.get("topic") or r.get("sub_topic"):
            sys_msg += f" (Topic: {r.get('topic','')} / {r.get('sub_topic','')})"
        messages = [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": r["message_1"]},
        ]
        prompt_ids = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            enable_thinking=self.thinking, truncation=True, max_length=self.max_num_tokens,
        )
        full_ids = self.tokenizer.apply_chat_template(
            messages + [{"role": "assistant", "content": r["message_2"]}],
            tokenize=True, add_generation_prompt=False,
            enable_thinking=self.thinking, truncation=True, max_length=self.max_num_tokens,
        )
        input_ids = torch.tensor([full_ids], dtype=torch.long)
        labels = torch.tensor([[-100] * len(prompt_ids) + full_ids[len(prompt_ids):]], dtype=torch.long)
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": torch.ones_like(input_ids),
            "id": f"camel/{idx}",
            "atom_embed": torch.zeros(0, 256, dtype=torch.float32),
        }


class ArxivAbstractDataset(Dataset):
    """JARVIS arXiv abstracts as instruction-tuning samples.

    Format (ChatML, prompt masked with -100, supervise only the abstract):
        system    : "You are a scientific writing assistant. Given a paper
                     title and arXiv categories, write a plausible abstract."
        user      : "Title: {title}\\nCategories: {cats}\\n\\nAbstract:"
        assistant : "{abstract}"

    Why not raw continued-pretraining: training every token (title prefix
    included, no ChatML wrapper, <|endoftext|> EOS) at 14% of gradient steps
    partially undoes Qwen3-8B's instruction-tuning. The model then falls back
    to base-LM web priors at eval time — markdown image embeds, imgur URLs,
    materialsproject.org URLs — most visibly on uncertain tasks (MaScQA 43%,
    MatterChat magnetic_order 41%, crystal_system 39%). Instruction-tuning the
    arxiv bucket keeps the same scientific-text exposure but inside the
    follow-the-user discipline that suppresses those fallbacks.
    """
    _SYSTEM = ("You are a scientific writing assistant. Given a paper title "
               "and arXiv categories, write a plausible abstract.")

    def __init__(self, tokenizer, parquet_path, max_num_tokens=1024,
                 split="train", val_size=500, split_seed=42, thinking=False):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_num_tokens = max_num_tokens
        self.thinking = thinking
        import pyarrow.parquet as pq
        t = pq.read_table(parquet_path, columns=["id", "title", "categories", "abstract"])
        ids = t.column("id").to_pylist()
        titles = t.column("title").to_pylist()
        cats = t.column("categories").to_pylist()
        abstracts = t.column("abstract").to_pylist()
        rng = random.Random(split_seed)
        perm = list(range(len(ids)))
        rng.shuffle(perm)
        val_set = set(perm[:min(val_size, len(ids))])
        keep = [i for i in range(len(ids)) if (i in val_set) == (split == "validation")]
        self._ids       = [ids[i]       for i in keep]
        self._titles    = [titles[i]    for i in keep]
        self._cats      = [cats[i]      for i in keep]
        self._abstracts = [abstracts[i] for i in keep]

    def __len__(self):
        return len(self._ids)

    def __getitem__(self, idx):
        title = self._titles[idx] or ""
        cats = self._cats[idx] or ""
        abstract = self._abstracts[idx] or ""
        messages = [
            {"role": "system", "content": self._SYSTEM},
            {"role": "user",   "content": f"Title: {title}\nCategories: {cats}\n\nAbstract:"},
        ]
        prompt_ids = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            enable_thinking=self.thinking, truncation=True,
            max_length=self.max_num_tokens,
        )
        full_ids = self.tokenizer.apply_chat_template(
            messages + [{"role": "assistant", "content": abstract}],
            tokenize=True, add_generation_prompt=False,
            enable_thinking=self.thinking, truncation=True,
            max_length=self.max_num_tokens,
        )
        # Truncation may chop the assistant turn before the prompt finishes;
        # in that case fall back to a labels-all-prompt sample (the trainer
        # treats it as a no-op gradient row).
        n_prompt = min(len(prompt_ids), len(full_ids))
        input_ids = torch.tensor([full_ids], dtype=torch.long)
        label_seq = [-100] * n_prompt + full_ids[n_prompt:]
        labels = torch.tensor([label_seq], dtype=torch.long)
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": torch.ones_like(input_ids),
            "id": self._ids[idx],
            "atom_embed": torch.zeros(0, 256, dtype=torch.float32),
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