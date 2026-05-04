import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

# orb imports
from orb_models.forcefield import atomic_system, pretrained
from orb_models.forcefield.base import batch_graphs

# torch dist imports
from utils import is_main_process

class AtomisticLanguageModel(nn.Module):
    # expects the underscored version of the OrbV3 atomistic model name
    # first implementation will has one atomistic token per atom.
    def __init__(self, llm_name='Qwen/Qwen3-8B', atomistic_model_name='orb_v3_direct_20_omat', device=None,
                 attn_implementation="flash_attention_2", use_cached_embeddings=False, max_atoms=None,
                 num_output_atom_tokens: int = 8):
        super().__init__()
        self.device = device if device is not None else torch.device("cuda")
        self.use_cached_embeddings = use_cached_embeddings
        # Matches the cap applied in AtomisticLanguageDataset.prepare_sample so live mode
        # doesn't silently splice the full (uncapped) atom count back into the sequence.
        self.max_atoms = max_atoms

        # load the frozen llm
        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_name,
            torch_dtype=torch.bfloat16, # let's stick with b16 for now.
            attn_implementation=attn_implementation
        )
        self.tokenizer = AutoTokenizer.from_pretrained(llm_name)
        self.llm_hidden_dim = self.llm.config.hidden_size

        # freeze all LLM parameters
        for param in self.llm.parameters():
            param.requires_grad = False

        # Frozen atomistic encoder. Skipped when using pre-cached OrbV3 features —
        # no point paying the 7 GB GPU load + per-step graph build if we'll never call it.
        if use_cached_embeddings:
            self.atomistic_model = None
        else:
            model = getattr(pretrained, atomistic_model_name)
            orbff = model(
                device=self.device,
                precision="float32-high",   # or "float32-highest" / "float64
            )
            self.atomistic_model = orbff
            for param in self.atomistic_model.parameters():
                param.requires_grad = False
        
        # trainable atomistic projector: only trainable part of this model
        llm_dim = self.llm_hidden_dim
        atomistic_dim = 256 # figured out by hand honestly
        self.projector = nn.Sequential(
            nn.Linear(atomistic_dim, llm_dim),
            nn.GELU(),
            nn.Linear(llm_dim,  llm_dim),
        )
        self.atom_token = '<atoms>'
        self.tokenizer.add_tokens([self.atom_token])
        self.atoms_token_id = self.tokenizer.convert_tokens_to_ids(self.atom_token)

        # Stage 3a output-side tokens: emitted at the end of the assistant turn; their
        # final-layer hidden states are the input to AtomsMapper → MatterGen conditioning.
        # Embedding rows are frozen (Stage 3a freezes the entire LLM) and need to be
        # deterministic across the offline cache run and inference — so we seed before
        # resize and restore the global RNG state afterward.
        self.num_output_atom_tokens = num_output_atom_tokens
        self.output_atom_tokens = [f'[atoms_{i}]' for i in range(num_output_atom_tokens)]
        self.tokenizer.add_tokens(self.output_atom_tokens, special_tokens=True)
        self.output_atom_token_ids = self.tokenizer.convert_tokens_to_ids(self.output_atom_tokens)
        rng_state = torch.get_rng_state()
        torch.manual_seed(42)
        self.llm.resize_token_embeddings(len(self.tokenizer))
        torch.set_rng_state(rng_state)

    def encode_atoms(self, row_batch):
        # expects ASE atoms rows
        # assumes using OrbV3

        with torch.no_grad():
            # Accepts ASE db rows (have .toatoms()) or raw ASE Atoms (GPTNarrativeDataset).
            atoms_list = [r.toatoms() if hasattr(r, "toatoms") else r for r in row_batch]
            if self.max_atoms is not None:
                atoms_list = [a[:self.max_atoms] for a in atoms_list]
            batch = [
                atomic_system.ase_atoms_to_atom_graphs(
                    atoms,
                    self.atomistic_model.system_config,
                    device=self.device,
                )
                for atoms in atoms_list
            ]
            graph = batch_graphs(batch)
            results = self.atomistic_model.model(graph)
            node_features = results["node_features"]
        out = self.projector(node_features)
        n_atoms = tuple(graph.n_node.tolist())
        return out, n_atoms

    def encode_cached_atoms(self, atom_embeds):
        # atom_embeds: list of (N_i, 256) float32 tensors already produced by OrbV3.
        # Concat → project; mirrors encode_atoms' return contract.
        n_atoms = tuple(a.shape[0] for a in atom_embeds)
        projector_dtype = next(self.projector.parameters()).dtype
        stacked = torch.cat(
            [a.to(device=self.device, dtype=projector_dtype, non_blocking=True) for a in atom_embeds],
            dim=0,
        )
        out = self.projector(stacked)
        return out, n_atoms


    def forward(self, input_ids, attention_mask, labels=None, row_batch=None,
                atom_embeds=None, output_atoms_hidden_states: bool = False):
        """Unified entry point.

        - Default (Stage 1/2 training, eval): compute the LM loss and return HF
          CausalLMOutput. `labels` are required.
        - With `output_atoms_hidden_states=True` (Stage 3a/3b joint training,
          inference): return (B, K, hidden_dim) at the K output-side `[atoms_i]`
          positions. `labels` are ignored.

        Routing both modes through `forward()` lets DDP wrap the outer ALM (Stage 2
        pattern, train_stage2.py:299): a single DDP.forward call dispatches the
        right computation, gradient sync hooks fire correctly on backward, and
        callers can do `alm.module.<attr>` for everything else (no per-attribute
        DDP unwrap hacks).

        For text-only prompts (no input-side <atoms>): pass atom_embeds as a list
        of zero-row tensors, e.g. [torch.zeros(0, 256)] * B. The splice is skipped
        and the position offset is 0 for all samples.
        """
        # get atomistic features
        if atom_embeds is not None:
            atomistic_features, n_atoms = self.encode_cached_atoms(atom_embeds)
        else:
            atomistic_features, n_atoms = self.encode_atoms(row_batch)
        atomistic_features = torch.split(atomistic_features, n_atoms)

        # get text embeddings — self.llm is the unwrapped PEFT model here (DDP wraps
        # the outer ALM, not self.llm), so direct method access works.
        embed_layer = self.llm.get_input_embeddings()
        text_embeds = [embed_layer(sample_input_ids) for sample_input_ids in input_ids]

        if output_atoms_hidden_states:
            # Hidden-states branch (Stage 3a/3b joint training; inference)
            K = len(self.output_atom_token_ids)
            dummy_labels = [torch.zeros_like(sids) for sids in input_ids]
            new_embeds, _, new_attention_mask = self._merge_embeddings(
                text_embeds, atomistic_features, input_ids, dummy_labels, attention_mask
            )
            # Map original-token positions of [atoms_{i}] to post-splice positions:
            # each input-side <atoms> at position p shifts everything after it by
            # (n_atoms[b] - 1).
            output_atom_set = set(self.output_atom_token_ids)
            positions_per_sample = []
            for b in range(len(input_ids)):
                ids = input_ids[b].tolist()
                running_offset = 0
                sample_positions = []
                for orig_idx, tid in enumerate(ids):
                    if tid == self.atoms_token_id:
                        running_offset += n_atoms[b] - 1
                    elif tid in output_atom_set:
                        sample_positions.append(orig_idx + running_offset)
                assert len(sample_positions) == K, (
                    f"Sample {b}: expected exactly {K} output atom tokens, "
                    f"found {len(sample_positions)} in input_ids of len {len(ids)}."
                )
                positions_per_sample.append(sample_positions)
            outputs = self.llm(
                inputs_embeds=new_embeds,
                attention_mask=new_attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
            last_hidden = outputs.hidden_states[-1]  # (B, max_len, hidden_dim)
            return torch.stack(
                [last_hidden[b, positions_per_sample[b], :] for b in range(len(input_ids))],
                dim=0,
            )  # (B, K, hidden_dim)

        # Loss branch (Stage 1/2)
        new_embeds, new_labels, new_attention_mask = self._merge_embeddings(
            text_embeds, atomistic_features, input_ids, labels, attention_mask
        )
        outputs = self.llm(
            inputs_embeds=new_embeds,
            attention_mask=new_attention_mask,
            labels=new_labels,
            return_dict=True,
        )
        return outputs  # loss = outputs.loss

    def extract_atoms_hidden_states(self, input_ids, attention_mask,
                                    row_batch=None, atom_embeds=None):
        """Thin alias for `self.forward(..., output_atoms_hidden_states=True)`.

        Kept for inference call sites (helper_scripts/generate_stage3a.py etc.)
        that don't go through DDP. In DDP training, call alm(...) directly so
        DDP.forward dispatches and gradient sync hooks fire on backward.
        """
        return self.forward(
            input_ids, attention_mask, labels=None,
            row_batch=row_batch, atom_embeds=atom_embeds,
            output_atoms_hidden_states=True,
        )

    def _merge_embeddings(self, text_embeds, atomistic_features, input_ids, labels, attention_mask):
        batch_size = len(text_embeds)

        new_embeds = []
        new_labels = []
        new_attention_mask = []

        for b in range(batch_size):
            atom_token_embed = atomistic_features[b].to(
                dtype=text_embeds[b].dtype,
                device=text_embeds[b].device,
            )
            num_atomistic_tokens = atom_token_embed.shape[0]
            atoms_positions = (input_ids[b] == self.atoms_token_id).nonzero(as_tuple=True)[0]

            if len(atoms_positions) == 0:
                new_embeds.append(text_embeds[b])
                new_labels.append(labels[b])
                new_attention_mask.append(attention_mask[b])
                continue

            cur_labels = labels[b]
            cur_embs = text_embeds[b]
            curr_attn_mask = attention_mask[b]
            for position in atoms_positions:
                position_idx = int(position.item())
                # split embeddings
                emb_before = cur_embs[:position_idx]
                emb_after = cur_embs[position_idx + 1:]

                # stitch
                cur_embs = torch.cat([emb_before, atom_token_embed, emb_after], dim=0)
                
                # set labels and attention mask
                before_labels = cur_labels[:position_idx]
                atom_labels = torch.full(
                    (num_atomistic_tokens,), -100,
                    dtype=cur_labels.dtype, device=cur_labels.device
                )
                after_labels = cur_labels[position_idx + 1:]
                cur_labels = torch.cat([before_labels, atom_labels, after_labels])

                before_mask = curr_attn_mask[:position_idx]
                atom_mask = torch.ones(
                    num_atomistic_tokens,
                    dtype=curr_attn_mask.dtype, device=curr_attn_mask.device
                )
                after_mask = curr_attn_mask[position_idx + 1:]
                curr_attn_mask = torch.cat([before_mask, atom_mask, after_mask])

            new_embeds.append(cur_embs)
            new_labels.append(cur_labels)
            new_attention_mask.append(curr_attn_mask)

        max_len = max(len(emb) for emb in new_embeds)
        embed_dim = text_embeds[0].shape[-1]
        padded_embeds = torch.zeros(
            batch_size,
            max_len,
            embed_dim,
            dtype=text_embeds[0].dtype,
            device=text_embeds[0].device,
        )
        padded_labels = torch.full(
            (batch_size, max_len),
            -100,
            dtype=new_labels[0].dtype,
            device=new_labels[0].device,
        )
        padded_mask = torch.zeros(
            batch_size,
            max_len,
            dtype=new_attention_mask[0].dtype,
            device=new_attention_mask[0].device,
        )

        for b in range(batch_size):
            cur_len = len(new_embeds[b])
            padded_embeds[b, :cur_len] = new_embeds[b]
            padded_labels[b, :cur_len] = new_labels[b]
            padded_mask[b, :cur_len] = new_attention_mask[b]

        return padded_embeds, padded_labels, padded_mask
