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
    def __init__(self, llm_name='Qwen/Qwen3-8B', atomistic_model_name='orb_v3_direct_20_omat', device=None):
        super().__init__()
        self.device = device if device is not None else torch.device("cuda")

        # load the frozen llm
        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_name,
            torch_dtype=torch.bfloat16, # let's stick with b16 for now.
            attn_implementation="flash_attention_2"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(llm_name)
        self.llm_hidden_dim = self.llm.config.hidden_size

        # freeze all LLM parameters
        for param in self.llm.parameters():
            param.requires_grad = False

        # load frozen atomistic encoder. for now, due to inference speed, let's go with OrbV3.
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
        self.llm.resize_token_embeddings(len(self.tokenizer))

    def encode_atoms(self, row_batch): 
        # expects ASE atoms rows
        # assumes using OrbV3

        with torch.no_grad():
            batch = [
                atomic_system.ase_atoms_to_atom_graphs(
                    row.toatoms(),
                    self.atomistic_model.system_config,
                    device=self.device,
                )
                for row in row_batch
            ]
            graph = batch_graphs(batch)
            results = self.atomistic_model.model(graph)
            node_features = results["node_features"]
        out = self.projector(node_features)
        n_atoms = tuple(graph.n_node.tolist())
        return out, n_atoms


    def forward(self, row_batch, input_ids, attention_mask, labels):
        # get atomistic features
        atomistic_features, n_atoms = self.encode_atoms(row_batch)
        atomistic_features = torch.split(atomistic_features, n_atoms)

        # get text embeddings
        embed_layer = self.llm.get_input_embeddings()
        text_embeds = [embed_layer(sample_input_ids) for sample_input_ids in input_ids]  # list[(seq, hidden)]

        # combine atomistic and text features
        new_embeds, new_labels, new_attention_mask = self._merge_embeddings(
            text_embeds, atomistic_features, input_ids, labels, attention_mask
        )

        # forward LLM pass
        outputs = self.llm(
            inputs_embeds=new_embeds,
            attention_mask=new_attention_mask,
            labels=new_labels,
            return_dict=True,
        )

        return outputs # loss = outputs.loss

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
