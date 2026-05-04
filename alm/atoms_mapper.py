"""AtomsMapper: maps Qwen3 hidden states at output-side [atoms_{i}] token positions
into MatterGen's `alm_embedding` adapter conditioning vector.

Architecture: per-position projection + mean-pool over K. The K=8 [atoms_i] tokens
are causally attended inside Qwen3 already, so each position's hidden state is its
own "summary" of the upstream prompt. AtomsMapper projects each K=8 summary into
the MatterGen hidden_dim (512), then averages them into a single conditioning
vector. This matches the LLaVA-projector pattern: simple per-token MLP + aggregate.

Plugged in two places:
1. Training (train_stage3a.py): ALM forward extracts (B, K, 4096) hidden states at
   [atoms_{i}] positions; these are detached and passed to diffusion_module.calc_loss
   via ChemGraph["alm_embedding"]. PropertyEmbedding.forward calls AtomsMapper here,
   and gradients flow back through AtomsMapper from L_diff.
2. Inference: generate_stage3a.py-style scripts call AtomsMapper(hidden) before
   handing to mattergen sampler as `properties_to_condition_on={"alm_embedding": v}`.

MatterGen wiring: `_target_: atoms_mapper.AtomsMapper` in alm_embedding.yaml.
PYTHONPATH=/home/sathyae/mclm/alm must be set (alm/ is not a package).

Input shape: accepts both (B, K*hidden_dim) [flat from PropertyEmbedding's data
collation] and (B, K, hidden_dim) [direct]. Reshapes flat → (B, K, hidden_dim).
Output shape: (B, out_dim=512) — MatterGen score-network hidden_dim.

Params: ~9M (Linear(4096→2048) + Linear(2048→512)). 15× smaller than the original
concat-style design (136M), in the LLaVA-projector range and well below GILL/
DreamLLM-class adapters (50-150M).
"""
import torch
import torch.nn as nn


class AtomsMapper(nn.Module):
    def __init__(self, hidden_dim: int = 4096, mid_dim: int = 2048, out_dim: int = 512, K: int = 8):
        super().__init__()
        self.K = K
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        # Per-position MLP: applied independently to each of K [atoms_i] positions.
        # The LLM's upstream causal attention has already done cross-position mixing
        # — this just projects each summary to MatterGen's conditioning space.
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, mid_dim),
            nn.GELU(),
            nn.Linear(mid_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Accept (B, K*hidden_dim) flat (from ChemGraph collation of pre-flattened
        # alm_embedding) or (B, K, hidden_dim) direct.
        if x.dim() == 2:
            B = x.size(0)
            x = x.view(B, self.K, self.hidden_dim)
        # Per-position projection then mean-pool over K positions.
        return self.proj(x).mean(dim=1)
