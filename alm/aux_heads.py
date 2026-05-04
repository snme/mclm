"""Auxiliary supervision heads for AtomsMapper output.

Phase 0 diagnostic showed AtomsMapper output (after run5_contrastive at step=5000)
encodes essentially no structural info: linear probe to composition has val recall
0.001, UMAP shows no clusters by element/oxide-presence/transition-metal-presence.
The contrastive loss decorrelated outputs but along composition-irrelevant axes.

This module adds direct supervision: a small head predicts a structural fingerprint
of the ground-truth target structure from `am_out`. Aux loss flows back through
AtomsMapper, forcing its output to encode the target signal. The downstream
cond_adapt/mixin learns to translate the encoded fingerprint into useful diffusion
conditioning.

Three target families, swappable via CLI:

  - `composition` — multi-hot over Z=1..100. BCE loss. Cheap (computed at
    Stage3aDataset __init__ from atoms_struct.elements). Drives presence
    encoding only — saturated at recall ~1.000 by run9 step=4500, but the
    count probe at the same checkpoint hits only 17% reduced-formula match,
    showing AtomsMapper does NOT encode stoichiometry.
  - `composition_count` — integer count vector per Z=1..100 (clamped to
    MAX_COUNT=20). Factored loss: BCE on (count > 0) presence + CE on the
    exact count given presence. Drop-in replacement for `composition`. The
    count branch is the new pressure that targets the 17% reduced-formula gap.
  - `orbv3_mean`  — mean of OrbV3 per-atom features (256-d). MSE loss. Richer
    signal: encodes composition AND local environments. Requires precomputed
    orbv3_means.bin (one-shot pass via helper_scripts/precompute_orbv3_means.py).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from composition_utils import MAX_COUNT, N_ELEMENTS


class AuxHead(nn.Module):
    """Base class. Subclasses set target_kind / target_dim and implement forward/loss."""
    target_kind: str = ""
    target_dim: int = 0

    def forward(self, am_out: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def metrics(self, pred: torch.Tensor, target: torch.Tensor) -> dict:
        """Diagnostic scalars for W&B (precision/recall, cosine sim, etc.)."""
        return {}


class CompositionHead(AuxHead):
    """Predicts multi-hot over Z=1..100. BCEWithLogits loss with pos_weight to
    rebalance against the heavy class imbalance (~3 positives / 100 classes).

    Without pos_weight, BCE has a stable trivial minimum at "predict all logits
    very negative" — sigmoid ≈ 0.05 across all classes, average BCE ≈ 0.14, and
    threshold-0.5 precision/recall both stay at 0 forever (we observed exactly
    this: aux loss plateaued at 0.12-0.14 with P=R=0 from step 50 onward).

    pos_weight = (1 - p_pos) / p_pos ≈ (1 - 0.03) / 0.03 ≈ 32 makes positive
    misclassifications ~32× more expensive than negative ones, pulling the
    optimizer toward predicting positives.

    target shape: (B, 100) float32 in {0, 1}. Slot Z-1 is 1 iff element with
    atomic number Z is present in the target structure.
    """
    target_kind = "composition"
    target_dim = 100

    def __init__(self, in_dim: int = 512, pos_weight: float = 32.0):
        super().__init__()
        self.proj = nn.Linear(in_dim, self.target_dim)
        # Buffer (not parameter) so it moves with .to(device) but isn't trained.
        self.register_buffer("pos_weight", torch.full((self.target_dim,), pos_weight))

    def forward(self, am_out: torch.Tensor) -> torch.Tensor:
        return self.proj(am_out)  # (B, 100) logits

    def loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(
            pred, target.float(), pos_weight=self.pos_weight
        )

    def metrics(self, pred: torch.Tensor, target: torch.Tensor) -> dict:
        # Micro-averaged precision/recall at threshold 0.5 (single-batch)
        with torch.no_grad():
            pred_bin = torch.sigmoid(pred) > 0.5
            tgt = target.bool()
            tp = (pred_bin & tgt).sum().float()
            fp = (pred_bin & ~tgt).sum().float()
            fn = (~pred_bin & tgt).sum().float()
            return {
                "precision": tp / (tp + fp + 1e-9),
                "recall":    tp / (tp + fn + 1e-9),
            }


class CompositionCountHead(AuxHead):
    """Predicts exact per-element atom counts. Factored loss:

      L = BCE(presence_logits, count > 0)
        + count_lambda * CE(count_logits[positive_slots], count[positive_slots])

    The presence branch replicates `CompositionHead`'s behavior exactly (same
    BCE with pos_weight=32) — so AtomsMapper still gets the run9-validated
    "saturate which elements are present" signal. The count branch is the new
    pressure: among present elements, predict the exact integer count
    (clamped to MAX_COUNT=20). Counts are decoded by per-slot cross-entropy
    over MAX_COUNT+1 bins; loss is masked to slots where the true count is
    positive, so the negative class doesn't dominate the gradient.

    Target shape: (B, N_ELEMENTS) float32 in {0, 1, ..., MAX_COUNT}. Slot Z-1
    holds the integer count (clamped to MAX_COUNT) of element Z in the target
    structure. Count-aware target replaces the multi-hot presence target —
    presence is recovered as `(target > 0)` inside this head.

    LOC of count head: 100 * (MAX_COUNT+1) = 2100 logits + 100 presence logits.
    """
    target_kind = "composition_count"
    target_dim = N_ELEMENTS  # the (B, N_ELEMENTS) count target shape

    def __init__(self, in_dim: int = 512, pos_weight: float = 32.0,
                 count_lambda: float = 1.0, max_count: int = MAX_COUNT):
        super().__init__()
        self.max_count = max_count
        self.count_lambda = count_lambda
        self.presence = nn.Linear(in_dim, N_ELEMENTS)
        self.count = nn.Linear(in_dim, N_ELEMENTS * (max_count + 1))
        self.register_buffer("pos_weight", torch.full((N_ELEMENTS,), pos_weight))

    def forward(self, am_out: torch.Tensor) -> dict:
        # Returning a dict so loss/metrics can read the two branches separately;
        # the trainer treats `pred` opaquely and just passes it back to .loss().
        return {
            "presence": self.presence(am_out),                              # (B, 100)
            "count": self.count(am_out).view(-1, N_ELEMENTS, self.max_count + 1),
        }

    def loss(self, pred: dict, target: torch.Tensor) -> torch.Tensor:
        # target shape: (B, N_ELEMENTS) float32 holding integer counts in [0, MAX].
        # Recover presence and integer counts from the unified target tensor.
        counts_long = target.long().clamp_(0, self.max_count)
        presence_target = (counts_long > 0).float()

        # Branch 1: presence BCE (same as CompositionHead).
        bce = F.binary_cross_entropy_with_logits(
            pred["presence"], presence_target, pos_weight=self.pos_weight,
        )

        # Branch 2: per-slot CE on positive slots only. If the entire batch
        # somehow has zero positive slots, skip CE (preserves graph correctness).
        pos_mask = (counts_long > 0)
        if pos_mask.any():
            count_logits_flat = pred["count"][pos_mask]            # (n_pos, MAX+1)
            count_target_flat = counts_long[pos_mask]               # (n_pos,)
            ce = F.cross_entropy(count_logits_flat, count_target_flat)
        else:
            ce = torch.tensor(0.0, device=bce.device, dtype=bce.dtype)

        return bce + self.count_lambda * ce

    def metrics(self, pred: dict, target: torch.Tensor) -> dict:
        with torch.no_grad():
            counts_long = target.long().clamp_(0, self.max_count)
            presence_target = (counts_long > 0)

            # Presence P/R at threshold 0.5 (same definition as CompositionHead)
            pres_pred = torch.sigmoid(pred["presence"]) > 0.5
            tp = (pres_pred & presence_target).sum().float()
            fp = (pres_pred & ~presence_target).sum().float()
            fn = (~pres_pred & presence_target).sum().float()

            # Count exact-match accuracy on positive slots
            count_pred = pred["count"].argmax(dim=-1)              # (B, N_ELEMENTS)
            pos_mask = presence_target
            if pos_mask.any():
                count_exact = (count_pred[pos_mask] == counts_long[pos_mask]).float().mean()
                count_mae = (count_pred[pos_mask] - counts_long[pos_mask]).abs().float().mean()
            else:
                count_exact = torch.tensor(0.0, device=tp.device)
                count_mae = torch.tensor(0.0, device=tp.device)

            # Reduced-formula whole-composition match: per-sample accuracy
            # checks ALL slots predict the right count (positive or zero).
            whole_exact = (count_pred == counts_long).all(dim=-1).float().mean()

            return {
                "precision": tp / (tp + fp + 1e-9),
                "recall":    tp / (tp + fn + 1e-9),
                "count_exact_pos":   count_exact,
                "count_mae_pos":     count_mae,
                "whole_exact":       whole_exact,
            }


class OrbV3MeanHead(AuxHead):
    """Predicts mean of OrbV3 per-atom features over the target structure. MSE loss.

    target shape: (B, 256) float32. Encodes composition + local environments.
    """
    target_kind = "orbv3_mean"
    target_dim = 256

    def __init__(self, in_dim: int = 512, mid_dim: int = 512):
        super().__init__()
        # Slightly more capacity than Linear since the target is denser/structured.
        self.proj = nn.Sequential(
            nn.Linear(in_dim, mid_dim),
            nn.GELU(),
            nn.Linear(mid_dim, self.target_dim),
        )

    def forward(self, am_out: torch.Tensor) -> torch.Tensor:
        return self.proj(am_out)  # (B, 256)

    def loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(pred, target.float())

    def metrics(self, pred: torch.Tensor, target: torch.Tensor) -> dict:
        with torch.no_grad():
            cs = F.cosine_similarity(pred, target.float(), dim=-1).mean()
            return {"cosine_sim": cs}


def build_aux_head(kind: str | None, in_dim: int = 512,
                   count_lambda: float = 1.0) -> AuxHead | None:
    """Factory. Returns None when no aux loss is requested.

    `count_lambda` is only consumed by the `composition_count` head — it weights
    the count-CE branch relative to the presence-BCE branch. 1.0 is a safe
    starting value (matches the magnitude regime of the BCE contribution).
    """
    if kind in (None, "none", ""):
        return None
    if kind == "composition":
        return CompositionHead(in_dim=in_dim)
    if kind == "composition_count":
        return CompositionCountHead(in_dim=in_dim, count_lambda=count_lambda)
    if kind == "orbv3_mean":
        return OrbV3MeanHead(in_dim=in_dim)
    raise ValueError(f"unknown aux_target_kind: {kind!r}; "
                     f"expected one of: composition, composition_count, orbv3_mean, none")
