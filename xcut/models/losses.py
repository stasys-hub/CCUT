"""Model-agnostic loss functions for Hi-C contact matrix enhancement.

All losses accept batched tensors of shape (B, 1, H, W) — predictions and
targets from any model architecture.

Generic losses:
    structure_consistency_loss  — local patch correlation
    HiCLoss                    — L1 + symmetry + smoothness (nn.Module)

Biologically-motivated losses:
    insulation_loss            — TAD boundary preservation (VEHiCLE-style)
    distance_decay_loss        — genomic distance decay profile (HiCRep-inspired)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Generic structural losses
# ---------------------------------------------------------------------------


def structure_consistency_loss(pred, target, window_size=7):
    """Local structural pattern loss using patch-wise correlation.

    Extracts all window_size x window_size patches, normalizes them, and
    computes cosine similarity. Penalizes structural dissimilarity.

    Args:
        pred: Predicted contact matrix (B, 1, H, W).
        target: Target contact matrix (B, 1, H, W).
        window_size: Patch size for local comparison.

    Returns:
        Scalar loss in [0, 2], where 0 = perfect correlation.
    """
    patches_pred = F.unfold(pred, kernel_size=window_size)
    patches_target = F.unfold(target, kernel_size=window_size)

    norm_pred = F.normalize(patches_pred, dim=1, eps=1e-8)
    norm_target = F.normalize(patches_target, dim=1, eps=1e-8)

    correlation = torch.sum(norm_pred * norm_target, dim=1)

    # Mask out patches where either pred or target had zero norm (sparse regions)
    pred_norms = patches_pred.norm(dim=1)
    target_norms = patches_target.norm(dim=1)
    valid = (pred_norms > 1e-8) & (target_norms > 1e-8)

    if valid.any():
        return 1 - correlation[valid].mean()
    return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)


class HiCLoss(nn.Module):
    """Combined loss for Hi-C contact matrix enhancement.

    Args:
        lambda_sym: Weight for symmetry loss. Only use > 0 when training on
            diagonal patches exclusively (symmetric coordinates). For off-diagonal
            patches, this loss is meaningless. Symmetrize predictions at inference
            regardless: pred = (pred + pred.transpose(-2, -1)) / 2
        lambda_smooth: Weight for smoothness/TV loss.
    """

    def __init__(self, lambda_sym=0.0, lambda_smooth=0.1):
        super().__init__()
        self.base_loss = nn.L1Loss()
        self.lambda_sym = lambda_sym
        self.lambda_smooth = lambda_smooth

    def forward(self, pred, target):
        base_loss = self.base_loss(pred, target)

        sym_loss = self.base_loss(pred, pred.transpose(2, 3))

        smooth_loss = torch.mean(
            torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :])
        ) + torch.mean(torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1]))

        return base_loss + self.lambda_sym * sym_loss + self.lambda_smooth * smooth_loss


# ---------------------------------------------------------------------------
# Biologically-motivated losses for Hi-C contact matrices
# ---------------------------------------------------------------------------


def insulation_loss(pred, target, window_size=10, delta_window=2):
    """Insulation score loss -- penalizes incorrect TAD boundary placement.

    Based on the insulation score method from VEHiCLE (Highsmith & Cheng, 2021,
    Scientific Reports). The insulation score detects TAD boundaries by measuring
    local contact depletion along the diagonal. The delta (finite derivative) of
    the insulation profile identifies boundaries as zero-crossings.

    This loss compares the delta vectors of pred and target, directly penalizing
    the model for shifting, missing, or hallucinating TAD boundaries.

    Reference:
        Crane et al., "Condensin-driven remodelling of X chromosome topology
        during dosage compensation", Nature 2015 -- original insulation score.
        Highsmith & Cheng, "VEHiCLE: a Variationally Encoded Hi-C Loss
        Enhancement algorithm", Scientific Reports 2021 -- insulation as loss.

    Args:
        pred: Predicted contact matrix (B, 1, H, W).
        target: Target contact matrix (B, 1, H, W).
        window_size: Size of the square window slid along the diagonal to
            compute insulation scores (in bins). At 50kb resolution,
            10 bins = 500kb, a reasonable TAD-boundary detection scale.
        delta_window: Number of bins upstream/downstream for the finite
            derivative of the insulation profile.

    Returns:
        Scalar loss (L1 between pred and target delta vectors).
    """
    def _insulation_profile(mat):
        # mat: (B, 1, H, W) -> squeeze to (B, H, W)
        m = mat[:, 0]
        B, H, W = m.shape
        n = min(H, W)

        scores = []
        for i in range(window_size, n - window_size):
            # Sum the window_size x window_size submatrix straddling the diagonal
            # Upper-left corner: (i - window_size, i)
            # This captures contacts crossing position i
            block = m[:, i - window_size:i, i:i + window_size]
            scores.append(block.sum(dim=(-2, -1)))

        # (B, n_positions)
        profile = torch.stack(scores, dim=1)

        # Log2-normalize relative to mean (avoid log of zero)
        mean_score = profile.mean(dim=1, keepdim=True).clamp(min=1e-8)
        profile = torch.log2((profile / mean_score).clamp(min=1e-8))

        return profile

    def _delta(profile):
        # Finite derivative: mean(downstream) - mean(upstream)
        # profile: (B, L)
        B, L = profile.shape
        if L <= 2 * delta_window:
            return profile
        deltas = []
        for i in range(delta_window, L - delta_window):
            upstream = profile[:, i - delta_window:i].mean(dim=1)
            downstream = profile[:, i:i + delta_window].mean(dim=1)
            deltas.append(downstream - upstream)
        return torch.stack(deltas, dim=1)

    pred_profile = _insulation_profile(pred)
    target_profile = _insulation_profile(target)

    pred_delta = _delta(pred_profile)
    target_delta = _delta(target_profile)

    return F.l1_loss(pred_delta, target_delta)


def distance_decay_loss(pred, target):
    """Distance-dependent decay loss -- preserves genomic distance signal.

    Hi-C contact frequency decays with genomic distance following a power law.
    This is the most fundamental property of contact matrices. Standard pixel
    losses (L1/MSE) weight all positions equally, but biologically the
    near-diagonal high-signal region matters far more than distant low-signal
    off-diagonal entries.

    This loss computes per-diagonal (stratum) mean values for both pred and
    target, then penalizes differences. This ensures the model preserves the
    correct decay profile rather than smearing signal uniformly.

    Conceptually related to the stratum-adjusted correlation (SCC) from HiCRep
    (Yang et al., Genome Research 2017), which also operates per-diagonal,
    but simpler and more stable as a loss function.

    Args:
        pred: Predicted contact matrix (B, 1, H, W).
        target: Target contact matrix (B, 1, H, W).

    Returns:
        Scalar loss (weighted L1 between per-diagonal mean profiles).
    """
    p = pred[:, 0]   # (B, H, W)
    t = target[:, 0]
    B, H, W = p.shape
    n = min(H, W)

    pred_means = []
    target_means = []

    for d in range(n):
        # Extract diagonal d (genomic distance = d bins)
        pred_diag = torch.diagonal(p, offset=d, dim1=1, dim2=2)   # (B, n-d)
        target_diag = torch.diagonal(t, offset=d, dim1=1, dim2=2)
        pred_means.append(pred_diag.mean(dim=1))
        target_means.append(target_diag.mean(dim=1))

    # (B, n_diagonals)
    pred_decay = torch.stack(pred_means, dim=1)
    target_decay = torch.stack(target_means, dim=1)

    # Weight by inverse distance: near-diagonal strata matter more.
    # weights[0] = 1.0 (main diagonal), decaying to ~0.01 for furthest.
    weights = 1.0 / (1.0 + torch.arange(n, device=pred.device, dtype=pred.dtype))
    weights = weights / weights.sum()  # normalize

    diff = torch.abs(pred_decay - target_decay)  # (B, n)
    weighted_diff = (diff * weights.unsqueeze(0)).sum(dim=1)  # (B,)

    return weighted_diff.mean()
