"""Image quality metrics for contact matrices (NumPy + PyTorch).

NumPy versions accept plain 2D arrays (H, W):
    lr = np.squeeze(dataset[idx]['lr'])
    hr = np.squeeze(dataset[idx]['hr'])
    print(ssim(lr, hr), psnr(lr, hr), mse(lr, hr), mae(lr, hr))

Torch versions accept batched tensors (B, C, H, W) or (B, 1, H, W):
    print(ssim_t(lr_batch, hr_batch), psnr_t(lr_batch, hr_batch))
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import uniform_filter


def mse(a: np.ndarray, b: np.ndarray) -> float:
    """Mean squared error."""
    return float(np.mean((a - b) ** 2))


def mae(a: np.ndarray, b: np.ndarray) -> float:
    """Mean absolute error."""
    return float(np.mean(np.abs(a - b)))


def psnr(a: np.ndarray, b: np.ndarray, data_range: float | None = None) -> float:
    """Peak signal-to-noise ratio.

    Args:
        a, b: Arrays to compare.
        data_range: Value range. If None, uses max(a.max(), b.max()) - min(a.min(), b.min()).
    """
    err = mse(a, b)
    if err == 0:
        return float("inf")
    if data_range is None:
        data_range = max(a.max(), b.max()) - min(a.min(), b.min())
    if data_range == 0:
        return 0.0
    return float(10 * np.log10(data_range**2 / err))


def ssim(
    a: np.ndarray,
    b: np.ndarray,
    window_size: int = 11,
    data_range: float | None = None,
) -> float:
    """Structural similarity index (SSIM).

    Uses uniform filter (box filter) for speed. Matches scikit-image behavior
    closely but without the dependency overhead.

    Args:
        a, b: 2D arrays to compare.
        window_size: Local window size for statistics.
        data_range: Value range. If None, inferred from data.
    """
    a = a.astype(np.float64)
    b = b.astype(np.float64)

    if data_range is None:
        data_range = max(a.max(), b.max()) - min(a.min(), b.min())
    if data_range == 0:
        return 1.0

    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    mu_a = uniform_filter(a, size=window_size)
    mu_b = uniform_filter(b, size=window_size)

    mu_a_sq = mu_a * mu_a
    mu_b_sq = mu_b * mu_b
    mu_ab = mu_a * mu_b

    sigma_a_sq = uniform_filter(a * a, size=window_size) - mu_a_sq
    sigma_b_sq = uniform_filter(b * b, size=window_size) - mu_b_sq
    sigma_ab = uniform_filter(a * b, size=window_size) - mu_ab

    num = (2 * mu_ab + C1) * (2 * sigma_ab + C2)
    den = (mu_a_sq + mu_b_sq + C1) * (sigma_a_sq + sigma_b_sq + C2)

    return float(np.mean(num / den))


# ---------------------------------------------------------------------------
# PyTorch versions — batched tensors (B, C, H, W)
# ---------------------------------------------------------------------------


def _gaussian_kernel_2d(size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-(coords[:, None] ** 2 + coords[None, :] ** 2) / (2 * sigma**2))
    g /= g.sum()
    return g.unsqueeze(0).unsqueeze(0)


def mse_t(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Mean squared error (batched, returns scalar tensor)."""
    return F.mse_loss(a, b)


def mae_t(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Mean absolute error (batched, returns scalar tensor)."""
    return F.l1_loss(a, b)


def psnr_t(a: torch.Tensor, b: torch.Tensor, data_range: float | None = None) -> torch.Tensor:
    """Peak signal-to-noise ratio (batched, returns scalar tensor).

    Args:
        a, b: Tensors of shape (B, C, H, W).
        data_range: Value range. If None, inferred from data (matches numpy psnr).
    """
    err = F.mse_loss(a, b)
    if err == 0:
        return torch.tensor(float("inf"))
    if data_range is None:
        data_range = float(max(a.max(), b.max()) - min(a.min(), b.min()))
    if data_range == 0:
        return torch.tensor(0.0)
    return 10 * torch.log10(torch.tensor(data_range**2) / err)


def ssim_t(
    a: torch.Tensor,
    b: torch.Tensor,
    window_size: int = 11,
    data_range: float | None = None,
) -> torch.Tensor:
    """Structural similarity index (batched, returns scalar tensor).

    Args:
        a, b: Tensors of shape (B, C, H, W).
        window_size: Gaussian kernel size.
        data_range: Value range. If None, inferred from data (matches numpy ssim).
    """
    if data_range is None:
        data_range = float(max(a.max(), b.max()) - min(a.min(), b.min()))
    if data_range == 0:
        return torch.tensor(1.0, device=a.device)

    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    kernel = _gaussian_kernel_2d(window_size).to(a.device)
    kernel = kernel.repeat(a.shape[1], 1, 1, 1)
    pad = window_size // 2

    mu_a = F.conv2d(a, kernel, padding=pad, groups=a.shape[1])
    mu_b = F.conv2d(b, kernel, padding=pad, groups=a.shape[1])

    mu_a_sq = mu_a * mu_a
    mu_b_sq = mu_b * mu_b
    mu_ab = mu_a * mu_b

    sigma_a_sq = F.conv2d(a * a, kernel, padding=pad, groups=a.shape[1]) - mu_a_sq
    sigma_b_sq = F.conv2d(b * b, kernel, padding=pad, groups=a.shape[1]) - mu_b_sq
    sigma_ab = F.conv2d(a * b, kernel, padding=pad, groups=a.shape[1]) - mu_ab

    num = (2 * mu_ab + C1) * (2 * sigma_ab + C2)
    den = (mu_a_sq + mu_b_sq + C1) * (sigma_a_sq + sigma_b_sq + C2)

    return (num / den).mean()
