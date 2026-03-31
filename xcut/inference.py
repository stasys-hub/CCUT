"""Inference engine for creating enhanced cooler files from trained models.

Provides the full pipeline: load checkpoint → run model on LR patches →
inverse-transform predictions to integer counts → stitch into a cooler file.

Uses Hann-window blending for seamless patch stitching (no grid artifacts).

Currently supports HINet-GAN. Other model types can be added later.

Usage:
    # Recommended: provide HR cooler for correct inverse scaling
    from xcut.inference import create_enhanced_cooler

    create_enhanced_cooler(
        checkpoint_path="runs/ablation_clipped_16x_50k/best_model.pth",
        config_path="runs/ablation_clipped_16x_50k/config.yaml",
        lr_cooler_path="data/new_sample.cool",
        hr_cooler_path="data/reference_hr.mcool::/resolutions/50000",
        output_path="output/enhanced.cool",
        chromosomes=["chr21", "chr22"],
    )
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cooler
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.sparse import coo_matrix, triu as sparse_triu
from tqdm import tqdm

from xcut.config import RunConfig, resolve_transforms
from xcut.data.coordinates import CoordinateGenerator, WindowConfig
from xcut.data.transforms import BaseTransform


# ---------------------------------------------------------------------------
# Hann window for blending
# ---------------------------------------------------------------------------


def _make_hann_window(size: int) -> np.ndarray:
    """Create a 2D Hann window for smooth patch blending.

    Args:
        size: Window size (square).

    Returns:
        2D array of shape (size, size) with values in (0, 1].
    """
    hann_1d = np.hanning(size + 2)[1:-1]  # remove the zeros at endpoints
    hann_2d = np.outer(hann_1d, hann_1d)
    return hann_2d.astype(np.float64)


# ---------------------------------------------------------------------------
# Inverse transforms
# ---------------------------------------------------------------------------


def _find_transform_params(transforms: list[BaseTransform]) -> dict[str, Any]:
    """Extract inverse-transform parameters from the forward pipeline.

    Inspects the transform list to determine:
    - Whether log1p was applied (ClipLogByPercentile or LogTransform)
    - The percentile used for clipping/normalization
    - Whether scaling is by percentile or chrom_max

    Args:
        transforms: List of forward transforms from config.

    Returns:
        Dict with keys: 'mode' ('log1p' | 'minmax'), 'percentile' (float | None).
    """
    from xcut.data.transforms import (
        ClipByPercentile,
        ClipLogByPercentile,
        DivideByMax,
        LogTransform,
        MinMaxNormalize,
    )

    mode = "minmax"
    percentile = None

    for t in transforms:
        if isinstance(t, ClipLogByPercentile):
            mode = "log1p"
            percentile = t.percentile
        elif isinstance(t, LogTransform):
            mode = "log1p"
        elif isinstance(t, (ClipByPercentile, DivideByMax, MinMaxNormalize)):
            pct = getattr(t, "percentile", None)
            if pct is not None:
                percentile = pct

    return {"mode": mode, "percentile": percentile}


def inverse_transform(
    matrix: np.ndarray,
    mode: str,
    percentile: float | None,
    chrom_stats: dict,
) -> np.ndarray:
    """Invert the forward transform back to integer counts.

    Args:
        matrix: Predicted matrix in [0, 1].
        mode: 'log1p' or 'minmax'.
        percentile: Percentile used during forward transform (e.g. 99.95).
        chrom_stats: Chromosome statistics dict with 'max' and 'percentiles'.
            Should be from the HR cooler (recommended) for correct count scale,
            or from the LR cooler as fallback.

    Returns:
        Integer count matrix.
    """
    chrom_max = chrom_stats["max"]

    # Determine the scale factor
    clip_val = None
    if percentile is not None:
        pctiles = chrom_stats.get("percentiles", {})
        clip_val = pctiles.get(str(percentile)) or pctiles.get(percentile)
        if clip_val is not None:
            clip_val = float(clip_val)

    if mode == "log1p":
        # Forward: raw → log(1+x) → clip(0, log(1+clip_val)) → / log(1+clip_val) → [0,1]
        # Inverse: [0,1] → * ceiling → expm1 → round → int
        if clip_val is not None:
            ceiling = float(np.log(1.0 + clip_val))
        else:
            ceiling = float(np.log(1.0 + chrom_max))

        if ceiling <= 0:
            return np.zeros_like(matrix, dtype=np.int32)

        log_counts = matrix * ceiling
        counts = np.expm1(log_counts)

    elif mode == "minmax":
        # Forward: raw → clip(0, clip_val) → / clip_val → [0,1]
        # Inverse: [0,1] → * clip_val → round → int
        scale = clip_val if clip_val is not None else chrom_max
        if scale <= 0:
            return np.zeros_like(matrix, dtype=np.int32)
        counts = matrix * scale

    else:
        raise ValueError(f"Unknown inverse mode: {mode!r}")

    counts = np.rint(counts).clip(0).astype(np.int32)
    return counts


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_generator(
    checkpoint_path: str | Path,
    config: RunConfig,
    device: torch.device,
) -> nn.Module:
    """Load a trained HINet generator from checkpoint.

    Args:
        checkpoint_path: Path to .pth checkpoint file.
        config: RunConfig for model architecture params.
        device: Target device.

    Returns:
        Generator model in eval mode.
    """
    from xcut.registry import build_model

    models = build_model(config.model, device)
    generator = models["generator"]

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    generator.load_state_dict(ckpt["generator_state_dict"])
    generator.eval()

    n_params = sum(p.numel() for p in generator.parameters())
    print(f"Loaded generator: {n_params / 1e6:.2f}M params, epoch {ckpt.get('epoch', '?')}")
    if "ssim_score" in ckpt:
        print(f"  Best SSIM: {ckpt['ssim_score']:.4f}, PSNR: {ckpt.get('psnr_score', 0):.4f}")

    return generator


# ---------------------------------------------------------------------------
# Chromosome statistics
# ---------------------------------------------------------------------------


def _compute_stats(
    cooler_path: str,
    chromosomes: list[str],
    window_cfg: WindowConfig,
    transforms: list[BaseTransform],
    n_processes: int,
    nonzero_percentile: bool = False,
) -> dict:
    """Compute chromosome statistics from a cooler file.

    Args:
        cooler_path: Path to cooler file.
        chromosomes: Chromosomes to process.
        window_cfg: Window configuration.
        transforms: Transforms (for percentile detection).
        n_processes: Number of parallel processes.
        nonzero_percentile: If True, compute percentiles over non-zero values only.

    Returns:
        Dict mapping chromosome names to stats dicts.
    """
    from xcut.data.datasets import _extract_percentiles_from_transforms

    coor_gen = CoordinateGenerator(
        window_cfg,
        n_processes=n_processes,
        nonzero_percentile=nonzero_percentile,
    )
    coor_gen.set_cooler(cooler_path)
    coor_gen.chromosomes = chromosomes

    detected_pcts = _extract_percentiles_from_transforms(transforms)
    return coor_gen.compute_chrom_stats(
        percentiles=sorted(detected_pcts, reverse=True) if detected_pcts else None,
        use_cache=True,
    )


# ---------------------------------------------------------------------------
# Chromosome prediction with Hann-window stitching
# ---------------------------------------------------------------------------


@torch.no_grad()
def predict_chromosome(
    generator: nn.Module,
    lr_cooler: cooler.Cooler,
    chrom: str,
    transforms: list[BaseTransform],
    inv_params: dict[str, Any],
    lr_chrom_stats: dict,
    inv_chrom_stats: dict,
    window_size: int,
    resolution: int,
    step: float,
    max_distance: int,
    device: torch.device,
    batch_size: int = 16,
) -> pd.DataFrame | None:
    """Run inference on all patches of a chromosome and stitch into a pixel table.

    Uses Hann-window blending for seamless stitching.

    Args:
        generator: Trained generator in eval mode.
        lr_cooler: Low-resolution cooler.
        chrom: Chromosome name (e.g. 'chr1').
        transforms: Forward transform pipeline.
        inv_params: Inverse transform parameters (mode, percentile).
        lr_chrom_stats: LR chromosome statistics (for forward transform).
        inv_chrom_stats: Chromosome statistics for inverse transform.
            Use HR stats (recommended) for correct count scale, or LR as fallback.
        window_size: Patch size in bins.
        resolution: Base pair resolution per bin.
        step: Window overlap fraction (0.5 = 50% overlap).
        max_distance: Max distance from diagonal in bp.
        device: Compute device.
        batch_size: Inference batch size.

    Returns:
        DataFrame with columns (bin1_id, bin2_id, count) or None if no patches.
    """
    # Build context for forward transforms using LR stats
    chrom_max = lr_chrom_stats["max"]
    context: dict[str, Any] = {"chrom": chrom, "chrom_max": chrom_max}
    pctiles = lr_chrom_stats.get("percentiles", {})
    for k, v in pctiles.items():
        context[f"chrom_{k}"] = float(v)

    # Generate patch coordinates
    chrom_size_bp = lr_cooler.chromsizes[chrom]
    exact_window_bp = window_size * resolution
    step_size_bp = int(exact_window_bp * step)

    coords: list[tuple[int, int, int, int]] = []
    starts1 = list(range(0, chrom_size_bp - exact_window_bp + 1, step_size_bp))
    last_possible1 = (chrom_size_bp - exact_window_bp) // resolution * resolution
    if last_possible1 > 0 and (not starts1 or starts1[-1] < last_possible1):
        starts1.append(last_possible1)

    for start1 in starts1:
        stop1 = start1 + exact_window_bp
        start2_min = max(0, start1 - max_distance)
        start2_max = min(chrom_size_bp - exact_window_bp + 1, start1 + max_distance)
        starts2 = list(range(start2_min, start2_max, step_size_bp))
        last_possible2 = min(chrom_size_bp - exact_window_bp, start2_max - 1) // resolution * resolution
        if last_possible2 >= start2_min and (not starts2 or starts2[-1] < last_possible2):
            starts2.append(last_possible2)

        for start2 in starts2:
            stop2 = start2 + exact_window_bp
            coords.append((start1, stop1, start2, stop2))

    if not coords:
        return None

    # Log inverse stats source info
    inv_max = inv_chrom_stats["max"]
    inv_pctiles = inv_chrom_stats.get("percentiles", {})
    inv_clip = None
    if inv_params["percentile"] is not None:
        inv_clip = inv_pctiles.get(str(inv_params["percentile"])) or inv_pctiles.get(
            inv_params["percentile"]
        )
    print(
        f"  {chrom}: {len(coords)} patches, "
        f"lr_max={chrom_max:.1f}, inv_max={inv_max:.1f}, inv_clip={inv_clip}"
    )

    # Fetch and transform LR patches
    lr_patches: list[np.ndarray] = []
    patch_coords: list[tuple[int, int, int, int]] = []

    for start1, stop1, start2, stop2 in tqdm(
        coords, desc=f"  Fetching {chrom}", leave=False
    ):
        region1 = f"{chrom}:{start1}-{stop1}"
        region2 = f"{chrom}:{start2}-{stop2}"

        lr_mat = lr_cooler.matrix(balance=False).fetch(region1, region2)
        lr_mat = lr_mat.astype(np.float32)

        # Apply forward transforms using LR context
        for t in transforms:
            lr_mat = t(lr_mat, context)

        # Pad if needed (boundary patches)
        h, w = lr_mat.shape
        if h < window_size or w < window_size:
            padded = np.zeros((window_size, window_size), dtype=np.float32)
            padded[:h, :w] = lr_mat
            lr_mat = padded

        lr_patches.append(lr_mat)
        patch_coords.append((start1, stop1, start2, stop2))

    # Run inference in batches
    predictions: list[np.ndarray] = []
    for i in tqdm(
        range(0, len(lr_patches), batch_size),
        desc=f"  Predicting {chrom}",
        leave=False,
    ):
        batch = lr_patches[i : i + batch_size]
        batch_tensor = torch.tensor(
            np.stack([p[np.newaxis] for p in batch]),
            dtype=torch.float32,
            device=device,
        )
        pred_tensor = generator(batch_tensor)[1]  # stage 2 output
        pred_tensor = pred_tensor.clamp(0.0, 1.0)
        pred_np = pred_tensor.cpu().numpy()
        for j in range(pred_np.shape[0]):
            predictions.append(pred_np[j, 0])

    # Stitch with Hann-window blending
    n_bins = chrom_size_bp // resolution
    accumulator = np.zeros((n_bins, n_bins), dtype=np.float64)
    weight_map = np.zeros((n_bins, n_bins), dtype=np.float64)
    hann = _make_hann_window(window_size)

    for idx, (start1, stop1, start2, stop2) in enumerate(patch_coords):
        pred = predictions[idx]

        # Enforce symmetry on diagonal patches
        if start1 == start2:
            pred = (pred + pred.T) / 2.0

        # Actual patch size (may be smaller at boundaries)
        actual_h = min(window_size, (stop1 - start1) // resolution)
        actual_w = min(window_size, (stop2 - start2) // resolution)
        pred = pred[:actual_h, :actual_w]
        w = hann[:actual_h, :actual_w]

        i0 = start1 // resolution
        j0 = start2 // resolution

        # Weighted accumulation
        accumulator[i0 : i0 + actual_h, j0 : j0 + actual_w] += pred * w
        weight_map[i0 : i0 + actual_h, j0 : j0 + actual_w] += w

        # Mirror for symmetric matrix (skip diagonal to avoid double-counting)
        if start1 != start2:
            accumulator[j0 : j0 + actual_w, i0 : i0 + actual_h] += pred.T * w.T
            weight_map[j0 : j0 + actual_w, i0 : i0 + actual_h] += w.T

    # Weighted average
    valid = weight_map > 0
    accumulator[valid] /= weight_map[valid]

    # Inverse transform using HR stats (or LR fallback)
    count_matrix = inverse_transform(
        accumulator, inv_params["mode"], inv_params["percentile"], inv_chrom_stats
    )

    # Enforce symmetry (rounding may break it slightly)
    count_matrix = np.maximum(count_matrix, count_matrix.T)

    # Extract upper triangle as sparse COO
    upper = sparse_triu(coo_matrix(count_matrix), k=0, format="coo")

    if upper.nnz == 0:
        return None

    # Convert to global bin IDs
    chrom_offset = lr_cooler.offset(chrom)
    pixels = pd.DataFrame(
        {
            "bin1_id": upper.row + chrom_offset,
            "bin2_id": upper.col + chrom_offset,
            "count": upper.data.astype(np.int32),
        }
    )
    pixels = pixels[pixels["count"] > 0].reset_index(drop=True)

    return pixels


# ---------------------------------------------------------------------------
# Top-level cooler creation
# ---------------------------------------------------------------------------


def create_enhanced_cooler(
    checkpoint_path: str | Path,
    config_path: str | Path,
    lr_cooler_path: str,
    output_path: str | Path,
    hr_cooler_path: str | None = None,
    chromosomes: list[str] | None = None,
    device: str | torch.device = "cuda",
    batch_size: int = 16,
) -> Path:
    """Create an enhanced cooler file from a trained model.

    Runs inference on LR patches, inverse-transforms predictions to integer
    counts, stitches patches with Hann-window blending, and writes a
    .cool file compatible with standard genomic tools.

    The forward transform uses LR cooler statistics (matching training).
    The inverse transform uses HR cooler statistics (recommended) to recover
    the correct count scale, since the model was trained to predict
    HR-normalized outputs. Falls back to LR stats if no HR cooler provided.

    Args:
        checkpoint_path: Path to model checkpoint (.pth file).
        config_path: Path to run config YAML.
        lr_cooler_path: Path to low-resolution cooler to enhance.
        output_path: Output path for the enhanced .cool file.
        hr_cooler_path: Path to high-resolution cooler (for inverse transform
            scaling). Recommended for correct output count scale. If None,
            falls back to LR stats (counts will be underscaled).
        chromosomes: Chromosomes to process (e.g. ['chr1', 'chr2']).
            If None, uses all chromosomes in the LR cooler.
        device: Compute device.
        batch_size: Inference batch size.

    Returns:
        Path to the created cooler file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device(device)

    # Load config
    config_path = Path(config_path)
    print(f"Loading config: {config_path}")
    config = RunConfig.load(config_path)

    # Load model
    print(f"Loading checkpoint: {checkpoint_path}")
    generator = load_generator(checkpoint_path, config, device)

    # Resolve transforms and inverse params
    transforms = resolve_transforms(config.data.transforms)
    inv_params = _find_transform_params(transforms)
    print(f"Transforms: {[repr(t) for t in transforms]}")
    print(f"Inverse mode: {inv_params['mode']}, percentile: {inv_params['percentile']}")

    # Open LR cooler
    lr_clr = cooler.Cooler(lr_cooler_path)
    bins = lr_clr.bins()[:]

    # Determine chromosomes — intersect LR and HR if both provided
    if chromosomes is None:
        chromosomes = list(lr_clr.chromnames)
    if hr_cooler_path is not None:
        hr_clr = cooler.Cooler(hr_cooler_path)
        hr_chroms = set(hr_clr.chromnames)
        lr_chroms = set(chromosomes)
        valid = lr_chroms & hr_chroms
        skipped = lr_chroms - hr_chroms
        if skipped:
            print(f"Skipping {len(skipped)} chroms not in HR cooler: {sorted(skipped)[:5]}{'...' if len(skipped) > 5 else ''}")
        chromosomes = [c for c in chromosomes if c in valid]
    print(f"Chromosomes ({len(chromosomes)}): {chromosomes[:10]}{'...' if len(chromosomes) > 10 else ''}")

    # Compute stats
    data = config.data
    window_cfg = WindowConfig(
        window_size=data.window_size,
        resolution=data.resolution,
        step=data.step,
    )

    print("Computing LR chromosome stats...")
    lr_stats = _compute_stats(
        lr_cooler_path, chromosomes, window_cfg, transforms, data.n_processes,
        nonzero_percentile=data.nonzero_percentile,
    )

    # Inverse stats: HR if provided, else fall back to LR
    if hr_cooler_path is not None:
        print(f"Computing HR chromosome stats (for inverse transform)...")
        print(f"  HR cooler: {hr_cooler_path}")
        inv_stats = _compute_stats(
            hr_cooler_path, chromosomes, window_cfg, transforms, data.n_processes,
            nonzero_percentile=data.nonzero_percentile,
        )
    else:
        print(
            "Warning: No HR cooler provided. Using LR stats for inverse transform. "
            "Output counts will be underscaled. Use --hr-cooler for correct scaling."
        )
        inv_stats = lr_stats

    # Process each chromosome
    print("Running inference...")
    pixel_chunks: list[pd.DataFrame] = []

    for chrom in chromosomes:
        if chrom not in lr_stats:
            print(f"  {chrom}: skipped (no LR stats)")
            continue
        if chrom not in inv_stats:
            print(f"  {chrom}: skipped (no inverse stats)")
            continue

        pixels = predict_chromosome(
            generator=generator,
            lr_cooler=lr_clr,
            chrom=chrom,
            transforms=transforms,
            inv_params=inv_params,
            lr_chrom_stats=lr_stats[chrom],
            inv_chrom_stats=inv_stats[chrom],
            window_size=data.window_size,
            resolution=data.resolution,
            step=data.step,
            max_distance=data.max_distance,
            device=device,
            batch_size=batch_size,
        )

        if pixels is not None and len(pixels) > 0:
            pixel_chunks.append(pixels)
            print(
                f"  {chrom}: {len(pixels):,} pixels, "
                f"count range [{pixels['count'].min()}, {pixels['count'].max()}]"
            )

    # Write cooler
    print("Writing cooler...")
    if pixel_chunks:
        pixels_all = pd.concat(pixel_chunks, ignore_index=True)
        pixels_all = pixels_all.sort_values(["bin1_id", "bin2_id"]).reset_index(drop=True)
    else:
        pixels_all = pd.DataFrame(columns=["bin1_id", "bin2_id", "count"])
        print("Warning: No pixels to write!")

    cooler.create_cooler(
        str(output_path),
        bins=bins,
        pixels=pixels_all,
        dtypes={"count": np.int32},
        ordered=True,
        symmetric_upper=True,
        metadata={
            "source_lr": lr_cooler_path,
            "source_hr": hr_cooler_path,
            "checkpoint": str(checkpoint_path),
            "config": str(config_path),
            "model_type": config.model.type,
            "inverse_mode": inv_params["mode"],
            "inverse_percentile": inv_params["percentile"],
            "inverse_stats_source": "hr" if hr_cooler_path else "lr",
            "chromosomes": chromosomes,
        },
    )

    # Verify
    out_clr = cooler.Cooler(str(output_path))
    total_count = out_clr.info.get("sum", 0)
    print(f"Output: {output_path}")
    print(f"Total contacts: {total_count:,}")
    print(f"Bins: {out_clr.info['nbins']:,}")

    return output_path
