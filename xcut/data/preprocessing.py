"""Preprocessing utilities for contact matrices.

This module provides functions for normalizing and preprocessing
contact matrices before model training/inference, as well as
creating downsampled cooler files for training data generation.
"""

from __future__ import annotations

from pathlib import Path

import cooler
import numpy as np
import pandas as pd
from scipy.sparse import triu as sparse_triu


def min_max_normalize(
    matrix: np.ndarray,
    min_val: float | None = None,
    max_val: float | None = None,
) -> tuple[np.ndarray, float, float]:
    """Normalize matrix to [0, 1] range using min-max scaling.

    If min_val and max_val are provided, uses those for normalization
    (enables reconstruction of original values). Otherwise, uses
    the matrix's own min/max.

    Args:
        matrix: Input matrix of any shape.
        min_val: Optional minimum value for normalization. If None, uses matrix.min().
        max_val: Optional maximum value for normalization. If None, uses matrix.max().

    Returns:
        Tuple of (normalized matrix, min_val used, max_val used).
        Normalized matrix is float32 in range [0, 1].

    Example:
        >>> m = np.array([[0, 10], [5, 20]])
        >>> normalized, min_v, max_v = min_max_normalize(m)
        >>> print(normalized.min(), normalized.max(), min_v, max_v)
        0.0 1.0 0 20
    """
    actual_min: float = min_val if min_val is not None else float(matrix.min())
    actual_max: float = max_val if max_val is not None else float(matrix.max())

    if actual_max > actual_min:
        matrix_normalized = (matrix - actual_min) / (actual_max - actual_min)
    else:
        matrix_normalized = np.zeros_like(matrix, dtype=np.float32)

    return matrix_normalized.astype(np.float32), actual_min, actual_max


def min_max_normalize_simple(matrix: np.ndarray) -> np.ndarray:
    """Normalize matrix to [0, 1] range using min-max scaling (simple version).

    This is the original implementation that only returns the normalized matrix.
    Use min_max_normalize() if you need to track min/max for reconstruction.

    Args:
        matrix: Input matrix of any shape.

    Returns:
        Normalized matrix as float32 in range [0, 1].
        Returns zeros if matrix max equals zero.

    Example:
        >>> m = np.array([[0, 10], [5, 20]])
        >>> normalized = min_max_normalize_simple(m)
        >>> print(normalized.min(), normalized.max())
        0.0 1.0
    """
    normalized, _, _ = min_max_normalize(matrix)
    return normalized


def clip_percentile(
    matrix: np.ndarray,
    percentile: float,
    min_val: float = 0.0,
) -> np.ndarray:
    """Clip matrix values at given percentile.

    Args:
        matrix: Input matrix of any shape.
        percentile: Upper percentile for clipping (e.g., 99.95).
        min_val: Minimum value for clipping (default 0.0).

    Returns:
        Clipped matrix.

    Example:
        >>> m = np.array([[0, 100], [50, 1000]])
        >>> clipped = clip_percentile(m, 99.0)
    """
    max_val = np.percentile(matrix[~np.isnan(matrix)], percentile)
    return np.clip(matrix, a_min=min_val, a_max=max_val)


def handle_nan(
    matrix: np.ndarray,
    fill_value: float = 0.0,
) -> np.ndarray:
    """Replace NaN values in matrix.

    Args:
        matrix: Input matrix potentially containing NaN values.
        fill_value: Value to replace NaN with (default 0.0).

    Returns:
        Matrix with NaN values replaced.

    Example:
        >>> m = np.array([[1.0, np.nan], [np.nan, 4.0]])
        >>> clean = handle_nan(m)
        >>> print(clean)
        [[1. 0.]
         [0. 4.]]
    """
    return np.nan_to_num(matrix, nan=fill_value)


def enforce_symmetry(matrix: np.ndarray) -> np.ndarray:
    """Enforce matrix symmetry by averaging with transpose.

    Contact matrices should be symmetric (M[i,j] = M[j,i]).
    This function enforces that property.

    Args:
        matrix: Input square matrix.

    Returns:
        Symmetric matrix where output = (input + input.T) / 2.

    Raises:
        ValueError: If matrix is not square.

    Example:
        >>> m = np.array([[1, 2], [3, 4]])
        >>> sym = enforce_symmetry(m)
        >>> print(sym)
        [[1.  2.5]
         [2.5 4. ]]
    """
    if matrix.shape[-2] != matrix.shape[-1]:
        raise ValueError(f"Matrix must be square, got shape {matrix.shape[-2:]}.")
    return (matrix + np.swapaxes(matrix, -2, -1)) / 2


def log_transform(
    matrix: np.ndarray,
    pseudocount: float = 1.0,
) -> np.ndarray:
    """Apply log transformation to matrix.

    Useful for compressing dynamic range of contact matrices.

    Args:
        matrix: Input matrix with non-negative values.
        pseudocount: Value added before log to handle zeros (default 1.0).

    Returns:
        Log-transformed matrix.

    Example:
        >>> m = np.array([[0, 10], [100, 1000]])
        >>> log_m = log_transform(m)
    """
    return np.log(matrix + pseudocount)


def standardize(
    matrix: np.ndarray,
    mean: float | None = None,
    std: float | None = None,
) -> tuple[np.ndarray, float, float]:
    """Standardize matrix to zero mean and unit variance.

    Args:
        matrix: Input matrix.
        mean: Pre-computed mean (if None, computed from matrix).
        std: Pre-computed std (if None, computed from matrix).

    Returns:
        Tuple of (standardized_matrix, mean, std).

    Example:
        >>> m = np.array([[1, 2], [3, 4]], dtype=float)
        >>> standardized, mean, std = standardize(m)
        >>> print(f"Mean: {standardized.mean():.2f}, Std: {standardized.std():.2f}")
        Mean: 0.00, Std: 1.00
    """
    actual_mean: float = mean if mean is not None else float(matrix.mean())
    actual_std: float = std if std is not None else float(matrix.std())
    if actual_std == 0:
        actual_std = 1.0
    return (matrix - actual_mean) / actual_std, actual_mean, actual_std


def binomial_downsample(
    matrix: np.ndarray,
    ratio: float,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Downsample a count matrix by drawing from Binomial(count, 1/ratio) per bin.

    Each bin's count is treated as independent Bernoulli trials. This produces
    the correct Poisson-like noise structure for simulating low-coverage
    sequencing from a high-coverage contact matrix.

    Only operates on integer counts. If the input contains floats, they are
    rounded to the nearest integer first (with a warning-worthy assumption
    that the input is raw counts, not balanced data).

    Args:
        matrix: Input count matrix (non-negative integers expected).
        ratio: Downsampling ratio (e.g., 16.0 means keep ~1/16 of reads).
            Must be >= 1.0.
        rng: NumPy random Generator for reproducibility. If None, creates
            a new default generator.

    Returns:
        Downsampled count matrix with same shape, dtype int32.

    Example:
        >>> rng = np.random.default_rng(42)
        >>> hr = np.array([[100, 50], [50, 200]])
        >>> lr = binomial_downsample(hr, ratio=10.0, rng=rng)
        >>> lr.dtype
        dtype('int32')
    """
    if ratio < 1.0:
        raise ValueError(f"Downsampling ratio must be >= 1.0, got {ratio}")

    if rng is None:
        rng = np.random.default_rng()

    counts = np.rint(matrix).astype(np.int64)
    counts = np.maximum(counts, 0)

    p = 1.0 / ratio
    downsampled = rng.binomial(counts, p).astype(np.int32)
    return downsampled


def denormalize(
    matrix: np.ndarray,
    min_val: float,
    max_val: float,
) -> np.ndarray:
    """Denormalize a matrix that was normalized with min-max scaling.

    Reconstructs original scale values from normalized [0, 1] range.

    Args:
        matrix: Normalized matrix in range [0, 1].
        min_val: Minimum value used during normalization.
        max_val: Maximum value used during normalization.

    Returns:
        Denormalized matrix in original scale.

    Example:
        >>> original = np.array([[0, 10], [5, 20]])
        >>> normalized, min_v, max_v = min_max_normalize(original)
        >>> reconstructed = denormalize(normalized, min_v, max_v)
        >>> np.allclose(original, reconstructed)
        True
    """
    return matrix * (max_val - min_val) + min_val


def create_downsampled_cooler(
    input_cooler_path: str,
    output_path: str | Path,
    ratio: float,
    seed: int = 42,
    chromosomes: list[str] | None = None,
    include_trans: bool = True,
) -> Path:
    """Create a new cooler file by Binomial downsampling of raw counts.

    Reads an existing cooler, applies Binomial(count, 1/ratio) independently
    to each bin in the upper triangle, and writes the result as a new cooler
    with identical bins and metadata.

    This is a deterministic preprocessing step (given the seed) that produces
    a proper cooler artifact for training or evaluation. The output cooler
    can be used with CoolerDataset or any standard cooler tool.

    Args:
        input_cooler_path: URI to source cooler
            (e.g., 'file.mcool::/resolutions/10000').
        output_path: Path for the output .cool file.
        ratio: Downsampling ratio (e.g., 16.0 means keep ~1/16 of reads).
        seed: Random seed for reproducibility.
        chromosomes: List of chromosome names to include. If None, uses all
            chromosomes from the input cooler.
        include_trans: If True, also downsample inter-chromosomal (trans)
            contacts. Default True produces complete, balanceable coolers.

    Returns:
        Path to the created cooler file.

    Example:
        >>> create_downsampled_cooler(
        ...     'data/sample.mcool::/resolutions/10000',
        ...     'data/sample.16x.cool',
        ...     ratio=16.0,
        ... )
    """
    clr = cooler.Cooler(input_cooler_path)
    bins = clr.bins()[:].copy()
    if isinstance(bins, pd.Series):
        bins = bins.to_frame()
    rng = np.random.default_rng(seed)

    if chromosomes is None:
        chromosomes = list(clr.chromnames)

    output_path = Path(output_path)

    pixel_chunks: list[pd.DataFrame] = []

    for chrom in chromosomes:
        mat = clr.matrix(balance=False, sparse=True).fetch(chrom)
        upper = sparse_triu(mat, k=0, format="coo")

        if upper.nnz == 0:
            continue

        counts = upper.data.astype(np.int64)
        counts = np.maximum(counts, 0)
        downsampled_counts = rng.binomial(counts, 1.0 / ratio).astype(np.int32)

        nonzero_mask = downsampled_counts > 0
        if not np.any(nonzero_mask):
            continue

        chrom_offset = clr.offset(chrom)
        chunk = pd.DataFrame(
            {
                "bin1_id": upper.row[nonzero_mask] + chrom_offset,
                "bin2_id": upper.col[nonzero_mask] + chrom_offset,
                "count": downsampled_counts[nonzero_mask],
            }
        )
        pixel_chunks.append(chunk)

    if include_trans:
        n_chroms = len(chromosomes)
        for i in range(n_chroms):
            chrom1 = chromosomes[i]
            offset1 = clr.offset(chrom1)
            for j in range(i + 1, n_chroms):
                chrom2 = chromosomes[j]
                offset2 = clr.offset(chrom2)

                mat = clr.matrix(balance=False, sparse=True).fetch(chrom1, chrom2)
                mat_coo = mat.tocoo()

                if mat_coo.nnz == 0:
                    continue

                counts = mat_coo.data.astype(np.int64)
                counts = np.maximum(counts, 0)
                downsampled_counts = rng.binomial(counts, 1.0 / ratio).astype(np.int32)

                nonzero_mask = downsampled_counts > 0
                if not np.any(nonzero_mask):
                    continue

                chunk = pd.DataFrame(
                    {
                        "bin1_id": mat_coo.row[nonzero_mask] + offset1,
                        "bin2_id": mat_coo.col[nonzero_mask] + offset2,
                        "count": downsampled_counts[nonzero_mask],
                    }
                )
                pixel_chunks.append(chunk)

    if pixel_chunks:
        pixels = pd.concat(pixel_chunks, ignore_index=True)
        pixels = pixels.sort_values(["bin1_id", "bin2_id"]).reset_index(drop=True)
    else:
        pixels = pd.DataFrame(columns=["bin1_id", "bin2_id", "count"])

    cooler.create_cooler(
        str(output_path),
        bins=bins,
        pixels=pixels,
        dtypes={"count": np.int32},
        ordered=True,
        symmetric_upper=True,
        metadata={
            "source": input_cooler_path,
            "downsample_ratio": ratio,
            "seed": seed,
        },
    )

    return output_path


def create_transformed_cooler(
    input_cooler_path: str,
    output_path: str | Path,
    transforms: list,
    chromosomes: list[str] | None = None,
    percentiles: list[float] | None = None,
    nonzero_percentile: bool = False,
) -> Path:
    """Create a new cooler by applying transforms to each chromosome.

    Reads an existing cooler, applies a transform pipeline per chromosome
    (using per-chromosome statistics for context), and writes the result.

    The output cooler contains integer counts (transformed values are rounded).
    For transforms that produce float values in [0, 1] (e.g. normalize),
    the output will be mostly 0s and 1s — use this mainly for clipping.

    Args:
        input_cooler_path: URI to source cooler.
        output_path: Path for the output .cool file.
        transforms: List of BaseTransform objects to apply sequentially.
        chromosomes: Chromosomes to include. If None, uses all.
        percentiles: Percentiles to compute for context. If None,
            auto-detected from transforms.
        nonzero_percentile: If True, compute percentiles over non-zero
            values only. Critical for sparse data (e.g. Pore-C).

    Returns:
        Path to the created cooler file.
    """
    from scipy.sparse import coo_matrix

    from .transforms import BaseTransform

    clr = cooler.Cooler(input_cooler_path)
    bins = clr.bins()[:].copy()
    if isinstance(bins, pd.Series):
        bins = bins.to_frame()

    if chromosomes is None:
        chromosomes = list(clr.chromnames)

    output_path = Path(output_path)

    # Auto-detect percentiles from transforms
    needed_pcts: set[float] = set()
    for t in transforms:
        if hasattr(t, "percentile") and getattr(t, "percentile") is not None:
            needed_pcts.add(t.percentile)
    if percentiles:
        needed_pcts.update(percentiles)
    pct_list = sorted(needed_pcts, reverse=True)

    pixel_chunks: list[pd.DataFrame] = []

    for chrom in chromosomes:
        mat = clr.matrix(balance=False).fetch(chrom)
        mat = mat.astype(np.float64)
        valid = mat[~np.isnan(mat)]

        if valid.size == 0:
            continue

        # Build context — use nonzero values for percentiles if requested
        pctile_data = valid[valid > 0] if nonzero_percentile else valid
        context: dict = {
            "chrom": chrom,
            "chrom_max": float(np.max(valid)),
        }
        for p in pct_list:
            if len(pctile_data) > 0:
                context[f"chrom_{p}"] = float(np.percentile(pctile_data, p))
            else:
                context[f"chrom_{p}"] = 0.0

        # Apply transforms
        for t in transforms:
            mat = t(mat, context)

        # Round to int and extract upper triangle
        count_matrix = np.rint(mat).clip(0).astype(np.int32)
        count_matrix = np.maximum(count_matrix, count_matrix.T)

        upper = sparse_triu(coo_matrix(count_matrix), k=0, format="coo")

        if upper.nnz == 0:
            continue

        chrom_offset = clr.offset(chrom)
        nonzero = upper.data > 0
        if not np.any(nonzero):
            continue

        chunk = pd.DataFrame(
            {
                "bin1_id": upper.row[nonzero] + chrom_offset,
                "bin2_id": upper.col[nonzero] + chrom_offset,
                "count": upper.data[nonzero],
            }
        )
        pixel_chunks.append(chunk)

    if pixel_chunks:
        pixels = pd.concat(pixel_chunks, ignore_index=True)
        pixels = pixels.sort_values(["bin1_id", "bin2_id"]).reset_index(drop=True)
    else:
        pixels = pd.DataFrame(columns=["bin1_id", "bin2_id", "count"])

    cooler.create_cooler(
        str(output_path),
        bins=bins,
        pixels=pixels,
        dtypes={"count": np.int32},
        ordered=True,
        symmetric_upper=True,
        metadata={
            "source": input_cooler_path,
            "transforms": [repr(t) for t in transforms],
        },
    )

    return output_path


def compute_ps_curve(
    clr: cooler.Cooler,
    chromosomes: list[str] | None = None,
    max_distance_bp: int = 10_000_000,
) -> dict[int, float]:
    """Compute P(s) distance decay curve from a cooler.

    For each genomic distance, computes the mean contact frequency
    across all diagonals at that distance.

    Args:
        clr: Cooler object.
        chromosomes: Chromosomes to include. If None, uses all autosomes.
        max_distance_bp: Maximum distance to compute P(s) for.

    Returns:
        Dictionary mapping distance (in bp) to mean contact frequency.
    """
    resolution = clr.binsize
    if resolution is None:
        raise ValueError("Cooler has no resolution")
    max_distance_bins = max_distance_bp // resolution

    if chromosomes is None:
        chromosomes = [c for c in clr.chromnames if not c.startswith("chrM")]

    distance_sums: dict[int, float] = {}
    distance_counts: dict[int, int] = {}

    for chrom in chromosomes:
        try:
            mat = clr.matrix(balance=False).fetch(chrom).astype(np.float64)
        except Exception:
            continue

        n_bins = mat.shape[0]

        for d in range(1, min(max_distance_bins, n_bins)):
            diag = np.diag(mat, k=d)
            nonzero = diag[diag > 0]
            if len(nonzero) > 0:
                distance_bp = d * resolution
                distance_sums[distance_bp] = (
                    distance_sums.get(distance_bp, 0.0) + nonzero.sum()
                )
                distance_counts[distance_bp] = distance_counts.get(
                    distance_bp, 0
                ) + len(nonzero)

    ps_curve = {}
    for d_bp in distance_sums:
        if distance_counts[d_bp] > 0:
            ps_curve[d_bp] = distance_sums[d_bp] / distance_counts[d_bp]

    return ps_curve


def fill_ps_background(
    enhanced_cooler_path: str | Path,
    reference_cooler_path: str | Path,
    output_path: str | Path,
    noise_scale: float = 0.1,
    max_distance_bp: int = 5_000_000,
    chromosomes: list[str] | None = None,
    seed: int | None = None,
) -> Path:
    """Fill zero bins in enhanced cooler with P(s) background from reference.

    For bins where the enhanced cooler has zero but the reference has signal,
    sample low counts from the expected P(s) distribution. This creates
    a realistic "background" that helps ICE balancing converge.

    Args:
        enhanced_cooler_path: Path to enhanced cooler (model output).
        reference_cooler_path: Path to reference cooler for P(s) computation.
        output_path: Output path for filled cooler.
        noise_scale: Fraction of expected frequency to sample (0.05-0.20).
        max_distance_bp: Maximum distance to fill background for.
        chromosomes: Chromosomes to process. If None, uses all in enhanced.
        seed: Random seed for reproducibility.

    Returns:
        Path to the filled cooler.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)

    enh_clr = cooler.Cooler(str(enhanced_cooler_path))
    ref_clr = cooler.Cooler(str(reference_cooler_path))

    resolution = enh_clr.binsize
    assert resolution is not None, "Cooler has no resolution"
    max_distance_bins = max_distance_bp // resolution

    if chromosomes is None:
        chromosomes = list(enh_clr.chromnames)

    ps_curve = compute_ps_curve(ref_clr, chromosomes, max_distance_bp)

    bins = enh_clr.bins()[:].copy()
    if isinstance(bins, pd.Series):
        bins = bins.to_frame()
    pixel_chunks: list[pd.DataFrame] = []

    for chrom in chromosomes:
        if chrom not in enh_clr.chromnames:
            continue

        try:
            enh_mat = enh_clr.matrix(balance=False).fetch(chrom).astype(np.float64)
            ref_mat = ref_clr.matrix(balance=False).fetch(chrom).astype(np.float64)
        except Exception:
            continue

        n_bins = enh_mat.shape[0]
        chrom_offset = enh_clr.offset(chrom)

        rows = []
        cols = []
        counts = []

        enh_coo = sparse_triu(enh_mat, k=0, format="coo")
        rows.extend(enh_coo.row.tolist())
        cols.extend(enh_coo.col.tolist())
        counts.extend(enh_coo.data.astype(np.int32).tolist())

        for d in range(1, min(max_distance_bins, n_bins)):
            enh_diag = np.diag(enh_mat, k=d)
            ref_diag = np.diag(ref_mat, k=d)

            distance_bp = d * resolution
            expected_freq = ps_curve.get(distance_bp, 0)

            if expected_freq <= 0:
                continue

            zero_mask = (enh_diag == 0) & (ref_diag > 0)
            zero_indices = np.where(zero_mask)[0]

            if len(zero_indices) == 0:
                continue

            sampled_counts = rng.poisson(expected_freq * noise_scale, len(zero_indices))
            sampled_counts = np.clip(sampled_counts, 0, None).astype(np.int32)

            nonzero_samples = sampled_counts > 0
            if not np.any(nonzero_samples):
                continue

            row_indices = zero_indices[nonzero_samples]
            col_indices = row_indices + d
            sample_values = sampled_counts[nonzero_samples]

            rows.extend(row_indices.tolist())
            cols.extend(col_indices.tolist())
            counts.extend(sample_values.tolist())

        if rows:
            chunk = pd.DataFrame(
                {
                    "bin1_id": np.array(rows) + chrom_offset,
                    "bin2_id": np.array(cols) + chrom_offset,
                    "count": counts,
                }
            )
            chunk = chunk[chunk["count"] > 0].copy()
            if isinstance(chunk, pd.Series):
                chunk = chunk.to_frame().T
            if len(chunk) > 0:
                pixel_chunks.append(chunk)

    n_chroms = len(chromosomes)
    for i in range(n_chroms):
        for j in range(i + 1, n_chroms):
            chrom1, chrom2 = chromosomes[i], chromosomes[j]
            try:
                mat = enh_clr.matrix(balance=False, sparse=True).fetch(chrom1, chrom2)
                mat_coo = mat.tocoo()
                if mat_coo.nnz == 0:
                    continue

                offset1 = enh_clr.offset(chrom1)
                offset2 = enh_clr.offset(chrom2)

                nonzero = mat_coo.data > 0
                chunk = pd.DataFrame(
                    {
                        "bin1_id": mat_coo.row[nonzero] + offset1,
                        "bin2_id": mat_coo.col[nonzero] + offset2,
                        "count": mat_coo.data[nonzero].astype(np.int32),
                    }
                )
                pixel_chunks.append(chunk)
            except Exception:
                pass

    if pixel_chunks:
        pixels = pd.concat(pixel_chunks, ignore_index=True)
        pixels = pixels.sort_values(["bin1_id", "bin2_id"]).reset_index(drop=True)
    else:
        pixels = pd.DataFrame(columns=["bin1_id", "bin2_id", "count"])

    cooler.create_cooler(
        str(output_path),
        bins=bins,
        pixels=pixels,
        dtypes={"count": np.int32},
        ordered=True,
        symmetric_upper=True,
        metadata={
            "source_enhanced": enhanced_cooler_path,
            "source_reference": reference_cooler_path,
            "noise_scale": noise_scale,
            "max_distance_bp": max_distance_bp,
            "seed": seed,
        },
    )

    return output_path
