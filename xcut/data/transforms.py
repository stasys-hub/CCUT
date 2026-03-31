"""Callable transforms for contact matrix preprocessing.

This module provides composable transform functions for preprocessing
contact matrices. All transforms inherit from BaseTransform and follow
the signature:

    (matrix: np.ndarray, context: dict) -> np.ndarray

The context dict provides chromosome-level statistics that some transforms
need for proper normalization:

    {
        'chrom_max': 500.0,       # Chromosome-level max count
        'chrom_99.95': 380.2,     # 99.95th percentile value
        'chrom_99.99': 450.1,     # 99.99th percentile value
        'chrom': 'chr1',          # Chromosome name
        'sample_row': pd.Series,  # Full coordinate row with stats
        'rng': Generator,         # Random generator (for stochastic)
    }

Transforms can be composed using the Compose class:

    >>> pipeline = Compose([
    ...     HandleNan(fill_value=0.0),
    ...     LogTransform(pseudocount=1.0),
    ...     ScaleByChromMax(),
    ... ])
    >>> transformed = pipeline(matrix, context)
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import numpy as np

logger = logging.getLogger(__name__)


class BaseTransform(ABC):
    """Abstract base class for all transforms.

    All transforms must implement __call__ with signature:
        (matrix: np.ndarray, context: dict) -> np.ndarray
    """

    @abstractmethod
    def __call__(self, matrix: np.ndarray, context: dict) -> np.ndarray:
        """Apply transform to matrix.

        Args:
            matrix: Input contact matrix.
            context: Context dict with chromosome-level stats.

        Returns:
            Transformed matrix.
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class Compose(BaseTransform):
    """Chain multiple transforms together.

    Args:
        transforms: List of transforms to apply sequentially.

    Example:
        >>> pipeline = Compose([
        ...     HandleNan(),
        ...     LogTransform(),
        ... ])
        >>> result = pipeline(matrix, {'chrom_max': 100.0})
    """

    def __init__(self, transforms: list[BaseTransform]) -> None:
        self.transforms = transforms

    def __call__(self, matrix: np.ndarray, context: dict) -> np.ndarray:
        for transform in self.transforms:
            matrix = transform(matrix, context)
        return matrix

    def __iter__(self):
        return iter(self.transforms)

    def __repr__(self) -> str:
        names = [t.__class__.__name__ for t in self.transforms]
        return f"Compose({names})"


class HandleNan(BaseTransform):
    """Replace NaN values in matrix.

    Args:
        fill_value: Value to replace NaN with (default 0.0).

    Example:
        >>> transform = HandleNan(fill_value=0.0)
        >>> clean = transform(matrix, {})
    """

    def __init__(self, fill_value: float = 0.0) -> None:
        self.fill_value = fill_value

    def __call__(self, matrix: np.ndarray, context: dict) -> np.ndarray:
        return np.nan_to_num(matrix, nan=self.fill_value)

    def __repr__(self) -> str:
        return f"HandleNan(fill_value={self.fill_value})"


class LogTransform(BaseTransform):
    """Apply log(1 + x) transformation.

    Args:
        pseudocount: Value added before log (default 1.0).
        scale_by_max: If True, divide by log(1 + chrom_max) after transform.
            Requires 'chrom_max' in context.

    Example:
        >>> transform = LogTransform(pseudocount=1.0, scale_by_max=True)
        >>> result = transform(matrix, {'chrom_max': 500.0})
    """

    def __init__(self, pseudocount: float = 1.0, scale_by_max: bool = False) -> None:
        self.pseudocount = pseudocount
        self.scale_by_max = scale_by_max

    def __call__(self, matrix: np.ndarray, context: dict) -> np.ndarray:
        matrix = np.log(matrix + self.pseudocount)
        if self.scale_by_max:
            chrom_max = context.get("chrom_max", 0.0)
            if chrom_max > 0:
                ceiling = np.log(1.0 + chrom_max)
                if ceiling > 0:
                    matrix = matrix / ceiling
        return matrix

    def __repr__(self) -> str:
        return f"LogTransform(pseudocount={self.pseudocount}, scale_by_max={self.scale_by_max})"


class Clip(BaseTransform):
    """Clip matrix values to a fixed range.

    Args:
        min_val: Minimum value (default 0.0).
        max_val: Maximum value (default None = no upper bound).

    Example:
        >>> transform = Clip(min_val=0.0, max_val=100.0)
        >>> clipped = transform(matrix, {})
    """

    def __init__(self, min_val: float = 0.0, max_val: float | None = None) -> None:
        self.min_val = min_val
        self.max_val = max_val

    def __call__(self, matrix: np.ndarray, context: dict) -> np.ndarray:
        return np.clip(matrix, a_min=self.min_val, a_max=self.max_val)

    def __repr__(self) -> str:
        return f"Clip(min_val={self.min_val}, max_val={self.max_val})"


class ClipByPercentile(BaseTransform):
    """Clip using chromosome-level percentile value from context.

    Looks up pre-computed percentile value stored in the context.
    Requires 'chrom_{percentile}' in context (e.g., 'chrom_99.95').

    Args:
        percentile: Percentile value to use for clipping (e.g., 99.95).
        min_val: Minimum value for clipping (default 0.0).

    Example:
        >>> transform = ClipByPercentile(percentile=99.95)
        >>> clipped = transform(matrix, {'chrom_99.95': 150.5})
    """

    def __init__(self, percentile: float = 99.95, min_val: float = 0.0) -> None:
        self.percentile = percentile
        self.min_val = min_val

    def __call__(self, matrix: np.ndarray, context: dict) -> np.ndarray:
        key = f"chrom_{self.percentile}"
        clip_val = context.get(key)
        if clip_val is None:
            logger.warning(
                f"ClipByPercentile: '{key}' not found in context for chrom="
                f"{context.get('chrom', '?')}. Returning matrix unclipped."
            )
            return matrix
        return np.clip(matrix, a_min=self.min_val, a_max=clip_val)

    def __repr__(self) -> str:
        return f"ClipByPercentile(percentile={self.percentile}, min_val={self.min_val})"


class ClipByChromValue(BaseTransform):
    """Clip using manually specified per-chromosome values.

    Useful when you have domain knowledge about appropriate clip thresholds
    per chromosome (e.g., from visual inspection or external analysis).

    Args:
        chrom_values: Dict mapping chromosome names to clip values.
            e.g. {'chr1': 500.0, 'chr2': 420.0, ...}
        min_val: Minimum value for clipping (default 0.0).
        default: Fallback clip value for chromosomes not in chrom_values.
            If None, no clipping is applied for missing chromosomes.

    Example:
        >>> transform = ClipByChromValue({'chr1': 500, 'chr2': 420}, default=400)
        >>> clipped = transform(matrix, {'chrom': 'chr1'})
    """

    def __init__(
        self,
        chrom_values: dict[str, float],
        min_val: float = 0.0,
        default: float | None = None,
    ) -> None:
        self.chrom_values = chrom_values
        self.min_val = min_val
        self.default = default

    def __call__(self, matrix: np.ndarray, context: dict) -> np.ndarray:
        chrom = context.get("chrom")
        if chrom is None:
            return matrix

        clip_val = self.chrom_values.get(chrom, self.default)
        if clip_val is None:
            return matrix
        return np.clip(matrix, a_min=self.min_val, a_max=clip_val)

    def __repr__(self) -> str:
        n = len(self.chrom_values)
        return f"ClipByChromValue(n_chroms={n}, default={self.default})"


class ClipLogByPercentile(BaseTransform):
    """Clip log-transformed values using chromosome-level percentile.

    Applies log transform first, then clips to log(1 + clip_val).
    This is useful when clip_val represents a pre-computed percentile
    of raw counts, but you want to work in log space.

    Requires 'chrom_{percentile}' in context.

    Args:
        percentile: Percentile value to use (e.g., 99.99).
        pseudocount: Value added before log (default 1.0).

    Example:
        >>> transform = ClipLogByPercentile(percentile=99.99)
        >>> result = transform(matrix, {'chrom_99.99': 150.5})
    """

    def __init__(self, percentile: float = 99.99, pseudocount: float = 1.0) -> None:
        self.percentile = percentile
        self.pseudocount = pseudocount

    def __call__(self, matrix: np.ndarray, context: dict) -> np.ndarray:
        matrix = np.log(matrix + self.pseudocount)
        key = f"chrom_{self.percentile}"
        clip_val = context.get(key)
        if clip_val is not None:
            ceiling = np.log(1.0 + clip_val)
            matrix = np.clip(matrix, a_min=0.0, a_max=ceiling)
            if ceiling > 0:
                matrix = matrix / ceiling
        else:
            chrom_max = context.get("chrom_max", 0.0)
            ceiling = np.log(1.0 + chrom_max)
            if ceiling > 0:
                matrix = matrix / ceiling
        return matrix

    def __repr__(self) -> str:
        return f"ClipLogByPercentile(percentile={self.percentile}, pseudocount={self.pseudocount})"


class DivideByMax(BaseTransform):
    """Scale matrix to [0, ~1] by dividing by a chromosome-level maximum.

    Divisor priority:
    1. If percentile is set → use chrom-level percentile value (e.g. chrom_99.9)
    2. Otherwise → use chrom_max

    Both are per-chromosome, ensuring consistent normalization across patches
    from the same chromosome and trivial inversion back to counts.

    Note: This is division-only (no min subtraction). Contact matrices have
    min=0 after HandleNan, so the result is equivalent to min-max for that case.

    Args:
        percentile: If set, divide by this percentile value from context.
            Requires 'chrom_{percentile}' in context.
            If None, divides by chrom_max.

    Example:
        >>> transform = DivideByMax(percentile=99.95)
        >>> normalized = transform(matrix, {'chrom_99.95': 100.0})
        >>> # Invert: counts = normalized * 100.0
    """

    def __init__(self, percentile: float | None = None) -> None:
        self.percentile = percentile

    def __call__(self, matrix: np.ndarray, context: dict) -> np.ndarray:
        matrix = matrix.astype(np.float32)

        if self.percentile is not None:
            key = f"chrom_{self.percentile}"
            divisor = context.get(key)
            if divisor is not None and divisor > 0:
                return (matrix / divisor).astype(np.float32)
            logger.warning(
                f"DivideByMax: '{key}' not found in context for chrom="
                f"{context.get('chrom', '?')}. Falling back to chrom_max."
            )

        # Fallback: always use chrom_max (per-chromosome, not per-sample)
        chrom_max = context.get("chrom_max", 0.0)
        if chrom_max > 0:
            return (matrix / chrom_max).astype(np.float32)
        return matrix

    def __repr__(self) -> str:
        return f"DivideByMax(percentile={self.percentile})"


# Backwards compatibility alias
MinMaxNormalize = DivideByMax


class ScaleByChromMax(BaseTransform):
    """Scale matrix by chromosome-level max from context.

    Divides all values by chrom_max. Useful for log-space normalization.

    Requires 'chrom_max' in context.

    Example:
        >>> transform = ScaleByChromMax()
        >>> scaled = transform(matrix, {'chrom_max': 500.0})
    """

    def __call__(self, matrix: np.ndarray, context: dict) -> np.ndarray:
        chrom_max = context.get("chrom_max", 0.0)
        if chrom_max > 0:
            return (matrix / chrom_max).astype(np.float32)
        return matrix


class BinomialDownsample(BaseTransform):
    """Downsample count matrix using Binomial sampling.

    Each bin's count is treated as independent Bernoulli trials, producing
    correct Poisson-like noise for simulating low-coverage sequencing.

    Note: This transform is stateful if rng is provided at init.
    For stochastic datasets, pass rng in context instead.

    Args:
        ratio: Downsampling ratio (e.g., 16.0 keeps ~1/16 of reads).
        rng: NumPy random Generator. If None, looks for 'rng' in context.

    Example:
        >>> rng = np.random.default_rng(42)
        >>> transform = BinomialDownsample(ratio=16.0, rng=rng)
        >>> downsampled = transform(matrix, {})
    """

    def __init__(
        self,
        ratio: float,
        rng: np.random.Generator | None = None,
    ) -> None:
        if ratio < 1.0:
            raise ValueError(f"Downsampling ratio must be >= 1.0, got {ratio}")
        self.ratio = ratio
        self.rng = rng

    def __call__(self, matrix: np.ndarray, context: dict) -> np.ndarray:
        rng = self.rng or context.get("rng")
        if rng is None:
            rng = np.random.default_rng()

        counts = np.rint(matrix).astype(np.int64)
        counts = np.maximum(counts, 0)
        p = 1.0 / self.ratio
        return rng.binomial(counts, p).astype(np.int32)

    def __repr__(self) -> str:
        return f"BinomialDownsample(ratio={self.ratio})"


class Identity(BaseTransform):
    """Pass-through transform (no-op).

    Useful as a placeholder or for testing.
    """

    def __call__(self, matrix: np.ndarray, context: dict) -> np.ndarray:
        return matrix


class EnsureFloat32(BaseTransform):
    """Convert matrix to float32 dtype.

    Useful as final transform in pipeline to ensure consistent output type.
    """

    def __call__(self, matrix: np.ndarray, context: dict) -> np.ndarray:
        return matrix.astype(np.float32)


# Pre-built transform pipelines for common use cases


def get_log1p_pipeline(percentile: float | None = 99.99) -> Compose:
    """Get standard log1p normalization pipeline.

    Args:
        percentile: Percentile for clipping (e.g., 99.99).
            Set to None to disable clipping and scale by chrom_max instead.

    Returns:
        Compose object with [HandleNan, ClipLogByPercentile, EnsureFloat32]
        or [HandleNan, LogTransform(scale_by_max=True), EnsureFloat32]
    """
    transforms: list[BaseTransform] = [HandleNan(fill_value=0.0)]
    if percentile is not None:
        transforms.append(ClipLogByPercentile(percentile=percentile, pseudocount=1.0))
    else:
        transforms.append(LogTransform(pseudocount=1.0, scale_by_max=True))
    transforms.append(EnsureFloat32())
    return Compose(transforms)


def get_minmax_pipeline(percentile: float | None = 99.95) -> Compose:
    """Get min-max normalization pipeline.

    Args:
        percentile: Percentile for scaling.
            Set to None to scale by chrom_max instead.

    Returns:
        Compose object with [HandleNan, ClipByPercentile, DivideByMax, EnsureFloat32]
        or [HandleNan, DivideByMax, EnsureFloat32]
    """
    transforms: list[BaseTransform] = [HandleNan(fill_value=0.0)]
    if percentile is not None:
        transforms.append(ClipByPercentile(percentile=percentile, min_val=0.0))
        transforms.append(DivideByMax(percentile=percentile))
    else:
        transforms.append(DivideByMax(percentile=None))
    transforms.append(EnsureFloat32())
    return Compose(transforms)


