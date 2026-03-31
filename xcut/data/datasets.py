"""PyTorch Dataset classes for contact matrix data.

This module provides Dataset implementations for loading paired
 low-resolution and high-resolution contact matrices from cooler files.

Architecture:
    - BaseContactDataset: Abstract base class with shared logic
    - CoolerDataset: Two-cooler dataset (separate LR and HR mcool files)
    - SingleCoolerDataset: One-cooler dataset with Binomial downsampling
    - TensorDataset: Simple wrapper for pre-loaded tensor data
    - FixedSizeWrapper: Ensures fixed-size outputs

Transform Pipeline:
    Datasets accept a list of callable transforms that are applied
    sequentially to the data. Each transform has signature:

        (matrix: np.ndarray, context: dict) -> np.ndarray

    The context dict contains chromosome-level statistics:

        {
            'chrom_max': 500.0,
            'chrom_99.95': 380.2,
            'chrom_99.99': 450.1,
            'chrom': 'chr1',
            'sample_row': pd.Series,
        }

    Example:
        >>> from xcut.data.transforms import HandleNan, LogTransform, ClipLogByPercentile
        >>> transforms = [HandleNan(), LogTransform(), ClipLogByPercentile(99.95)]
        >>> dataset = CoolerDataset(..., transforms=transforms)
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any

import cooler
import numpy as np
import pandas as pd
import yaml
from torch.utils.data import Dataset

from .coordinates import CoordinateGenerator, WindowConfig
from .transforms import BaseTransform, BinomialDownsample

logger = logging.getLogger(__name__)

DEFAULT_PERCENTILES = [99.99, 99.95, 99.9, 99.5]


def _extract_percentiles_from_transforms(
    transforms: list[BaseTransform] | None,
) -> set[float]:
    """Extract all percentile values from transforms recursively.

    Inspects transforms for 'percentile' attributes, including nested Compose.

    Args:
        transforms: List of transforms to inspect.

    Returns:
        Set of percentile values found.
    """
    percentiles: set[float] = set()

    if transforms is None:
        return percentiles

    for transform in transforms:
        if hasattr(transform, "percentile"):
            p = getattr(transform, "percentile")
            if p is not None:
                percentiles.add(p)

        if hasattr(transform, "transforms"):
            nested = _extract_percentiles_from_transforms(
                getattr(transform, "transforms")
            )
            percentiles.update(nested)

    return percentiles


def _scalar_str(val: Any) -> str:
    """Extract string scalar from Series value."""
    if isinstance(val, pd.Series):
        return str(val.iloc[0])
    return str(val)


def _scalar_int(val: Any) -> int:
    """Extract int scalar from Series value."""
    if isinstance(val, pd.Series):
        return int(val.iloc[0])
    return int(val)


def _scalar_float(val: Any) -> float:
    """Extract float scalar from Series value."""
    if isinstance(val, pd.Series):
        return float(val.iloc[0])
    return float(val)


class BaseContactDataset(Dataset, ABC):
    """Abstract base class for contact matrix datasets.

    Provides shared logic for:
    - Coordinate generation and caching
    - Transform pipeline application
    - Context dict construction from sample_row
    - Meta dump utilities

    Subclasses must implement:
    - _fetch_lr(idx, sample_row): Fetch low-resolution matrix
    - _fetch_hr(idx, sample_row): Fetch high-resolution matrix
    - _set_cooler_for_coordinates(): Set cooler for coordinate generation
    - _get_meta_extras(): Return subclass-specific metadata

    Args:
        window_config: Configuration for sliding window parameters.
        transforms: List of transforms to apply to matrices.
            Each transform receives (matrix, context) and returns transformed matrix.
            Default: None (no transforms applied).
        percentiles: List of percentile values to compute and store in context.
            If None, percentiles are auto-detected from transforms (e.g., ClipByPercentile(99.95)).
            If provided, these are used instead of auto-detection.
            (Note: explicit list replaces auto-detected values.)
            Default: None (auto-detect from transforms).
        sym_coor: If True, generate diagonal-only coordinates.
        max_distance: Maximum distance from diagonal for offset coordinates.
        zero_threshold: Maximum fraction of zeros allowed in valid matrices.
        cache_coors: Whether to cache generated coordinates.
        chrom_range: Chromosomes to include.
        n_processes: Number of parallel processes for coordinate computation.
        nonzero_percentile: If True, compute percentiles over non-zero values only.
            Critical for sparse data (e.g. Pore-C) where >95% zeros cause
            whole-matrix percentiles to collapse near zero.
    """

    def __init__(
        self,
        window_config: WindowConfig,
        transforms: list[BaseTransform] | None = None,
        percentiles: list[float] | None = None,
        sym_coor: bool = False,
        max_distance: int = 2_000_000,
        zero_threshold: float = 1.0,
        cache_coors: bool = True,
        chrom_range: str | list[str] | range = range(1, 23),
        n_processes: int | None = 4,
        nonzero_percentile: bool = False,
    ) -> None:
        self.window_config = window_config
        self.transforms = transforms

        # Auto-detect percentiles from transforms
        detected_percentiles = _extract_percentiles_from_transforms(transforms)

        # Merge with explicitly provided percentiles
        if percentiles is not None:
            # User provided explicit list - merge with detected
            final_percentiles = set(percentiles) | detected_percentiles
            self.percentiles = sorted(final_percentiles, reverse=True)
            logger.debug(f"Using explicit + detected percentiles: {self.percentiles}")
        elif detected_percentiles:
            # Only detected percentiles
            self.percentiles = sorted(detected_percentiles, reverse=True)
            logger.debug(
                f"Auto-detected percentiles from transforms: {self.percentiles}"
            )
        else:
            # No percentiles detected or provided - use defaults
            self.percentiles = DEFAULT_PERCENTILES.copy()
            logger.debug(f"Using default percentiles: {self.percentiles}")

        self.is_symmetric = sym_coor
        self.cache_coors = cache_coors
        self.zero_threshold = zero_threshold
        self.chrom_range = chrom_range
        self.max_distance = max_distance
        self.nonzero_percentile = nonzero_percentile

        self._coordinates_df: pd.DataFrame | None = None
        self._coor_generator: CoordinateGenerator | None = None
        self._chrom_stats: dict | None = None
        self._init_coordinates(n_processes)

    def _init_coordinates(self, n_processes: int | None) -> None:
        """Initialize coordinate generator and compute coordinates."""
        self._coor_generator = CoordinateGenerator(
            self.window_config,
            n_processes=n_processes,
            nonzero_percentile=self.nonzero_percentile,
        )
        self._set_cooler_for_coordinates()

        if self.is_symmetric:
            self._coor_generator.generate_symmetric_coordinates(
                chromosomes=self.chrom_range, use_cache=self.cache_coors
            )
        else:
            self._coor_generator.generate_offset_coordinates(
                chromosomes=self.chrom_range,
                max_distance=self.max_distance,
                use_cache=self.cache_coors,
            )

        self._coor_generator.add_stats_to_coordinates(
            percentiles=self.percentiles,
            use_cache=self.cache_coors,
        )
        self._coordinates_df = self._coor_generator.sanitize_coordinates(
            zero_threshold=self.zero_threshold, use_cache=self.cache_coors
        )
        self._chrom_stats = self._coor_generator.get_chrom_stats()

    @abstractmethod
    def _set_cooler_for_coordinates(self) -> None:
        """Set the cooler file for coordinate generation."""
        pass

    @abstractmethod
    def _fetch_lr(self, idx: int, sample_row: pd.Series) -> np.ndarray:
        """Fetch low-resolution matrix for given sample.

        Args:
            idx: Sample index.
            sample_row: Row from coordinates DataFrame with genomic coords.

        Returns:
            Low-resolution contact matrix.
        """
        pass

    @abstractmethod
    def _fetch_hr(self, idx: int, sample_row: pd.Series) -> np.ndarray:
        """Fetch high-resolution matrix for given sample.

        Args:
            idx: Sample index.
            sample_row: Row from coordinates DataFrame with genomic coords.

        Returns:
            High-resolution contact matrix.
        """
        pass

    def _build_context(self, sample_row: pd.Series) -> dict:
        """Build context dict for transforms from sample_row.

        Args:
            sample_row: Row from coordinates DataFrame.

        Returns:
            Context dict with chrom_max, percentiles, chrom, sample_row.
        """
        context: dict = {
            "chrom": _scalar_str(sample_row["CHR"]),
            "sample_row": sample_row,
            "chrom_max": _scalar_float(sample_row["chrom_max"]),
        }
        for p in self.percentiles:
            col = f"chrom_{p}"
            if col in sample_row.index:
                context[col] = _scalar_float(sample_row[col])
        return context

    def _apply_transforms(self, matrix: np.ndarray, context: dict) -> np.ndarray:
        """Apply transform pipeline to matrix.

        Args:
            matrix: Input matrix.
            context: Context dict for transforms.

        Returns:
            Transformed matrix.
        """
        if self.transforms is None:
            return matrix.astype(np.float32)

        for transform in self.transforms:
            matrix = transform(matrix, context)
        return matrix.astype(np.float32)

    def _extract_coords(self, sample_row: pd.Series) -> dict:
        """Extract coordinate dict from sample_row.

        Args:
            sample_row: Row from coordinates DataFrame.

        Returns:
            Dict with CHR, START1, STOP1, START2, STOP2.
        """
        return {
            "CHR": _scalar_str(sample_row["CHR"]),
            "START1": _scalar_int(sample_row["START1"]),
            "STOP1": _scalar_int(sample_row["STOP1"]),
            "START2": _scalar_int(sample_row["START2"]),
            "STOP2": _scalar_int(sample_row["STOP2"]),
        }

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        if self._coordinates_df is None:
            return 0
        return len(self._coordinates_df)

    def _build_lr_context(self, sample_row: pd.Series) -> dict:
        """Build context dict for LR transforms.

        By default returns the same context as HR. Subclasses with separate
        LR cooler files override this to use LR-specific statistics.

        Args:
            sample_row: Row from coordinates DataFrame.

        Returns:
            Context dict for LR transforms.
        """
        return self._build_context(sample_row)

    def __getitem__(self, idx: int) -> dict:
        """Get a sample by index.

        Args:
            idx: Sample index.

        Returns:
            Dictionary with 'lr', 'hr', and 'coor' keys.
        """
        assert self._coordinates_df is not None
        sample_row = self._coordinates_df.iloc[idx]

        lr = self._fetch_lr(idx, sample_row)
        hr = self._fetch_hr(idx, sample_row)

        lr_context = self._build_lr_context(sample_row)
        hr_context = self._build_context(sample_row)
        lr = self._apply_transforms(lr, lr_context)
        hr = self._apply_transforms(hr, hr_context)

        return {
            "lr": np.expand_dims(lr, axis=0),
            "hr": np.expand_dims(hr, axis=0),
            "coor": self._extract_coords(sample_row),
        }

    # Meta dump methods

    def get_meta(self) -> dict:
        """Return complete dataset configuration as dict.

        Returns:
            Dict with dataset config, transforms, and statistics.
        """
        chrom_list = self._get_chrom_list()
        return {
            "dataset": {
                "type": self.__class__.__name__,
                "window_config": {
                    "window_size": self.window_config.window_size,
                    "resolution": self.window_config.resolution,
                    "step": self.window_config.step,
                },
                "percentiles": self.percentiles,
                "nonzero_percentile": self.nonzero_percentile,
                "chrom_range": chrom_list,
                "is_symmetric": self.is_symmetric,
                "zero_threshold": self.zero_threshold,
                **self._get_meta_extras(),
            },
            "transforms": [repr(t) for t in self.transforms] if self.transforms else [],
            "statistics": {
                "total_samples": len(self),
                "coordinate_type": "symmetric" if self.is_symmetric else "offset",
                "chromosomes": self._chrom_stats or {},
                "exported_at": datetime.now().isoformat(),
            },
        }

    @abstractmethod
    def _get_meta_extras(self) -> dict:
        """Return subclass-specific metadata.

        Override in subclasses to add extra fields to the meta.

        Returns:
            Dict with additional dataset-specific fields.
        """
        return {}

    def _get_chrom_list(self) -> list[str]:
        """Get list of chromosomes used."""
        if self._coordinates_df is None:
            return []
        return list(self._coordinates_df["CHR"].unique())

    def _get_cooler_hash(self) -> str | None:
        """Get hash of cooler file for caching.

        Override in subclasses.

        Returns:
            Hash string or None.
        """
        return None

    def to_yaml(self, path: str | Path | None = None) -> None:
        """Export dataset meta to YAML file.

        Args:
            path: Output path. If None, uses default cache location.
        """
        meta = self.get_meta()
        if path is None:
            cooler_hash = self._get_cooler_hash()
            if cooler_hash:
                path = Path(".cache") / f"dataset_meta_{cooler_hash}.yaml"
            else:
                path = (
                    Path(".cache")
                    / f"dataset_meta_{datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml"
                )

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            yaml.dump(meta, f, default_flow_style=False, sort_keys=False)

        logger.info(f"Exported dataset meta to {path}")

    def to_json(self, path: str | Path | None = None) -> None:
        """Export dataset meta to JSON file.

        Args:
            path: Output path. If None, uses default cache location.
        """
        meta = self.get_meta()
        if path is None:
            cooler_hash = self._get_cooler_hash()
            if cooler_hash:
                path = Path(".cache") / f"dataset_meta_{cooler_hash}.json"
            else:
                path = (
                    Path(".cache")
                    / f"dataset_meta_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                )

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            json.dump(meta, f, indent=2)

        logger.info(f"Exported dataset meta to {path}")

    def load_meta(self, path: str | Path | None = None) -> dict | None:
        """Load cached dataset meta if exists.

        Args:
            path: Input path. If None, uses default cache location.

        Returns:
            Dict with meta or None if not found.
        """
        if path is None:
            cooler_hash = self._get_cooler_hash()
            if cooler_hash is None:
                return None
            path = Path(".cache") / f"dataset_meta_{cooler_hash}.json"

        path = Path(path)
        if not path.exists():
            return None

        with open(path) as f:
            return json.load(f)


class CoolerDataset(BaseContactDataset):
    """PyTorch Dataset for paired LR/HR contact matrices from mcool files.

    Loads contact matrices on-the-fly from separate cooler files.

    Args:
        window_config: Configuration for sliding window parameters.
        lr_cooler_path: Path to low-resolution cooler with resolution
            (e.g., 'file.mcool::/resolutions/10000').
        hr_cooler_path: Path to high-resolution cooler with resolution.
        balance: Normalization method (False for raw, 'ICE', 'VC', etc.).
        transforms: List of transforms to apply.
        percentiles: List of percentile values to compute (default [99.99, 99.95, 99.9, 99.5]).
        sym_coor: If True, generate diagonal-only coordinates.
        max_distance: Maximum distance from diagonal for offset coordinates.
        zero_threshold: Maximum fraction of zeros allowed.
        cache_coors: Whether to cache generated coordinates.
        chrom_range: Chromosomes to include.
        n_processes: Number of parallel processes.

    Example:
        >>> from xcut.data.transforms import HandleNan, LogTransform, ClipLogByPercentile
        >>> config = WindowConfig(window_size=128, resolution=10_000, step=0.5)
        >>> dataset = CoolerDataset(
        ...     window_config=config,
        ...     lr_cooler_path='data/sample.100x.mcool::/resolutions/10000',
        ...     hr_cooler_path='data/sample.mcool::/resolutions/10000',
        ...     transforms=[HandleNan(), LogTransform(), ClipLogByPercentile(99.95)],
        ...     percentiles=[99.95, 99.99],
        ...     chrom_range=range(1, 19),
        ... )
    """

    def __init__(
        self,
        window_config: WindowConfig,
        lr_cooler_path: str,
        hr_cooler_path: str,
        balance: str | bool = False,
        transforms: list[BaseTransform] | None = None,
        percentiles: list[float] | None = None,
        sym_coor: bool = False,
        max_distance: int = 2_000_000,
        zero_threshold: float = 1.0,
        cache_coors: bool = True,
        chrom_range: str | list[str] | range = range(1, 23),
        n_processes: int | None = 4,
        nonzero_percentile: bool = False,
    ) -> None:
        self.lr_cooler_path = lr_cooler_path
        self.hr_cooler_path = hr_cooler_path
        self.lr_cooler = cooler.Cooler(lr_cooler_path)
        self.hr_cooler = cooler.Cooler(hr_cooler_path)
        self.balance = balance

        # Will be populated after super().__init__ computes HR coordinates
        self._lr_chrom_stats: dict | None = None

        super().__init__(
            window_config=window_config,
            transforms=transforms,
            percentiles=percentiles,
            sym_coor=sym_coor,
            max_distance=max_distance,
            zero_threshold=zero_threshold,
            cache_coors=cache_coors,
            chrom_range=chrom_range,
            n_processes=n_processes,
            nonzero_percentile=nonzero_percentile,
        )

        # Compute LR-specific chromosome stats for independent normalization
        self._compute_lr_stats(n_processes)

    def _compute_lr_stats(self, n_processes: int | None) -> None:
        """Compute chromosome statistics from the LR cooler."""
        lr_gen = CoordinateGenerator(
            self.window_config,
            n_processes=n_processes,
            nonzero_percentile=self.nonzero_percentile,
        )
        lr_gen.set_cooler(self.lr_cooler_path)
        lr_gen.chromosomes = self.chrom_range
        lr_stats = lr_gen.compute_chrom_stats(
            percentiles=self.percentiles,
            use_cache=self.cache_coors,
        )
        self._lr_chrom_stats = lr_stats

    def _set_cooler_for_coordinates(self) -> None:
        if self._coor_generator is not None:
            self._coor_generator.set_cooler(self.hr_cooler_path)

    def _get_cooler_hash(self) -> str | None:
        return self._coor_generator._get_cooler_hash() if self._coor_generator else None

    def _get_meta_extras(self) -> dict:
        return {
            "lr_cooler_path": self.lr_cooler_path,
            "hr_cooler_path": self.hr_cooler_path,
            "balance": self.balance,
            "cooler_hash": self._get_cooler_hash(),
        }

    def _build_lr_context(self, sample_row: pd.Series) -> dict:
        """Build context dict for LR transforms using LR cooler stats."""
        chrom = _scalar_str(sample_row["CHR"])
        context: dict = {
            "chrom": chrom,
            "sample_row": sample_row,
        }

        if self._lr_chrom_stats and chrom in self._lr_chrom_stats:
            lr_stats = self._lr_chrom_stats[chrom]
            context["chrom_max"] = float(lr_stats["max"])
            for p in self.percentiles:
                # Handle both string and float keys (JSON round-trip)
                pctiles = lr_stats.get("percentiles", {})
                val = pctiles.get(str(p)) if str(p) in pctiles else pctiles.get(p)
                if val is not None:
                    context[f"chrom_{p}"] = float(val)
        else:
            # Fallback to HR context if LR stats not available
            return self._build_context(sample_row)

        return context

    def _fetch_lr(self, idx: int, sample_row: pd.Series) -> np.ndarray:
        chrom = _scalar_str(sample_row["CHR"])
        start1 = _scalar_int(sample_row["START1"])
        stop1 = _scalar_int(sample_row["STOP1"])
        start2 = _scalar_int(sample_row["START2"])
        stop2 = _scalar_int(sample_row["STOP2"])

        return self.lr_cooler.matrix(balance=self.balance).fetch(
            f"{chrom}:{start1}-{stop1}",
            f"{chrom}:{start2}-{stop2}",
        )

    def _fetch_hr(self, idx: int, sample_row: pd.Series) -> np.ndarray:
        chrom = _scalar_str(sample_row["CHR"])
        start1 = _scalar_int(sample_row["START1"])
        stop1 = _scalar_int(sample_row["STOP1"])
        start2 = _scalar_int(sample_row["START2"])
        stop2 = _scalar_int(sample_row["STOP2"])

        return self.hr_cooler.matrix(balance=self.balance).fetch(
            f"{chrom}:{start1}-{stop1}",
            f"{chrom}:{start2}-{stop2}",
        )


class SingleCoolerDataset(BaseContactDataset):
    """PyTorch Dataset using a single HR cooler with Binomial downsampling.

    Generates low-resolution input on the fly via Binomial(count, 1/ratio)
    subsampling of raw counts.

    Supports two downsampling modes:
    - deterministic: Pre-computes downsampled matrix per chromosome at init.
    - stochastic: Re-draws the Binomial sample every __getitem__ call.

    Args:
        window_config: Configuration for sliding window parameters.
        cooler_path: Path to the high-resolution cooler with resolution.
        downsample_ratio: Factor to reduce read depth (e.g., 16.0 = 1/16 reads).
        transforms: List of transforms to apply.
        percentiles: List of percentile values to compute.
        stochastic: If True, re-sample LR every __getitem__.
        seed: Random seed for reproducibility.
        sym_coor: If True, generate diagonal-only coordinates.
        max_distance: Maximum distance from diagonal.
        zero_threshold: Maximum fraction of zeros allowed.
        cache_coors: Whether to cache generated coordinates.
        chrom_range: Chromosomes to include.
        n_processes: Number of parallel processes.

    Example:
        >>> config = WindowConfig(window_size=64, resolution=10_000, step=0.5)
        >>> dataset = SingleCoolerDataset(
        ...     window_config=config,
        ...     cooler_path='data/sample.mcool::/resolutions/10000',
        ...     downsample_ratio=16.0,
        ...     transforms=[HandleNan(), LogTransform()],
        ...     percentiles=[99.95, 99.99],
        ...     stochastic=True,
        ...     chrom_range=range(1, 19),
        ... )
    """

    def __init__(
        self,
        window_config: WindowConfig,
        cooler_path: str,
        downsample_ratio: float = 16.0,
        transforms: list[BaseTransform] | None = None,
        percentiles: list[float] | None = None,
        stochastic: bool = False,
        seed: int = 42,
        sym_coor: bool = False,
        max_distance: int = 2_000_000,
        zero_threshold: float = 1.0,
        cache_coors: bool = True,
        chrom_range: str | list[str] | range = range(1, 23),
        n_processes: int | None = 4,
        nonzero_percentile: bool = False,
    ) -> None:
        self.cooler_path = cooler_path
        self.cooler = cooler.Cooler(cooler_path)
        self.downsample_ratio = downsample_ratio
        self.stochastic = stochastic
        self.seed = seed

        self._rng = np.random.default_rng(seed)
        self._lr_matrices: dict[str, np.ndarray] | None = None

        super().__init__(
            window_config=window_config,
            transforms=transforms,
            percentiles=percentiles,
            sym_coor=sym_coor,
            max_distance=max_distance,
            zero_threshold=zero_threshold,
            cache_coors=cache_coors,
            chrom_range=chrom_range,
            n_processes=n_processes,
            nonzero_percentile=nonzero_percentile,
        )

        if not self.stochastic:
            self._precompute_lr()

    def _set_cooler_for_coordinates(self) -> None:
        if self._coor_generator is not None:
            self._coor_generator.set_cooler(self.cooler_path)

    def _get_cooler_hash(self) -> str | None:
        return self._coor_generator._get_cooler_hash() if self._coor_generator else None

    def _get_meta_extras(self) -> dict:
        return {
            "cooler_path": self.cooler_path,
            "downsample_ratio": self.downsample_ratio,
            "stochastic": self.stochastic,
            "seed": self.seed,
            "cooler_hash": self._get_cooler_hash(),
        }

    def _precompute_lr(self) -> None:
        """Pre-compute downsampled matrices for all chromosomes."""
        if self._coordinates_df is None:
            return

        chroms = self._coordinates_df["CHR"].unique()
        rng = np.random.default_rng(self.seed)
        self._lr_matrices = {}

        downsample = BinomialDownsample(ratio=self.downsample_ratio, rng=rng)

        for chrom in chroms:
            hr_full = self.cooler.matrix(balance=False).fetch(chrom)
            hr_full = np.nan_to_num(hr_full, nan=0.0)
            self._lr_matrices[chrom] = downsample(hr_full, {})

    def _fetch_region(
        self,
        matrix: np.ndarray,
        chrom: str,
        start1: int,
        stop1: int,
        start2: int,
        stop2: int,
    ) -> np.ndarray:
        """Extract patch from full chromosome matrix using genomic coords."""
        resolution = self.window_config.resolution
        chrom_offset = self.cooler.offset(chrom)
        bin1_start = start1 // resolution - chrom_offset
        bin1_stop = stop1 // resolution - chrom_offset
        bin2_start = start2 // resolution - chrom_offset
        bin2_stop = stop2 // resolution - chrom_offset
        return matrix[bin1_start:bin1_stop, bin2_start:bin2_stop].copy()

    def _fetch_lr(self, idx: int, sample_row: pd.Series) -> np.ndarray:
        chrom = _scalar_str(sample_row["CHR"])
        start1 = _scalar_int(sample_row["START1"])
        stop1 = _scalar_int(sample_row["STOP1"])
        start2 = _scalar_int(sample_row["START2"])
        stop2 = _scalar_int(sample_row["STOP2"])

        if self.stochastic:
            hr = self._fetch_hr(idx, sample_row)
            hr = np.nan_to_num(hr, nan=0.0)
            downsample = BinomialDownsample(ratio=self.downsample_ratio, rng=self._rng)
            return downsample(hr, {})
        else:
            assert self._lr_matrices is not None
            return self._fetch_region(
                self._lr_matrices[chrom], chrom, start1, stop1, start2, stop2
            )

    def _fetch_hr(self, idx: int, sample_row: pd.Series) -> np.ndarray:
        chrom = _scalar_str(sample_row["CHR"])
        start1 = _scalar_int(sample_row["START1"])
        stop1 = _scalar_int(sample_row["STOP1"])
        start2 = _scalar_int(sample_row["START2"])
        stop2 = _scalar_int(sample_row["STOP2"])

        return self.cooler.matrix(balance=False).fetch(
            f"{chrom}:{start1}-{stop1}",
            f"{chrom}:{start2}-{stop2}",
        )

    def _build_context(self, sample_row: pd.Series) -> dict:
        context = super()._build_context(sample_row)
        if self.stochastic:
            context["rng"] = self._rng
        return context


class TensorDataset(Dataset):
    """Simple dataset for pre-loaded tensor data.

    Useful for testing or when data is already in memory.

    Args:
        lr_data: Low-resolution data tensor of shape (N, 1, H, W).
        hr_data: High-resolution data tensor of shape (N, 1, H, W).

    Returns:
        Tuple of (lr_tensor, hr_tensor) each of shape (1, H, W).

    Example:
        >>> import torch
        >>> lr = torch.randn(100, 1, 128, 128)
        >>> hr = torch.randn(100, 1, 128, 128)
        >>> dataset = TensorDataset(lr, hr)
        >>> lr_sample, hr_sample = dataset[0]
    """

    def __init__(self, lr_data: np.ndarray, hr_data: np.ndarray) -> None:
        if lr_data.shape[0] != hr_data.shape[0]:
            raise ValueError(
                f"LR and HR data must have same number of samples. "
                f"Got LR: {lr_data.shape[0]}, HR: {hr_data.shape[0]}"
            )
        self.lr_data = lr_data
        self.hr_data = hr_data

    def __len__(self) -> int:
        return self.lr_data.shape[0]

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        return self.lr_data[idx], self.hr_data[idx]


class FixedSizeWrapper(Dataset):
    """Wraps CoolerDataset to guarantee fixed-size outputs.

    Crops or pads samples to (1, size, size) — some coordinate windows
    near chromosome boundaries can produce slightly different shapes.

    Args:
        dataset: CoolerDataset to wrap.
        size: Target size for output matrices.

    Returns:
        Dict with 'lr', 'hr' (each shape 1, size, size), and 'coor'.
    """

    def __init__(self, dataset: BaseContactDataset, size: int) -> None:
        self.dataset = dataset
        self.size = size

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict:
        sample = self.dataset[idx]
        lr = self._fix(sample["lr"])
        hr = self._fix(sample["hr"])
        return {"lr": lr, "hr": hr, "coor": sample["coor"]}

    def _fix(self, x: np.ndarray) -> np.ndarray:
        """Crop or pad to (1, size, size)."""
        _, h, w = x.shape
        s = self.size
        if h >= s and w >= s:
            return x[:, :s, :s]
        out = np.zeros((1, s, s), dtype=x.dtype)
        out[:, : min(h, s), : min(w, s)] = x[:, : min(h, s), : min(w, s)]
        return out
