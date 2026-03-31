"""Genomic coordinate generation for contact matrix extraction.

This module provides tools for generating sliding window coordinates
over chromosomes for extracting contact matrix samples from cooler files.

Key components:
- WindowConfig: Configuration dataclass for sliding window parameters
- CoordinateGenerator: Generates and caches genomic coordinates
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path

import cooler
import numpy as np
import pandas as pd
from tqdm import tqdm


# -----------------------------------------------------------------------------
# Multiprocessing Worker Functions (must be at module level for pickling)
# -----------------------------------------------------------------------------


def _compute_chrom_stats_worker(
    args: tuple[str, cooler.Cooler, list[float], bool],
) -> tuple[str, dict]:
    """Worker function for parallel chromosome stats computation.

    Args:
        args: Tuple of (chromosome name, cooler object, percentiles list,
              nonzero flag).

    Returns:
        Tuple of (chromosome name, stats dictionary).
    """
    chrom, clr, percentiles, nonzero = args

    matrix = clr.matrix(balance=False).fetch(chrom)
    valid_data = matrix[~np.isnan(matrix)]

    # For percentiles: use only non-zero values when nonzero=True.
    # This is critical for sparse data (e.g. Pore-C) where >95% of the
    # matrix is zero and whole-matrix percentiles collapse to near-zero.
    pctile_data = valid_data[valid_data > 0] if nonzero else valid_data

    stats = {
        "min": float(np.min(valid_data)),
        "max": float(np.max(valid_data)),
        "mean": float(np.mean(valid_data)),
        "median": float(np.median(valid_data)),
        "std": float(np.std(valid_data)),
        "nonzero_percentile": nonzero,
        "percentiles": {p: float(np.percentile(pctile_data, p)) for p in percentiles}
        if len(pctile_data) > 0
        else {p: 0.0 for p in percentiles},
    }

    return chrom, stats


def _sanitize_coordinates_worker(
    args: tuple[int, pd.Series, cooler.Cooler, float],
) -> tuple[int, bool]:
    """Worker function for parallel coordinate sanitization.

    Args:
        args: Tuple of (index, row Series, cooler object, zero_threshold).

    Returns:
        Tuple of (index, is_valid boolean).
    """
    idx, row, clr, zero_threshold = args

    matrix = clr.matrix(balance=False).fetch(
        f"{row['CHR']}:{row['START1']}-{row['STOP1']}",
        f"{row['CHR']}:{row['START2']}-{row['STOP2']}",
    )

    # Check for all-NaN matrix
    if np.all(np.isnan(matrix)):
        return idx, False

    # Check zero ratio
    zero_ratio = np.sum(matrix == 0) / matrix.size
    if zero_ratio >= zero_threshold:
        return idx, False

    return idx, True


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------


@dataclass
class WindowConfig:
    """Configuration for sliding window parameters.

    Attributes:
        window_size: Number of genomic bins per window dimension.
        resolution: Base pair resolution per bin (e.g., 10000 for 10kb).
        step: Controls overlap between windows (1.0 = no overlap, 0.5 = 50% overlap).
        threshold: Base pairs to trim from chromosome ends.

    Example:
        >>> config = WindowConfig(window_size=128, resolution=10_000, step=0.5)
        >>> # This creates 1.28 Mb windows with 50% overlap
    """

    window_size: int = 128
    resolution: int = 10_000
    step: float = 1.0
    threshold: int = 0


# -----------------------------------------------------------------------------
# Coordinate Generator
# -----------------------------------------------------------------------------


class CoordinateGenerator:
    """Generates genomic coordinates using a sliding window approach.

    Supports two modes of coordinate generation:
    - Symmetric: Windows along the diagonal (START1 == START2)
    - Offset: Windows within a specified distance from the diagonal

    Features intelligent caching based on cooler file hash and chromosome range.

    Args:
        config: Window configuration parameters.
        cache_dir: Directory for caching coordinates and statistics.
        n_processes: Number of parallel processes for computation.

    Example:
        >>> config = WindowConfig(window_size=128, resolution=10_000)
        >>> generator = CoordinateGenerator(config)
        >>> generator.set_cooler("data/sample.mcool::/resolutions/10000")
        >>> coords = generator.generate_symmetric_coordinates(range(1, 19))
        >>> print(f"Generated {len(coords)} coordinate windows")
    """

    def __init__(
        self,
        config: WindowConfig | None = None,
        cache_dir: str = ".cache",
        n_processes: int | None = 4,
        nonzero_percentile: bool = False,
    ) -> None:
        self.config = config or WindowConfig()
        self._coordinates: pd.DataFrame | None = None
        self._cooler: cooler.Cooler | None = None
        self._chrom_stats: dict | None = None
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(exist_ok=True)
        self._coord_type: str | None = None
        self.chromosomes: str | list[str] | range | None = None
        self.n_processes = n_processes
        self.nonzero_percentile = nonzero_percentile

    def _get_cooler_hash(self) -> str:
        """Generate a hash incorporating cooler file, chromosome range, and window config.

        Returns:
            MD5 hash string for cache key generation.
        """
        if self._cooler is None:
            return ""
        cooler_path = self._cooler.filename
        mtime = os.path.getmtime(cooler_path)
        chrom_str = "_".join(map(str, self._parse_chromosome_input(self.chromosomes)))
        nz_str = "_nz" if self.nonzero_percentile else ""
        config_str = (
            f"{self.config.window_size}_{self.config.resolution}"
            f"_{self.config.step}_{self.config.threshold}{nz_str}"
        )
        hash_str = f"{cooler_path}_{mtime}_{chrom_str}_{config_str}"
        return hashlib.md5(hash_str.encode()).hexdigest()

    def _save_coordinates(self, coord_type: str, stage: str = "base") -> None:
        """Save current coordinates to cache.

        Args:
            coord_type: Coordinate type ('symmetric' or 'offset').
            stage: Cache stage ('base', 'stats', or 'sanitized').
        """
        if self._coordinates is None:
            return

        cache_path = (
            self._cache_dir
            / f"coordinates_{coord_type}_{stage}_{self._get_cooler_hash()}.parquet"
        )
        try:
            self._coordinates.to_parquet(cache_path)
        except Exception as e:
            print(f"Failed to save coordinates cache: {e}")

    def _load_coordinates(self, coord_type: str, stage: str = "base") -> pd.DataFrame | None:
        """Try to load cached coordinates for specific coordinate type and stage.

        Args:
            coord_type: Coordinate type ('symmetric' or 'offset').
            stage: Cache stage ('base', 'stats', or 'sanitized').
        """
        cache_path = (
            self._cache_dir
            / f"coordinates_{coord_type}_{stage}_{self._get_cooler_hash()}.parquet"
        )
        if cache_path.exists():
            try:
                return pd.read_parquet(cache_path)
            except Exception as e:
                print(f"Failed to load coordinates cache: {e}")
                return None
        return None

    def _save_chrom_stats(self) -> None:
        """Save chromosome statistics to cache."""
        if self._chrom_stats is None:
            return

        cache_path = self._cache_dir / f"chrom_stats_{self._get_cooler_hash()}.json"
        try:
            with open(cache_path, "w") as f:
                json.dump(self._chrom_stats, f)
        except Exception as e:
            print(f"Failed to save chromosome stats cache: {e}")

    def _load_chrom_stats(self) -> dict | None:
        """Try to load cached chromosome statistics."""
        cache_path = self._cache_dir / f"chrom_stats_{self._get_cooler_hash()}.json"
        if cache_path.exists():
            try:
                with open(cache_path, "r") as f:
                    return json.load(f)
            except Exception as e:
                print(f"Failed to load chromosome stats cache: {e}")
                return None
        return None

    def clear_cache(self) -> None:
        """Clear all cached data for current cooler file."""
        if not self._get_cooler_hash():
            return

        for cache_file in self._cache_dir.glob(f"*_{self._get_cooler_hash()}.*"):
            try:
                cache_file.unlink()
            except Exception as e:
                print(f"Failed to remove cache file {cache_file}: {e}")

    def set_cooler(self, cooler_path: str) -> None:
        """Set or update the cooler file to use.

        Args:
            cooler_path: Path to cooler file with resolution
                (e.g., 'file.mcool::/resolutions/10000').
        """
        self._cooler = cooler.Cooler(cooler_path)

    def generate_symmetric_coordinates(
        self,
        chromosomes: str | list[str] | range,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """Generate coordinates for sliding windows along the diagonal.

        Windows will overlap based on the step parameter in WindowConfig.

        Args:
            chromosomes: Chromosome identifiers (e.g., range(1, 19) for chr1-chr18).
            use_cache: If True, try to load from cache before computing.

        Returns:
            DataFrame with columns: CHR, START1, STOP1, START2, STOP2

        Raises:
            ValueError: If no cooler file has been set.
        """
        self.chromosomes = chromosomes
        if self._cooler is None:
            raise ValueError("No cooler file set. Call set_cooler() first.")

        self._coord_type = "symmetric"
        if use_cache:
            cached_coords = self._load_coordinates(self._coord_type, stage="base")
            if cached_coords is not None:
                self._coordinates = cached_coords
                return self._coordinates

        chrom_list = self._parse_chromosome_input(chromosomes)
        coords_list = []

        for chrom in chrom_list:
            if chrom not in self._cooler.chromsizes:
                print(
                    f"Warning: Chromosome {chrom} not found in cooler file, skipping."
                )
                continue

            chrom_size = self._get_adjusted_chrom_size(chrom)
            window_coords = self._generate_symmetric_coordinates(chrom, chrom_size)
            coords_list.extend(window_coords)

        self._coordinates = pd.DataFrame(
            coords_list,
            columns=["CHR", "START1", "STOP1", "START2", "STOP2"],
        )
        self._save_coordinates("symmetric", stage="base")
        return self._coordinates

    def generate_offset_coordinates(
        self,
        chromosomes: str | list[str] | range,
        max_distance: int,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """Generate coordinates for windows within max_distance of the diagonal.

        Windows will overlap based on step parameter in both x and y directions.

        Args:
            chromosomes: Chromosome identifiers.
            max_distance: Maximum distance from diagonal in base pairs.
            use_cache: If True, try to load from cache before computing.

        Returns:
            DataFrame with columns: CHR, START1, STOP1, START2, STOP2

        Raises:
            ValueError: If no cooler file has been set.
        """
        self.chromosomes = chromosomes
        if self._cooler is None:
            raise ValueError("No cooler file set. Call set_cooler() first.")

        self._coord_type = "offset"
        if use_cache:
            cached_coords = self._load_coordinates("offset", stage="base")
            if cached_coords is not None:
                self._coordinates = cached_coords
                return self._coordinates

        chrom_list = self._parse_chromosome_input(chromosomes)
        coords_list = []

        for chrom in chrom_list:
            if chrom not in self._cooler.chromsizes:
                print(
                    f"Warning: Chromosome {chrom} not found in cooler file, skipping."
                )
                continue

            chrom_size = self._get_adjusted_chrom_size(chrom)
            window_coords = self._generate_offset_coordinates(
                chrom, chrom_size, max_distance
            )
            coords_list.extend(window_coords)

        self._coordinates = pd.DataFrame(
            coords_list,
            columns=["CHR", "START1", "STOP1", "START2", "STOP2"],
        )
        self._save_coordinates("offset", stage="base")
        return self._coordinates

    # Broad set of percentiles to always compute, so cache stays valid
    # across different transform configurations
    _BASE_PERCENTILES = [99.99, 99.975, 99.95, 99.9, 99.5, 99.0, 95.0]

    def compute_chrom_stats(
        self,
        percentiles: list[float] | None = None,
        use_cache: bool = True,
    ) -> dict:
        """Compute statistics for each chromosome using multiprocessing.

        Always computes a broad base set of percentiles so that the cache
        remains valid across different transform configurations. Any extra
        percentiles requested beyond the base set are included automatically.

        Args:
            percentiles: Additional percentiles to compute beyond the base set.
            use_cache: If True, try to load cached stats before computing.

        Returns:
            Dictionary mapping chromosome names to their statistics.

        Raises:
            ValueError: If no cooler file or chromosomes have been set.
        """
        requested = set(percentiles) if percentiles else set()
        all_percentiles = sorted(set(self._BASE_PERCENTILES) | requested, reverse=True)

        if self._cooler is None:
            raise ValueError("No cooler file set. Call set_cooler() first.")
        if self.chromosomes is None:
            raise ValueError("No chromosomes specified. Generate coordinates first.")

        if use_cache:
            cached_stats = self._load_chrom_stats()
            if cached_stats is not None:
                # Check if cached stats cover all requested percentiles
                sample_chrom = next(iter(cached_stats))
                cached_pctiles = set(cached_stats[sample_chrom].get("percentiles", {}))
                # Keys may be stored as strings after JSON round-trip
                cached_floats = {float(k) for k in cached_pctiles}
                if set(all_percentiles) <= cached_floats:
                    self._chrom_stats = cached_stats
                    return self._chrom_stats
                # Cache exists but is missing percentiles — recompute

        chrom_list = self._parse_chromosome_input(self.chromosomes)
        worker_args = [
            (chrom, self._cooler, all_percentiles, self.nonzero_percentile)
            for chrom in chrom_list
        ]

        stats = {}
        with Pool(processes=self.n_processes) as pool:
            with tqdm(
                total=len(chrom_list),
                desc="Computing chromosome stats",
                ncols=80,
                bar_format="{l_bar}{bar:30}{r_bar}{bar:-10b}",
            ) as pbar:
                for chrom, chrom_stats in pool.imap_unordered(
                    _compute_chrom_stats_worker, worker_args
                ):
                    stats[chrom] = chrom_stats
                    pbar.update(1)

        self._chrom_stats = stats
        self._save_chrom_stats()
        return stats

    def get_chrom_stats(self) -> dict | None:
        """Return the computed chromosome statistics.

        Returns:
            Dictionary mapping chromosome names to their statistics,
            or None if stats have not been computed yet.
        """
        return self._chrom_stats

    def add_stats_to_coordinates(
        self,
        percentiles: list[float] | None = None,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """Add chromosome-level statistics to coordinate DataFrame.

        Args:
            percentiles: List of percentiles to include.
            use_cache: If True, try to load cached stats before computing.

        Returns:
            DataFrame with added statistic columns.

        Raises:
            ValueError: If no coordinates have been generated.
        """
        if percentiles is None:
            percentiles = list(self._BASE_PERCENTILES)

        if self._coordinates is None or self._coord_type is None:
            raise ValueError("No coordinates generated yet.")

        if self._chrom_stats is None:
            self.compute_chrom_stats(percentiles, use_cache)

        df = self._coordinates.copy()

        for stat in ["min", "max", "mean", "median", "std"]:
            df[f"chrom_{stat}"] = df["CHR"].map(lambda x: self._chrom_stats[x][stat])

        # Handle percentiles - try both string and float keys
        # NOTE: p is bound as default arg to avoid late-binding closure bug
        for p in percentiles:

            def get_percentile(x: str, _p=p) -> float:
                try:
                    return self._chrom_stats[x]["percentiles"][str(_p)]
                except KeyError:
                    return self._chrom_stats[x]["percentiles"][_p]

            df[f"chrom_{p}"] = df["CHR"].map(get_percentile)

        self._coordinates = df
        self._save_coordinates(coord_type=self._coord_type, stage="stats")
        return df

    def sanitize_coordinates(
        self,
        zero_threshold: float = 1.0,
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """Remove coordinate pairs that result in zero or NaN matrices.

        Uses multiprocessing to check each coordinate pair.

        Args:
            zero_threshold: Maximum fraction of zeros allowed (1.0 means all zeros).
            use_cache: If True, try to load from cache before computing.

        Returns:
            Sanitized DataFrame with invalid coordinates removed.

        Raises:
            ValueError: If no coordinates have been generated.
        """
        if self._coordinates is None or self._coord_type is None:
            raise ValueError("No coordinates generated yet.")

        if use_cache:
            cached_coords = self._load_coordinates(self._coord_type, stage="sanitized")
            if cached_coords is not None:
                # Validate that cached parquet has all percentile columns
                # present in the current (freshly computed) coordinates
                required_cols = set(self._coordinates.columns)
                if required_cols <= set(cached_coords.columns):
                    self._coordinates = cached_coords
                    return self._coordinates
                # Stale cache — missing columns, recompute sanitization

        worker_args = [
            (idx, row, self._cooler, zero_threshold)
            for idx, row in self._coordinates.iterrows()
        ]

        valid_indices = []
        with Pool(processes=self.n_processes) as pool:
            with tqdm(
                total=len(self._coordinates),
                desc="Sanitizing coordinates",
                ncols=80,
                bar_format="{l_bar}{bar:30}{r_bar}{bar:-10b}",
            ) as pbar:
                for idx, is_valid in pool.imap_unordered(
                    _sanitize_coordinates_worker, worker_args
                ):
                    if is_valid:
                        valid_indices.append(idx)
                    pbar.update(1)

        self._coordinates = self._coordinates.loc[valid_indices].reset_index(drop=True)
        self._save_coordinates(self._coord_type, stage="sanitized")
        return self._coordinates

    def _generate_symmetric_coordinates(
        self, chrom: str, chrom_size: int
    ) -> list[list]:
        """Generate coordinates along diagonal with overlap controlled by step."""
        exact_window_size = self.config.window_size * self.config.resolution
        step_size = int(exact_window_size * self.config.step)
        coords = []

        for start in range(0, chrom_size - exact_window_size + 1, step_size):
            stop = start + exact_window_size
            coords.append([chrom, start, stop, start, stop])

        return coords

    def _generate_offset_coordinates(
        self, chrom: str, chrom_size: int, max_distance: int
    ) -> list[list]:
        """Generate coordinates within max_distance of diagonal.

        For each diagonal position, always includes the diagonal patch
        (start1 == start2), then steps outward in both directions until
        max_distance is reached or the chromosome boundary is hit.
        """
        exact_window_size = self.config.window_size * self.config.resolution
        step_size = int(exact_window_size * self.config.step)
        coords = []

        for start1 in range(0, chrom_size - exact_window_size + 1, step_size):
            stop1 = start1 + exact_window_size

            # Always include the diagonal patch
            coords.append([chrom, start1, stop1, start1, start1 + exact_window_size])

            # Step outward from diagonal in both directions
            for offset in range(step_size, max_distance + 1, step_size):
                # Below diagonal (start2 > start1)
                start2 = start1 + offset
                if start2 + exact_window_size <= chrom_size:
                    coords.append([chrom, start1, stop1, start2, start2 + exact_window_size])

                # Above diagonal (start2 < start1)
                start2 = start1 - offset
                if start2 >= 0:
                    coords.append([chrom, start1, stop1, start2, start2 + exact_window_size])

        return coords

    def _parse_chromosome_input(
        self, chromosomes: str | list[str] | range | None
    ) -> list[str]:
        """Convert input to list of chromosome names."""
        if chromosomes is None:
            return []
        if isinstance(chromosomes, str):
            return [chromosomes]
        elif isinstance(chromosomes, range):
            return [f"chr{i}" for i in chromosomes]
        elif isinstance(chromosomes, list):
            if chromosomes and isinstance(chromosomes[0], int):
                return [f"chr{i}" for i in chromosomes]
            return chromosomes
        raise ValueError(f"Unsupported chromosome input type: {type(chromosomes)}")

    def _get_adjusted_chrom_size(self, chrom: str) -> int:
        """Get chromosome size adjusted by threshold."""
        size = self._cooler.chromsizes[chrom]
        if self.config.threshold > 0:
            size = max(0, size - self.config.threshold)
        return size

    def save_coordinates(self, output_path: str) -> None:
        """Save generated coordinates to CSV file.

        Args:
            output_path: Path for output CSV file.

        Raises:
            ValueError: If no coordinates have been generated.
        """
        if self._coordinates is None:
            raise ValueError("No coordinates generated yet.")
        self._coordinates.to_csv(output_path, index=False)

    @property
    def coordinates(self) -> pd.DataFrame | None:
        """Access the currently generated coordinates."""
        return self._coordinates
