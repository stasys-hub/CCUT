"""Pydantic configuration models for xcut training runs.

All training parameters live here. Configs are serializable to/from YAML
so each run's settings are reproducible.

Usage:
    config = RunConfig(
        data=DataConfig(hr_cooler="sample.mcool::/resolutions/10000"),
        model=ModelConfig(type="hinet"),
        train=TrainConfig(epochs=100, lr=1e-4),
    )
    config.save("runs/my_run/config.yaml")

    # Later:
    config = RunConfig.load("runs/my_run/config.yaml")
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


class DataConfig(BaseModel):
    """Data loading configuration."""

    hr_cooler: str = Field(
        default="REQUIRED",
        description="Path to HR cooler (e.g. 'file.mcool::/resolutions/10000')",
    )
    lr_cooler: str | None = Field(
        default=None,
        description="Path to LR cooler. None = use SingleCoolerDataset with downsampling.",
    )
    downsample_ratio: float = Field(
        default=16.0,
        description="Binomial downsampling ratio (only used when lr_cooler is None)",
    )
    stochastic: bool = Field(
        default=False,
        description="Re-sample LR every __getitem__ (only for SingleCoolerDataset)",
    )

    window_size: int = Field(
        default=128, description="Number of bins per window dimension"
    )
    resolution: int = Field(default=10_000, description="Base pair resolution per bin")
    step: float = Field(
        default=0.5, description="Window overlap (1.0 = no overlap, 0.5 = 50%)"
    )
    threshold: int = Field(
        default=0, description="Base pairs to trim from chromosome ends"
    )

    chromosomes: list[int | str] = Field(
        default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
        description="Chromosome names or numbers to include",
    )
    val_chromosomes: list[int | str] = Field(
        default=[19], description="Chromosome names or numbers for validation"
    )
    test_chromosomes: list[int | str] = Field(
        default=[20, 21, 22], description="Chromosome names or numbers for testing"
    )
    sym_coor: bool = Field(default=False, description="Diagonal-only coordinates")
    max_distance: int = Field(
        default=2_000_000,
        description="Max distance from diagonal (bp) for offset coordinates",
    )
    zero_threshold: float = Field(
        default=1.0, description="Max fraction of zeros allowed in valid patches"
    )

    transforms: list[dict] | str = Field(
        default="log1p_99.99",
        description=(
            "Transform pipeline. Either a shorthand string ('log1p_99.99', 'minmax_99.95', "
            "'log1p_none', 'minmax_none') or an explicit list of transform dicts, each with "
            "'type' and optional params. Example:\n"
            "  transforms:\n"
            "    - type: HandleNan\n"
            "      fill_value: 0.0\n"
            "    - type: ClipByPercentile\n"
            "      percentile: 99.95\n"
            "    - type: MinMaxNormalize\n"
            "      percentile: 99.95\n"
            "    - type: EnsureFloat32"
        ),
    )
    percentiles: list[float] | None = Field(
        default=None,
        description="Extra percentiles to compute (auto-detected from transforms if None)",
    )
    nonzero_percentile: bool = Field(
        default=False,
        description="Compute percentiles over non-zero values only. Critical for sparse data (e.g. Pore-C) where whole-matrix percentiles collapse near zero.",
    )
    n_processes: int = Field(
        default=4, description="Parallel processes for coordinate computation"
    )


class ModelConfig(BaseModel):
    """Model architecture configuration."""

    type: str = Field(default="hinet", description="Model architecture key")
    params: dict = Field(
        default_factory=dict,
        description="Architecture-specific parameters passed to factory/init",
    )


class GANConfig(BaseModel):
    """GAN-specific training configuration."""

    enabled: bool = Field(default=False, description="Enable adversarial training")
    lr_d: float = Field(default=1e-4, description="Discriminator learning rate")
    gp_weight: float = Field(
        default=5.0, description="Gradient penalty weight (WGAN-GP)"
    )
    pixel_weight: float = Field(
        default=40.0, description="Pixel loss weight in generator"
    )
    structure_weight: float = Field(
        default=10.0, description="Structure consistency loss weight"
    )
    label_smoothing: float = Field(
        default=0.9, description="Label smoothing for discriminator"
    )
    noise_std: float = Field(
        default=0.1, description="Noise std on discriminator predictions"
    )


class FlowConfig(BaseModel):
    """Rectified flow training configuration."""

    structure_weight: float = Field(
        default=0.0, description="Structure consistency loss weight on predicted x0"
    )
    insulation_weight: float = Field(
        default=0.0, description="Insulation score loss weight"
    )
    decay_weight: float = Field(default=0.0, description="Distance decay loss weight")
    weight_decay: float = Field(default=1e-4, description="AdamW weight decay")


class TrainConfig(BaseModel):
    """Training loop configuration."""

    epochs: int = Field(default=100, description="Number of training epochs")
    batch_size: int = Field(default=16, description="Batch size")
    lr: float = Field(default=1e-4, description="Generator/model learning rate")
    beta1: float = Field(default=0.5, description="Adam beta1")
    beta2: float = Field(default=0.999, description="Adam beta2")
    grad_clip: float = Field(
        default=1.0, description="Max gradient norm (0 = disabled)"
    )

    patience: int = Field(default=7, description="Early stopping patience (epochs)")
    metric: Literal["ssim", "psnr"] = Field(
        default="ssim", description="Metric for early stopping / best model"
    )

    num_workers: int = Field(default=4, description="DataLoader workers")
    device: str = Field(default="cuda", description="Device (cuda, cpu, etc.)")
    seed: int = Field(default=42, description="Random seed")

    log_images_every: int = Field(
        default=5, description="Log sample images every N epochs (0 = disabled)"
    )

    gan: GANConfig = Field(
        default_factory=GANConfig, description="GAN training settings"
    )
    flow: FlowConfig = Field(
        default_factory=FlowConfig, description="Rectified flow training settings"
    )


class RunConfig(BaseModel):
    """Complete run configuration. Combines data, model, and training."""

    name: str = Field(default="run", description="Run name (used for output directory)")
    output_dir: str = Field(default="runs", description="Base output directory")

    data: DataConfig = Field(default_factory=DataConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    train: TrainConfig = Field(default_factory=TrainConfig)

    @property
    def run_dir(self) -> Path:
        """Full path to this run's output directory."""
        return Path(self.output_dir) / self.name

    def save(self, path: str | Path | None = None) -> Path:
        """Save config to YAML."""
        if path is None:
            path = self.run_dir / "config.yaml"
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False, sort_keys=False)
        return path

    @classmethod
    def load(cls, path: str | Path) -> RunConfig:
        """Load config from YAML."""
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(**data)

    @classmethod
    def load_with_overrides(
        cls, path: str | Path, overrides: list[str] | None = None
    ) -> RunConfig:
        """Load config from YAML, then apply dot-path overrides."""
        with open(path) as f:
            data = yaml.safe_load(f)

        for override in overrides or []:
            key, _, value = override.partition("=")
            if not key or not _:
                raise ValueError(
                    f"Invalid override format: '{override}'. Expected 'key=value'."
                )
            _set_nested(data, key.strip(), _parse_value(value.strip()))

        return cls(**data)


def _set_nested(d: dict, dotted_key: str, value: object) -> None:
    """Set a value in a nested dict using dot-separated key path."""
    keys = dotted_key.split(".")
    for key in keys[:-1]:
        if key not in d:
            d[key] = {}
        d = d[key]
    d[keys[-1]] = value


def _parse_value(s: str) -> object:
    """Parse a CLI override value string to a Python object."""
    low = s.lower()
    if low == "true":
        return True
    if low == "false":
        return False
    if low in ("null", "none"):
        return None
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    if s.startswith("[") or s.startswith("{"):
        try:
            return yaml.safe_load(s)
        except yaml.YAMLError:
            pass
    return s


def resolve_transforms(transforms_cfg: list[dict] | str) -> list:
    """Resolve transforms config to transform objects."""
    if isinstance(transforms_cfg, str):
        return _resolve_shorthand(transforms_cfg)
    return _resolve_transform_list(transforms_cfg)


def _resolve_shorthand(pipeline_key: str) -> list:
    """Resolve a shorthand pipeline string to transform objects."""
    from xcut.data.transforms import get_log1p_pipeline, get_minmax_pipeline

    match = re.match(r"^(log1p|minmax)_(.+)$", pipeline_key)
    if not match:
        raise ValueError(
            f"Unknown transform shorthand '{pipeline_key}'. "
            "Expected format: 'log1p_{{percentile}}' or 'minmax_{{percentile}}' "
            "(e.g., 'log1p_99.99', 'minmax_none')."
        )

    kind, pval = match.groups()
    percentile = None if pval.lower() == "none" else float(pval)

    if kind == "log1p":
        return list(get_log1p_pipeline(percentile).transforms)
    else:
        return list(get_minmax_pipeline(percentile).transforms)


_TRANSFORM_REGISTRY: dict[str, type] | None = None


def _get_transform_registry() -> dict[str, type]:
    """Build name -> class mapping from xcut.data.transforms (lazy, cached)."""
    global _TRANSFORM_REGISTRY
    if _TRANSFORM_REGISTRY is not None:
        return _TRANSFORM_REGISTRY

    from xcut.data import transforms as T

    _TRANSFORM_REGISTRY = {
        name: getattr(T, name)
        for name in [
            "HandleNan",
            "LogTransform",
            "Clip",
            "ClipByPercentile",
            "ClipByChromValue",
            "ClipLogByPercentile",
            "DivideByMax",
            "MinMaxNormalize",
            "ScaleByChromMax",
            "BinomialDownsample",
            "Identity",
            "EnsureFloat32",
        ]
    }
    return _TRANSFORM_REGISTRY


def _resolve_transform_list(specs: list[dict]) -> list:
    """Instantiate transforms from a list of {type: ..., **params} dicts."""
    registry = _get_transform_registry()
    result = []
    for i, spec in enumerate(specs):
        spec = dict(spec)
        type_name = spec.pop("type", None)
        if type_name is None:
            raise ValueError(f"Transform at index {i} missing 'type' key. Got: {spec}")
        if type_name not in registry:
            available = ", ".join(sorted(registry.keys()))
            raise ValueError(
                f"Unknown transform type '{type_name}' at index {i}. "
                f"Available: {available}"
            )
        result.append(registry[type_name](**spec))
    return result
