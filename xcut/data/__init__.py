"""xcut.data - Data loading and preprocessing for contact matrices."""

from xcut.data.coordinates import CoordinateGenerator, WindowConfig
from xcut.data.datasets import (
    BaseContactDataset,
    CoolerDataset,
    FixedSizeWrapper,
    SingleCoolerDataset,
    TensorDataset,
)
from xcut.data.preprocessing import (
    binomial_downsample,
    clip_percentile,
    create_downsampled_cooler,
    create_transformed_cooler,
    denormalize,
    enforce_symmetry,
    fill_ps_background,
    handle_nan,
    log_transform,
    min_max_normalize,
    min_max_normalize_simple,
    standardize,
)
from xcut.data.transforms import (
    BaseTransform,
    BinomialDownsample,
    Clip,
    ClipByChromValue,
    ClipByPercentile,
    ClipLogByPercentile,
    Compose,
    DivideByMax,
    EnsureFloat32,
    HandleNan,
    Identity,
    LogTransform,
    MinMaxNormalize,
    ScaleByChromMax,
    get_log1p_pipeline,
    get_minmax_pipeline,
)

# Backwards compatibility aliases
ClipByContext = ClipByPercentile
ClipLogByContext = ClipLogByPercentile

__all__ = [
    "CoordinateGenerator",
    "WindowConfig",
    "BaseContactDataset",
    "CoolerDataset",
    "SingleCoolerDataset",
    "TensorDataset",
    "FixedSizeWrapper",
    "BaseTransform",
    "Compose",
    "HandleNan",
    "LogTransform",
    "Clip",
    "ClipByChromValue",
    "ClipByPercentile",
    "ClipLogByPercentile",
    "DivideByMax",
    "MinMaxNormalize",
    "ScaleByChromMax",
    "BinomialDownsample",
    "Identity",
    "EnsureFloat32",
    "get_log1p_pipeline",
    "get_minmax_pipeline",
    # Preprocessing functions
    "binomial_downsample",
    "create_downsampled_cooler",
    "create_transformed_cooler",
    "min_max_normalize",
    "min_max_normalize_simple",
    "denormalize",
    "clip_percentile",
    "handle_nan",
    "enforce_symmetry",
    "log_transform",
    "standardize",
    "fill_ps_background",
    # Backwards compatibility
    "ClipByContext",
    "ClipLogByContext",
]
