"""Command-line interface for xcut toolkit."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Literal

import click
import numpy as np

from xcut.data.preprocessing import create_downsampled_cooler


@click.group()
def cli() -> None:
    """xcut - Hi-C contact matrix enhancement toolkit."""
    pass


@cli.command()
@click.argument("config_path", type=click.Path(exists=True))
@click.option(
    "--set",
    "overrides",
    multiple=True,
    help="Override config values with dot-path notation. "
    "Example: --set train.lr=3e-4 --set data.batch_size=32",
)
@click.option(
    "--device",
    type=str,
    default=None,
    help="Device override (e.g., cuda, cuda:0, cuda:1, cpu). "
    "Shorthand for --set train.device=...",
)
def train(config_path: str, overrides: tuple[str, ...], device: str | None) -> None:
    """Train a model from a YAML configuration file.

    Loads the config, applies any --set overrides, then runs training.

    Examples:

        xcut train configs/hinet_gan_4x.yaml

        xcut train configs/hinet_gan_4x.yaml --device cuda:1

        xcut train configs/hinet_gan_4x.yaml --set train.lr=3e-4

        xcut train configs/hinet_gan_4x.yaml --set train.epochs=50 --set train.gan.pixel_weight=20
    """
    from xcut.config import RunConfig
    from xcut.training import train_from_config

    override_list = list(overrides) if overrides else []
    if device is not None:
        override_list.append(f"train.device={device}")

    click.secho(f"Loading config: {config_path}", fg="blue")
    if override_list:
        for o in override_list:
            click.secho(f"  Override: {o}", fg="yellow")

    cfg = RunConfig.load_with_overrides(config_path, override_list or None)

    click.secho(f"Run: {cfg.name}", fg="blue")
    click.secho(f"Model: {cfg.model.type}", fg="blue")
    click.secho(f"Device: {cfg.train.device}", fg="blue")
    click.secho(f"Output: {cfg.run_dir}", fg="blue")
    click.echo()

    train_from_config(cfg)


@cli.command()
@click.argument(
    "lr_cooler",
    type=str,
)
@click.option(
    "--checkpoint",
    required=True,
    type=click.Path(exists=True),
    help="Path to model checkpoint (.pth file).",
)
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True),
    default=None,
    help="Path to run config YAML. Default: auto-detect from checkpoint directory.",
)
@click.option(
    "-o",
    "--output",
    type=click.Path(),
    default=None,
    help="Output path for enhanced .cool file. Default: <checkpoint_dir>/enhanced.cool.",
)
@click.option(
    "--chroms",
    type=str,
    default=None,
    help="Comma-separated chromosomes (e.g. chr1,chr2). Default: all in LR cooler.",
)
@click.option(
    "--batch-size",
    type=int,
    default=16,
    show_default=True,
    help="Inference batch size.",
)
@click.option(
    "--hr-cooler",
    type=str,
    default=None,
    help="Path to HR cooler for inverse transform scaling (recommended). "
    "Without this, output counts will be underscaled.",
)
@click.option(
    "--device",
    type=str,
    default=None,
    help="Compute device (default: auto-detect).",
)
def enhance(
    lr_cooler: str,
    checkpoint: str,
    config_path: str | None,
    output: str | None,
    chroms: str | None,
    batch_size: int,
    hr_cooler: str | None,
    device: str | None,
) -> None:
    """Enhance a low-resolution cooler using a trained model.

    Takes a LR cooler file and produces an enhanced .cool file.
    Uses the transform pipeline from the training config for normalization
    and its inverse for converting predictions back to counts.

    The model outputs predictions in [0, 1]. To convert back to integer
    counts, the inverse transform needs a scale factor. Using --hr-cooler
    (recommended) provides the correct HR count scale. Without it, LR
    stats are used as fallback, which underestimates counts.

    Examples:

        # Recommended: with HR cooler for correct scaling
        xcut enhance data/lr.cool --checkpoint runs/my_run/best_model.pth \\
            --hr-cooler "data/hr.mcool::/resolutions/50000"

        # Without HR cooler (counts will be underscaled)
        xcut enhance data/lr.cool --checkpoint runs/my_run/best_model.pth

        # Specific chromosomes
        xcut enhance data/lr.cool --checkpoint runs/my_run/best_model.pth \\
            --hr-cooler data/hr.cool --chroms chr21,chr22
    """
    import torch

    from xcut.inference import create_enhanced_cooler

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Auto-detect config from checkpoint directory
    if config_path is None:
        ckpt_dir = Path(checkpoint).parent
        candidate = ckpt_dir / "config.yaml"
        if candidate.exists():
            config_path = str(candidate)
        else:
            click.secho(
                f"Error: No config.yaml found in {ckpt_dir}. "
                "Provide --config explicitly.",
                fg="red",
            )
            raise SystemExit(1)

    # Auto-generate output path
    if output is None:
        lr_stem = Path(lr_cooler.split("::")[0]).stem
        ckpt_dir = Path(checkpoint).parent
        output = str(ckpt_dir / f"enhanced_{lr_stem}.cool")

    chromosomes = [c.strip() for c in chroms.split(",")] if chroms else None

    click.secho(f"LR cooler:   {lr_cooler}", fg="blue")
    click.secho(f"Checkpoint:  {checkpoint}", fg="blue")
    click.secho(f"Config:      {config_path}", fg="blue")
    if hr_cooler:
        click.secho(f"HR cooler:   {hr_cooler} (for inverse scaling)", fg="blue")
    else:
        click.secho(f"HR cooler:   not provided (using LR stats fallback)", fg="yellow")
    click.secho(f"Output:      {output}", fg="blue")
    click.secho(f"Device:      {device}", fg="blue")
    if chromosomes:
        click.secho(f"Chromosomes: {', '.join(chromosomes)}", fg="blue")
    click.echo()

    start = time.time()
    result = create_enhanced_cooler(
        checkpoint_path=checkpoint,
        config_path=config_path,
        lr_cooler_path=lr_cooler,
        output_path=output,
        hr_cooler_path=hr_cooler,
        chromosomes=chromosomes,
        device=device,
        batch_size=batch_size,
    )
    elapsed = time.time() - start

    click.secho(f"Done in {elapsed:.1f}s", fg="green")
    click.secho(f" -> {result}", fg="green")


@cli.command("transform")
@click.argument(
    "input_path",
    type=str,
)
@click.option(
    "-o",
    "--output",
    type=click.Path(),
    default=None,
    help="Output path for transformed .cool file. Auto-generated if not provided.",
)
@click.option(
    "--clip",
    type=float,
    default=None,
    help="Clip at Nth percentile per chromosome (e.g. 99.95).",
)
@click.option(
    "--log",
    "apply_log",
    is_flag=True,
    default=False,
    help="Apply log1p transform.",
)
@click.option(
    "--normalize",
    is_flag=True,
    default=False,
    help="Divide by per-chromosome max (after clip/log). "
    "Warning: output will be float values rounded to int.",
)
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True),
    default=None,
    help="Apply exact transform pipeline from a training config YAML. "
    "Mutually exclusive with --clip/--log/--normalize.",
)
@click.option(
    "--chroms",
    type=str,
    default=None,
    help="Comma-separated chromosomes (default: all).",
)
@click.option(
    "--resolution",
    type=int,
    default=None,
    help="Resolution in bp. Required for .mcool files.",
)
@click.option(
    "--nonzero",
    is_flag=True,
    default=False,
    help="Compute percentiles over non-zero values only. "
    "Critical for sparse data (e.g. Pore-C) where >95%% zeros cause "
    "whole-matrix percentiles to collapse near zero. "
    "When using --config, this is read from the config automatically.",
)
def transform_cmd(
    input_path: str,
    output: str | None,
    clip: float | None,
    apply_log: bool,
    normalize: bool,
    config_path: str | None,
    chroms: str | None,
    resolution: int | None,
    nonzero: bool,
) -> None:
    """Apply transforms to a cooler file and write a new cooler.

    Useful for producing clipped/normalized coolers for fair comparison
    with model predictions. For example, clip both HR and LR coolers at
    the same percentile the model was trained with.

    Examples:

        # Clip at 99.95th percentile
        xcut transform data/hr.cool -o data/hr_clipped.cool --clip 99.95

        # Clip using nonzero percentiles (for sparse Pore-C data)
        xcut transform data/hr.cool -o data/hr_clipped.cool --clip 99.95 --nonzero

        # Match exact training pipeline from config
        xcut transform data/hr.cool -o data/hr_preprocessed.cool \\
            --config runs/my_run/config.yaml

        # Specific chromosomes
        xcut transform data/hr.cool --clip 99.95 --chroms chr1,chr2
    """
    from xcut.data.preprocessing import create_transformed_cooler

    # Validate mutual exclusivity
    manual_flags = clip is not None or apply_log or normalize
    if config_path and manual_flags:
        click.secho(
            "Error: --config is mutually exclusive with --clip/--log/--normalize.",
            fg="red",
        )
        raise SystemExit(1)
    if not config_path and not manual_flags:
        click.secho(
            "Error: Specify at least one of --clip, --log, --normalize, or --config.",
            fg="red",
        )
        raise SystemExit(1)

    # Resolve cooler path
    _validate_cooler_path(input_path)
    resolved_path = _resolve_cooler_path(input_path, resolution)

    # Build transforms
    if config_path:
        from xcut.config import RunConfig, resolve_transforms

        cfg = RunConfig.load(config_path)
        transforms = resolve_transforms(cfg.data.transforms)
        # Inherit nonzero_percentile from config unless explicitly set on CLI
        if not nonzero:
            nonzero = cfg.data.nonzero_percentile
        suffix = "config"
    else:
        from xcut.data.transforms import (
            ClipByPercentile,
            DivideByMax,
            EnsureFloat32,
            HandleNan,
            LogTransform,
        )

        transforms = [HandleNan(fill_value=0.0)]
        suffix_parts = []

        if clip is not None:
            transforms.append(ClipByPercentile(percentile=clip))
            suffix_parts.append(f"clip{clip}")

        if apply_log:
            transforms.append(LogTransform(pseudocount=1.0))
            suffix_parts.append("log1p")

        if normalize:
            pct = clip if clip is not None else None
            transforms.append(DivideByMax(percentile=pct))
            suffix_parts.append("norm")

        transforms.append(EnsureFloat32())
        suffix = "_".join(suffix_parts)

    # Auto-generate output path
    if output is None:
        base = Path(input_path.split("::")[0])
        output = str(base.parent / f"{base.stem}.{suffix}.cool")

    chromosomes = [c.strip() for c in chroms.split(",")] if chroms else None

    click.secho(f"Input:      {resolved_path}", fg="blue")
    click.secho(f"Output:     {output}", fg="blue")
    click.secho(f"Transforms: {[repr(t) for t in transforms]}", fg="blue")
    if nonzero:
        click.secho(f"Percentiles: nonzero only", fg="blue")
    if chromosomes:
        click.secho(f"Chromosomes: {', '.join(chromosomes)}", fg="blue")
    click.echo()

    start = time.time()
    result = create_transformed_cooler(
        input_cooler_path=resolved_path,
        output_path=output,
        transforms=transforms,
        chromosomes=chromosomes,
        nonzero_percentile=nonzero,
    )
    elapsed = time.time() - start

    click.secho(f"Done in {elapsed:.1f}s", fg="green")
    click.secho(f" -> {result}", fg="green")


@cli.command()
@click.argument(
    "input_path",
    type=str,
)
@click.option(
    "-o",
    "--output",
    type=click.Path(),
    default=None,
    help="Output path. Auto-generated if not provided.",
)
@click.option(
    "-r",
    "--ratio",
    type=float,
    default=16.0,
    help="Downsample ratio (e.g., 16.0 keeps ~1/16 of reads).",
)
@click.option(
    "--resolution",
    type=int,
    default=None,
    help="Resolution in bp. Required for .mcool files without explicit resolution path.",
)
@click.option(
    "--seed",
    type=int,
    default=42,
    help="Random seed for reproducibility.",
)
@click.option(
    "--chromosomes",
    type=str,
    default=None,
    help="Comma-separated chromosome names (default: all chromosomes).",
)
@click.option(
    "--include-trans/--cis-only",
    default=True,
    help="Include inter-chromosomal (trans) contacts in downsampling. "
    "Default includes trans, producing complete, balanceable coolers. "
    "Use --cis-only to exclude trans contacts.",
)
@click.option(
    "--force",
    is_flag=True,
    default=False,
    help="Skip confirmation prompt.",
)
def downsample(
    input_path: str,
    output: str | None,
    ratio: float,
    resolution: int | None,
    seed: int,
    chromosomes: str | None,
    include_trans: bool,
    force: bool,
) -> None:
    """Downsample a cooler file using Binomial subsampling.

    Each bin's count is treated as independent Bernoulli trials, producing
    statistically correct Poisson-like noise for simulating low-coverage
    sequencing from a high-coverage contact matrix.

    Examples:

        # Downsample a .cool file
        xcut downsample data/sample.cool

        # Downsample .mcool at specific resolution
        xcut downsample data/sample.mcool --resolution 10000

        # Explicit mcool path with resolution
        xcut downsample "data/sample.mcool::/resolutions/10000"

        # Custom output and ratio
        xcut downsample data/sample.cool -o downsampled.cool -r 100
    """
    input_str = str(input_path)

    # Validate that the base file exists (before :: if present)
    base_path = input_str.split("::")[0]
    if not Path(base_path).exists():
        click.secho(f"Error: File does not exist: {base_path}", fg="red")
        raise SystemExit(1)

    if ratio < 1.0:
        click.secho("Error: Ratio must be >= 1.0", fg="red")
        raise SystemExit(1)

    if ".mcool" in input_str.lower() and "::/resolutions/" not in input_str.lower():
        if resolution is None:
            click.secho("Error: .mcool files require --resolution option.", fg="red")
            click.secho(
                "Example: xcut downsample file.mcool --resolution 10000", fg="yellow"
            )
            click.secho(
                "Or use explicit path: xcut downsample 'file.mcool::/resolutions/10000'",
                fg="yellow",
            )
            raise SystemExit(1)
        input_str = f"{input_str}::/resolutions/{resolution}"

    input_path_obj = Path(input_path)

    if output is None:
        base_name = input_path_obj.name.split("::")[0]
        stem = Path(base_name).stem
        parent = input_path_obj.parent if "::" not in str(input_path) else Path(".")
        output = str(parent / f"{stem}.{int(ratio)}x.binomial.down.cool")

    output_path = Path(output)

    if output_path.exists() and not force:
        if not click.confirm(
            click.style(f"Output file exists: {output}\nOverwrite?", fg="yellow"),
            default=False,
        ):
            click.secho("Aborted.", fg="red")
            raise SystemExit(0)

    chrom_list = None
    if chromosomes:
        chrom_list = [c.strip() for c in chromosomes.split(",")]

    click.secho(f"Input:    {input_str}", fg="blue")
    click.secho(f"Output:   {output}", fg="blue")
    click.secho(f"Ratio:    {ratio}x", fg="blue")
    click.secho(f"Seed:     {seed}", fg="blue")
    click.secho(f"Trans:    {'included' if include_trans else 'excluded'}", fg="blue")
    if chrom_list:
        click.secho(f"Chroms:   {', '.join(chrom_list)}", fg="blue")
    click.echo()

    start = time.time()
    result = create_downsampled_cooler(
        input_cooler_path=input_str,
        output_path=output,
        ratio=ratio,
        seed=seed,
        chromosomes=chrom_list,
        include_trans=include_trans,
    )
    elapsed = time.time() - start

    click.secho(f"Done in {elapsed:.1f}s", fg="green")
    click.secho(f" -> {result}", fg="green")


@cli.command()
@click.argument(
    "cooler_path",
    type=str,
)
@click.option(
    "-c",
    "--chromosome",
    type=str,
    required=True,
    help="Chromosome name (e.g., chr1)",
)
@click.option(
    "-s",
    "--start",
    type=int,
    required=True,
    help="Start position in base pairs",
)
@click.option(
    "-e",
    "--end",
    type=int,
    required=True,
    help="End position in base pairs",
)
@click.option(
    "--resolution",
    type=int,
    default=None,
    help="Resolution in bp. Required for .mcool.",
)
@click.option(
    "--cmap",
    type=str,
    default="RdYlBu_r",
    help="Matplotlib colormap",
)
@click.option(
    "-o",
    "--output",
    type=click.Path(),
    default=None,
    help="Output path. Auto-generated if not provided.",
)
@click.option(
    "-f",
    "--format",
    type=str,
    default="png",
    help="Output format: png, pdf, svg, jpg",
)
@click.option(
    "--dpi",
    type=int,
    default=300,
    help="DPI for raster formats",
)
@click.option(
    "--log",
    is_flag=True,
    default=False,
    help="Apply log1p transform",
)
@click.option(
    "--clip",
    type=float,
    default=None,
    help="Clip values at given percentile (e.g., 99.9, 99.95, 99.99).",
)
@click.option(
    "--clip-nonzero",
    is_flag=True,
    default=False,
    help="Compute clip percentile from non-zero values only.",
)
@click.option(
    "--title",
    type=str,
    default=None,
    help="Custom figure title (overrides default).",
)
@click.option(
    "--mark",
    type=str,
    default=None,
    help="Mark a genomic region with a line and label. Format: START-END:LABEL",
)
def viz(
    cooler_path: str,
    chromosome: str,
    start: int,
    end: int,
    resolution: int | None,
    cmap: str,
    output: str | None,
    format: str,
    dpi: int,
    log: bool,
    clip: float | None,
    clip_nonzero: bool,
    title: str | None,
    mark: str | None,
) -> None:
    """Plot a genomic interval from a cooler file.

    Examples:

        xcut viz data/sample.cool -c chr1 -s 0 -e 1000000

        xcut viz data/sample.mcool -c chr1 -s 0 -e 1000000 --resolution 10000

        xcut viz data/sample.mcool -c chr1 -s 0 -e 1000000 -o plot.png --cmap viridis

        xcut viz data/sample.cool -c chrX -s 6000000 -e 7000000 --mark 6296494-6297792:rex-33
    """
    import cooler
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    _validate_cooler_path(cooler_path)
    resolved_path = _resolve_cooler_path(cooler_path, resolution)

    if output is None:
        chrom_part = chromosome.replace("chr", "")
        output = str(
            Path(cooler_path).parent
            / f"{Path(cooler_path).stem}_{chrom_part}_{start}_{end}.{format}"
        )

    click.secho(f"Loading: {resolved_path}", fg="blue")
    click.secho(f"Region: {chromosome}:{start:,}-{end:,}", fg="blue")
    click.secho(f"Resolution: {resolution or 'from file'}", fg="blue")

    clr = cooler.Cooler(resolved_path)
    matrix = clr.matrix(balance=False).fetch(
        f"{chromosome}:{start}-{end}", f"{chromosome}:{start}-{end}"
    )
    matrix = matrix.astype(np.float32)
    matrix = np.nan_to_num(matrix, nan=0.0)

    # Track clipping info for annotation
    original_max = float(matrix.max())
    clipped_max = None

    if clip is not None:
        if clip_nonzero:
            nonzero = matrix[matrix > 0]
            clip_val = float(np.percentile(nonzero, clip)) if len(nonzero) > 0 else 0.0
        else:
            clip_val = float(np.percentile(matrix, clip))
        matrix = np.clip(matrix, a_min=0, a_max=clip_val)
        clipped_max = float(matrix.max())

    if log:
        matrix = np.log1p(matrix)

    if clip is not None:
        mx = matrix.max()
        if mx > 0:
            matrix = matrix / mx

    effective_res = resolution or clr.binsize
    assert effective_res is not None, "Could not determine resolution"

    # Build title
    if title is None:
        title_suffix = ""
        if log:
            title_suffix = " (log1p)"
        if clip is not None:
            nz_tag = ", nonzero" if clip_nonzero else ""
            title_suffix += f" [clip {clip}{nz_tag}]"
        title = f"{chromosome}: {start:,}-{end:,}{title_suffix}"

    n = matrix.shape[0]
    mb_scale_x = np.linspace(start / 1_000_000, end / 1_000_000, n)
    mb_scale_y = np.linspace(start / 1_000_000, end / 1_000_000, n)

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(matrix, cmap=cmap, interpolation="nearest")

    ax.set_title(title)

    ticks = np.linspace(0, n - 1, 8)
    mb_labels_x = [f"{mb_scale_x[int(t)]:.2f}" for t in ticks]
    mb_labels_y = [f"{mb_scale_y[int(t)]:.2f}" for t in ticks]

    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(mb_labels_x)
    ax.set_yticklabels(mb_labels_y)
    ax.set_xlabel("Position (Mb)")

    # Position label on left y-axis
    region_label = f"{chromosome}:{start:,}-{end:,}"
    ax.set_ylabel(region_label)

    # Clipping annotation in top-right corner
    if clipped_max is not None:
        annotation = f"max: {original_max:.0f}\nclipped: {clipped_max:.0f}"
        ax.text(
            0.98,
            0.98,
            annotation,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8,
            color="white",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.6),
        )

    # Mark region with diagonal line segment and label
    if mark is not None:
        try:
            parts = mark.split(":")
            coords = parts[0]
            label = parts[1] if len(parts) > 1 else None
            mstart, mend = map(int, coords.split("-"))
            mcenter = (mstart + mend) / 2
            if start <= mcenter <= end:
                px = (mcenter - start) / (end - start) * n
                segment_len = n * 0.15
                ax.plot(
                    [px - segment_len, px + segment_len],
                    [px + segment_len, px - segment_len],
                    color="black",
                    linestyle="--",
                    linewidth=2.5,
                    alpha=0.9,
                )
                if label:
                    ax.text(
                        px + segment_len + 3,
                        px - segment_len,
                        label,
                        color="black",
                        ha="left",
                        va="center",
                        fontsize=10,
                        fontweight="bold",
                    )
            else:
                click.secho(
                    f"Warning: mark position {mcenter:,} outside region {start:,}-{end:,}",
                    fg="yellow",
                )
        except (ValueError, IndexError):
            click.secho(
                f"Warning: invalid --mark format '{mark}'. Expected START-END:LABEL",
                fg="yellow",
            )

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.1)
    plt.colorbar(im, cax=cax)

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    click.secho(f"Saved: {output}", fg="green")


@cli.command("compare")
@click.argument(
    "cooler_path1",
    type=str,
)
@click.argument(
    "cooler_path2",
    type=str,
)
@click.option(
    "-c",
    "--chromosome",
    type=str,
    required=True,
    help="Chromosome name (e.g., chr1)",
)
@click.option(
    "-s",
    "--start",
    type=int,
    required=True,
    help="Start position in base pairs",
)
@click.option(
    "-e",
    "--end",
    type=int,
    required=True,
    help="End position in base pairs",
)
@click.option(
    "--resolution",
    type=int,
    default=None,
    help="Resolution in bp. Required for .mcool.",
)
@click.option(
    "--cmap",
    type=str,
    default="RdYlBu_r",
    help="Matplotlib colormap",
)
@click.option(
    "-o",
    "--output",
    type=click.Path(),
    default=None,
    help="Output path. Auto-generated if not provided.",
)
@click.option(
    "-f",
    "--format",
    type=str,
    default="png",
    help="Output format: png, pdf, svg, jpg",
)
@click.option(
    "--dpi",
    type=int,
    default=300,
    help="DPI for raster formats",
)
@click.option(
    "--title1",
    type=str,
    default=None,
    help="Title for first panel",
)
@click.option(
    "--title2",
    type=str,
    default=None,
    help="Title for second panel",
)
@click.option(
    "--log",
    is_flag=True,
    default=False,
    help="Apply log1p transform",
)
@click.option(
    "--clip",
    type=float,
    default=None,
    help="Clip values at given percentile (e.g., 99.9, 99.95, 99.99).",
)
def compare_cmd(
    cooler_path1: str,
    cooler_path2: str,
    chromosome: str,
    start: int,
    end: int,
    resolution: int | None,
    cmap: str,
    output: str | None,
    format: str,
    dpi: int,
    title1: str | None,
    title2: str | None,
    log: bool,
    clip: float | None,
) -> None:
    """Compare two cooler files side by side.

    Computes metrics (PSNR, SSIM, MSE) between the two matrices.

    Examples:

        xcut compare data/lr.cool data/hr.cool -c chr1 -s 0 -e 1000000

        xcut compare data/pred.mcool data/hr.mcool -c chr1 -s 0 -e 1000000 --resolution 10000
    """
    import cooler
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from skimage.metrics import structural_similarity as ssim

    _validate_cooler_path(cooler_path1)
    _validate_cooler_path(cooler_path2)
    resolved_path1 = _resolve_cooler_path(cooler_path1, resolution)
    resolved_path2 = _resolve_cooler_path(cooler_path2, resolution)

    if output is None:
        chrom_part = chromosome.replace("chr", "")
        output = str(
            Path(cooler_path1).parent / f"compare_{chrom_part}_{start}_{end}.{format}"
        )

    click.secho("Loading:", fg="blue")
    click.secho(f"  1: {resolved_path1}", fg="blue")
    click.secho(f"  2: {resolved_path2}", fg="blue")
    click.secho(f"Region: {chromosome}:{start:,}-{end:,}", fg="blue")

    clr1 = cooler.Cooler(resolved_path1)
    clr2 = cooler.Cooler(resolved_path2)

    matrix1 = clr1.matrix(balance=False).fetch(
        f"{chromosome}:{start}-{end}", f"{chromosome}:{start}-{end}"
    )
    matrix2 = clr2.matrix(balance=False).fetch(
        f"{chromosome}:{start}-{end}", f"{chromosome}:{start}-{end}"
    )

    matrix1 = matrix1.astype(np.float32)
    matrix2 = matrix2.astype(np.float32)
    matrix1 = np.nan_to_num(matrix1, nan=0.0)
    matrix2 = np.nan_to_num(matrix2, nan=0.0)

    if clip is not None:
        vmax1 = np.percentile(matrix1, clip)
        vmax2 = np.percentile(matrix2, clip)
        matrix1 = np.clip(matrix1, a_min=0, a_max=vmax1)
        matrix2 = np.clip(matrix2, a_min=0, a_max=vmax2)

    if log:
        matrix1 = np.log1p(matrix1)
        matrix2 = np.log1p(matrix2)

    if clip is not None:
        matrix1 = matrix1 / matrix1.max()
        matrix2 = matrix2 / matrix2.max()

    mse = float(np.mean((matrix1 - matrix2) ** 2))
    if mse == 0:
        psnr = float("inf")
    else:
        max_pixel = max(matrix1.max(), matrix2.max())
        psnr = 20 * np.log10(max_pixel / np.sqrt(mse))

    ssim_val = ssim(matrix1, matrix2, data_range=max(matrix1.max(), matrix2.max()))

    click.secho(
        f"Metrics: PSNR: {psnr:.3f}, SSIM: {ssim_val:.3f}, MSE: {mse:.4f}", fg="blue"
    )

    name1 = title1 or Path(cooler_path1).name.split("::")[0]
    name2 = title2 or Path(cooler_path2).name.split("::")[0]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    vmin = min(matrix1.min(), matrix2.min())
    vmax = max(matrix1.max(), matrix2.max())

    n = matrix1.shape[0]
    mb_scale_x = np.linspace(start / 1_000_000, end / 1_000_000, n)
    mb_scale_y = np.linspace(start / 1_000_000, end / 1_000_000, n)

    im = None
    for ax, matrix, name in zip(axes, [matrix1, matrix2], [name1, name2]):
        im = ax.imshow(matrix, cmap=cmap, interpolation="nearest", vmin=vmin, vmax=vmax)
        ax.set_title(name)

        ticks = np.linspace(0, n - 1, 8)
        mb_labels_x = [f"{mb_scale_x[int(t)]:.2f}" for t in ticks]
        mb_labels_y = [f"{mb_scale_y[int(t)]:.2f}" for t in ticks]

        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels(mb_labels_x)
        ax.set_yticklabels(mb_labels_y)
        ax.set_xlabel("Position (Mb)")
        ax.set_ylabel("Position (Mb)")

    axes[0].set_xlabel(f"PSNR: {psnr:.3f}, SSIM: {ssim_val:.3f}, MSE: {mse:.4f}")

    divider = make_axes_locatable(axes[-1])
    cax = divider.append_axes("right", size="3%", pad=0.1)
    plt.colorbar(im, cax=cax)

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    click.secho(f"Saved: {output}", fg="green")


def _validate_cooler_path(path: str) -> None:
    """Validate that the base file of a cooler path exists."""
    base_path = path.split("::")[0]
    if not Path(base_path).exists():
        click.secho(f"Error: File does not exist: {base_path}", fg="red")
        raise SystemExit(1)


def _resolve_cooler_path(path: str, resolution: int | None) -> str:
    """Resolve cooler path, handling .mcool files and resolution."""
    if ".mcool" in path.lower() and "::/resolutions/" not in path.lower():
        if resolution is None:
            click.secho("Error: .mcool files require --resolution option.", fg="red")
            raise SystemExit(1)
        path = f"{path}::/resolutions/{resolution}"
    return path


if __name__ == "__main__":
    cli()
