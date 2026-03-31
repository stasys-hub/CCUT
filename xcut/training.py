"""Config-driven training entrypoint.

Bridges RunConfig → datasets → model registry → training loop.

Usage:
    from xcut.config import RunConfig
    from xcut.training import train_from_config

    cfg = RunConfig.load("configs/hinet_gan_4x.yaml")
    train_from_config(cfg)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from xcut.config import RunConfig, resolve_transforms
from xcut.data import CoolerDataset, FixedSizeWrapper, SingleCoolerDataset, WindowConfig
from xcut.metrics import psnr_t as calculate_psnr, ssim_t as calculate_ssim
from xcut.registry import build_model


def _build_datasets(cfg: RunConfig) -> tuple[DataLoader, DataLoader]:
    """Build train and validation DataLoaders from RunConfig."""
    data = cfg.data
    transforms = resolve_transforms(data.transforms)

    window_config = WindowConfig(
        window_size=data.window_size,
        resolution=data.resolution,
        step=data.step,
    )

    # Auto-detect percentiles from transforms if not explicitly set
    percentiles = data.percentiles

    shared_kwargs = dict(
        window_config=window_config,
        transforms=transforms,
        percentiles=percentiles,
        sym_coor=data.sym_coor,
        max_distance=data.max_distance,
        zero_threshold=data.zero_threshold,
        n_processes=data.n_processes,
        nonzero_percentile=data.nonzero_percentile,
    )

    if data.lr_cooler is not None:
        # Two-cooler mode
        train_ds = CoolerDataset(
            lr_cooler_path=data.lr_cooler,
            hr_cooler_path=data.hr_cooler,
            chrom_range=data.chromosomes,
            **shared_kwargs,
        )
        val_ds = CoolerDataset(
            lr_cooler_path=data.lr_cooler,
            hr_cooler_path=data.hr_cooler,
            chrom_range=data.val_chromosomes,
            **shared_kwargs,
        )
    else:
        # Single-cooler + binomial downsampling
        train_ds = SingleCoolerDataset(
            cooler_path=data.hr_cooler,
            downsample_ratio=data.downsample_ratio,
            stochastic=data.stochastic,
            seed=cfg.train.seed,
            chrom_range=data.chromosomes,
            **shared_kwargs,
        )
        val_ds = SingleCoolerDataset(
            cooler_path=data.hr_cooler,
            downsample_ratio=data.downsample_ratio,
            stochastic=False,
            seed=cfg.train.seed,
            chrom_range=data.val_chromosomes,
            **shared_kwargs,
        )

    # Wrap for fixed output size
    train_ds = FixedSizeWrapper(train_ds, size=data.window_size)
    val_ds = FixedSizeWrapper(val_ds, size=data.window_size)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.train.num_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.train.num_workers,
    )

    return train_loader, val_loader


def train_from_config(cfg: RunConfig) -> None:
    """Run training from a RunConfig.

    Steps:
        1. Save resolved config to run_dir/config.yaml
        2. Build datasets and dataloaders
        3. Build model(s) via registry
        4. Run appropriate training loop
    """
    run_dir = cfg.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save the fully resolved config for reproducibility
    cfg.save(run_dir / "config.yaml")

    device = torch.device(cfg.train.device)
    print(f"Run directory: {run_dir}")
    print(f"Device: {device}")

    # --- Seed ---
    torch.manual_seed(cfg.train.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(cfg.train.seed)

    # --- Data ---
    print("Building datasets...")
    train_loader, val_loader = _build_datasets(cfg)
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    # --- Model ---
    print(f"Building model: {cfg.model.type}")
    models = build_model(cfg.model, device)
    generator = models["generator"]
    print(f"Generator params: {sum(p.numel() for p in generator.parameters()):,}")

    # --- Training ---
    if cfg.model.type == "rectified_flow":
        _train_flow(cfg, generator, models.get("backbone"), train_loader, val_loader, device)
    elif cfg.model.type == "hinet_gan" or cfg.train.gan.enabled:
        _train_gan(cfg, generator, models.get("discriminator"), train_loader, val_loader, device)
    else:
        raise NotImplementedError(
            f"Supervised training loop not yet implemented for model type '{cfg.model.type}'. "
            "Use 'hinet_gan' or set train.gan.enabled=true."
        )

    print(f"Training complete. Results in: {run_dir}")


def _train_gan(
    cfg: RunConfig,
    generator: torch.nn.Module,
    discriminator: torch.nn.Module | None,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
) -> None:
    """Run GAN training using the existing train_gan() function."""
    from xcut.models.hinet_gan import Discriminator, train_gan

    if discriminator is None:
        in_chn = cfg.model.params.get("in_chn", 1)
        discriminator = Discriminator(in_channels=in_chn).to(device)

    print(f"Discriminator params: {sum(p.numel() for p in discriminator.parameters()):,}")

    run_dir = cfg.run_dir
    plot_dir = run_dir / "plots"
    log_every = cfg.train.log_images_every

    def on_epoch_end(epoch, gen, metrics, dev):
        if log_every > 0 and (epoch % log_every == 0 or epoch == 0):
            plot_samples(epoch, gen, val_loader, dev, plot_dir, n_samples=4, tag="val")

    # Plot initial state (untrained)
    plot_samples(0, generator, val_loader, device, plot_dir, n_samples=4, tag="val_init")

    gan_cfg = cfg.train.gan
    generator, discriminator = train_gan(
        generator=generator,
        discriminator=discriminator,
        train_loader=train_loader,
        test_loader=val_loader,
        num_epochs=cfg.train.epochs,
        device=device,
        metric=cfg.train.metric,
        patience=cfg.train.patience,
        save_path=str(run_dir / "best_model.pth"),
        log_dir=str(run_dir / "logs"),
        on_epoch_end=on_epoch_end,
        lr_g=cfg.train.lr,
        lr_d=gan_cfg.lr_d,
        betas=(cfg.train.beta1, cfg.train.beta2),
        grad_clip=cfg.train.grad_clip,
        pixel_weight=gan_cfg.pixel_weight,
        structure_weight=gan_cfg.structure_weight,
        gp_weight=gan_cfg.gp_weight,
        label_smoothing=gan_cfg.label_smoothing,
        noise_std=gan_cfg.noise_std,
    )

    # Final plot
    plot_samples("final", generator, val_loader, device, plot_dir, n_samples=4, tag="val_final")


def _train_flow(
    cfg: RunConfig,
    diffusion: torch.nn.Module,
    backbone: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
) -> None:
    """Train rectified flow model."""
    import csv
    import numpy as np
    from datetime import datetime
    from tqdm import tqdm
    from xcut.models.losses import structure_consistency_loss, insulation_loss, distance_decay_loss

    run_dir = cfg.run_dir
    plot_dir = run_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    log_dir = run_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    flow_cfg = cfg.train.flow
    print(f"  lr={cfg.train.lr}, weight_decay={flow_cfg.weight_decay}, "
          f"structure_weight={flow_cfg.structure_weight}, "
          f"insulation_weight={flow_cfg.insulation_weight}, "
          f"decay_weight={flow_cfg.decay_weight}")

    optimizer = torch.optim.AdamW(
        backbone.parameters(), lr=cfg.train.lr, weight_decay=flow_cfg.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.train.epochs, eta_min=1e-6
    )

    # CSV logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = log_dir / f"training_log_{timestamp}.csv"
    csv_header = ["epoch", "train_loss", "val_ssim", "val_psnr", "lr"]
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow(csv_header)

    # Early stopping
    best_val_ssim = -float("inf")
    no_improvement = 0
    log_every = cfg.train.log_images_every

    # Plot initial state
    plot_samples_flow(0, diffusion, val_loader, device, plot_dir, n_samples=4, tag="val_init")

    for epoch in range(cfg.train.epochs):
        diffusion.train()
        epoch_loss = 0.0
        n_batches = 0

        train_bar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for data in train_bar:
            lr_batch = data["lr"].to(device)
            hr_batch = data["hr"].to(device)

            optimizer.zero_grad()

            # Core flow matching loss
            loss = diffusion(hr_batch, lr_batch)

            # Auxiliary losses on predicted x0
            use_aux = (flow_cfg.structure_weight > 0
                       or flow_cfg.insulation_weight > 0
                       or flow_cfg.decay_weight > 0)
            if use_aux:
                _, pred_x0 = diffusion.forward_with_pred(hr_batch, lr_batch)
                if flow_cfg.structure_weight > 0:
                    loss = loss + flow_cfg.structure_weight * structure_consistency_loss(pred_x0, hr_batch)
                if flow_cfg.insulation_weight > 0:
                    loss = loss + flow_cfg.insulation_weight * insulation_loss(pred_x0, hr_batch)
                if flow_cfg.decay_weight > 0:
                    loss = loss + flow_cfg.decay_weight * distance_decay_loss(pred_x0, hr_batch)

            loss.backward()
            if cfg.train.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(backbone.parameters(), max_norm=cfg.train.grad_clip)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1
            train_bar.set_postfix({"loss": f"{loss.item():.4f}"})

            if not torch.isfinite(loss):
                print("Training stopped due to infinite loss")
                return

        scheduler.step()
        epoch_loss /= n_batches

        # Validation
        diffusion.eval()
        val_ssim = 0.0
        val_psnr = 0.0
        val_count = 0

        with torch.no_grad():
            for data in val_loader:
                lr_batch = data["lr"].to(device)
                hr_batch = data["hr"].to(device)
                pred = diffusion.sample(batch_size=lr_batch.shape[0], condition=lr_batch)
                val_ssim += calculate_ssim(pred, hr_batch).item()
                val_psnr += calculate_psnr(pred, hr_batch).item()
                val_count += 1

        val_ssim /= val_count
        val_psnr /= val_count
        current_lr = scheduler.get_last_lr()[0]

        # Log
        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, epoch_loss, val_ssim, val_psnr, current_lr])

        # Best model
        if val_ssim > best_val_ssim:
            best_val_ssim = val_ssim
            no_improvement = 0
            torch.save(
                {
                    "model_state_dict": backbone.state_dict(),
                    "epoch": epoch,
                    "ssim": val_ssim,
                    "psnr": val_psnr,
                },
                str(run_dir / "best_model.pth"),
            )
            print(f"Saved best model with SSIM: {val_ssim:.4f}, PSNR: {val_psnr:.4f}")
        else:
            no_improvement += 1

        print(
            f"Epoch {epoch}: loss={epoch_loss:.4f}, SSIM={val_ssim:.4f}, "
            f"PSNR={val_psnr:.4f}, lr={current_lr:.2e}"
        )

        if log_every > 0 and (epoch % log_every == 0 or epoch == 0):
            plot_samples_flow(epoch, diffusion, val_loader, device, plot_dir, n_samples=4, tag="val")

        if no_improvement >= cfg.train.patience:
            print(f"Early stopping triggered at epoch {epoch}")
            break

    # Final plot
    plot_samples_flow("final", diffusion, val_loader, device, plot_dir, n_samples=4, tag="val_final")


def plot_samples_flow(
    epoch: int | str,
    diffusion: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    save_dir: Path,
    n_samples: int = 4,
    tag: str = "val",
) -> None:
    """Plot samples for flow models (uses diffusion.sample instead of generator forward)."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    diffusion.eval()
    batch = next(iter(loader))
    lr = batch["lr"].to(device)
    hr = batch["hr"].to(device)

    with torch.no_grad():
        pred = diffusion.sample(batch_size=lr.shape[0], condition=lr)

    n = min(n_samples, lr.shape[0])
    fig, axes = plt.subplots(n, 3, figsize=(15, 4 * n), squeeze=False)

    for i in range(n):
        ssim_pred = float(calculate_ssim(pred[i : i + 1], hr[i : i + 1]))
        psnr_pred = float(calculate_psnr(pred[i : i + 1], hr[i : i + 1]))
        ssim_lr = float(calculate_ssim(lr[i : i + 1], hr[i : i + 1]))
        psnr_lr = float(calculate_psnr(lr[i : i + 1], hr[i : i + 1]))

        axes[i, 0].imshow(lr[i, 0].cpu().numpy(), cmap="RdYlBu_r", interpolation="nearest")
        axes[i, 0].set_title(f"LR\nSSIM={ssim_lr:.3f}  PSNR={psnr_lr:.1f}", fontsize=9)
        axes[i, 0].axis("off")

        axes[i, 1].imshow(pred[i, 0].cpu().numpy(), cmap="RdYlBu_r", interpolation="nearest")
        axes[i, 1].set_title(f"Prediction\nSSIM={ssim_pred:.3f}  PSNR={psnr_pred:.1f}", fontsize=9)
        axes[i, 1].axis("off")

        axes[i, 2].imshow(hr[i, 0].cpu().numpy(), cmap="RdYlBu_r", interpolation="nearest")
        axes[i, 2].set_title("HR (target)", fontsize=9)
        axes[i, 2].axis("off")

    fig.suptitle(f"Epoch {epoch} — {tag}", fontsize=13, y=1.01)
    fig.tight_layout()
    epoch_str = f"{epoch:03d}" if isinstance(epoch, int) else str(epoch)
    fig.savefig(save_dir / f"{tag}_epoch_{epoch_str}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {tag} plot: {save_dir / f'{tag}_epoch_{epoch_str}.png'}")


def plot_samples(
    epoch: int | str,
    generator: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    save_dir: Path,
    n_samples: int = 4,
    tag: str = "val",
) -> None:
    """Plot n_samples 3-panel comparisons: LR | Prediction | HR with metrics."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    generator.eval()
    batch = next(iter(loader))
    lr = batch["lr"].to(device)
    hr = batch["hr"].to(device)

    with torch.no_grad():
        pred = generator(lr)[1]

    n = min(n_samples, lr.shape[0])
    fig, axes = plt.subplots(n, 3, figsize=(15, 4 * n), squeeze=False)

    for i in range(n):
        lr_np = lr[i, 0].cpu().numpy()
        pred_np = pred[i, 0].cpu().numpy()
        hr_np = hr[i, 0].cpu().numpy()

        ssim_pred = float(calculate_ssim(pred[i : i + 1], hr[i : i + 1]))
        psnr_pred = float(calculate_psnr(pred[i : i + 1], hr[i : i + 1]))
        mse_pred = float(F.mse_loss(pred[i : i + 1], hr[i : i + 1]))

        ssim_lr = float(calculate_ssim(lr[i : i + 1], hr[i : i + 1]))
        psnr_lr = float(calculate_psnr(lr[i : i + 1], hr[i : i + 1]))
        mse_lr = float(F.mse_loss(lr[i : i + 1], hr[i : i + 1]))

        axes[i, 0].imshow(lr_np, cmap="RdYlBu_r", interpolation="nearest")
        axes[i, 0].set_title(
            f"LR\nSSIM={ssim_lr:.3f}  PSNR={psnr_lr:.1f}  MSE={mse_lr:.4f}",
            fontsize=9,
        )
        axes[i, 0].axis("off")

        axes[i, 1].imshow(pred_np, cmap="RdYlBu_r", interpolation="nearest")
        axes[i, 1].set_title(
            f"Prediction\nSSIM={ssim_pred:.3f}  PSNR={psnr_pred:.1f}  MSE={mse_pred:.4f}",
            fontsize=9,
        )
        axes[i, 1].axis("off")

        axes[i, 2].imshow(hr_np, cmap="RdYlBu_r", interpolation="nearest")
        axes[i, 2].set_title("HR (target)", fontsize=9)
        axes[i, 2].axis("off")

    fig.suptitle(f"Epoch {epoch} — {tag}", fontsize=13, y=1.01)
    fig.tight_layout()
    epoch_str = f"{epoch:03d}" if isinstance(epoch, int) else str(epoch)
    fig.savefig(save_dir / f"{tag}_epoch_{epoch_str}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {tag} plot: {save_dir / f'{tag}_epoch_{epoch_str}.png'}")
