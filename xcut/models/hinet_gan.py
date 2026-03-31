import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import numpy as np
import csv
from datetime import datetime
from tqdm import tqdm

from xcut.metrics import ssim_t as calculate_ssim, psnr_t as calculate_psnr
from xcut.models.losses import structure_consistency_loss


class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, stride=2, normalize=True):
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=stride, padding=1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels, 64, normalize=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512, stride=1),
            nn.Conv2d(512, 1, 4, padding=1),
            nn.AdaptiveAvgPool2d(1),  # Global average pooling to get final shape
            nn.Flatten(),  # Flatten to [batch_size, 1]
        )

    def forward(self, img):
        return self.model(img)


class EarlyStopping:
    def __init__(self, patience=7, mode="max", min_delta=0):
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.val_loss_min = np.inf if mode == "min" else -np.inf

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif (self.mode == "min" and val_loss > self.best_loss + self.min_delta) or (
            self.mode == "max" and val_loss < self.best_loss - self.min_delta
        ):
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
        return self.early_stop


def compute_gradient_penalty(discriminator, real_samples, fake_samples, device):
    alpha = torch.rand(real_samples.size(0), 1, 1, 1).to(device)
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(
        True
    )
    d_interpolates = discriminator(interpolates)
    fake = torch.ones(real_samples.size(0), 1).to(device)
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


def add_gaussian_noise(x, std=0.01):
    return x + torch.randn_like(x) * std


def train_gan(
    generator,
    discriminator,
    train_loader,
    test_loader,
    num_epochs,
    device,
    metric="ssim",
    patience=7,
    save_path="best_model.pth",
    log_dir="training_logs",
    on_epoch_end=None,
    lr_g=1e-4,
    lr_d=1e-4,
    betas=(0.5, 0.999),
    grad_clip=1.0,
    pixel_weight=40.0,
    structure_weight=10.0,
    gp_weight=5.0,
    label_smoothing=0.9,
    noise_std=0.1,
):
    """Train GAN with stabilization techniques.

    Args:
        on_epoch_end: Optional callback called after each epoch with signature:
            on_epoch_end(epoch, generator, metrics_dict, device)
            where metrics_dict has keys: g_loss, d_loss, val_ssim, val_psnr.
        lr_g: Generator learning rate.
        lr_d: Discriminator learning rate.
        betas: Adam betas tuple.
        grad_clip: Max gradient norm (0 = disabled).
        pixel_weight: Weight for L1 pixel loss in generator objective.
        structure_weight: Weight for structure consistency loss.
        gp_weight: Weight for gradient penalty (WGAN-GP).
        label_smoothing: Label smoothing factor for discriminator targets.
        noise_std: Std of noise added to discriminator predictions.
    """
    os.makedirs(log_dir, exist_ok=True)

    # Initialize CSV logger
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(log_dir, f"training_log_{timestamp}.csv")
    csv_header = [
        "epoch",
        "g_loss",
        "d_loss",
        "pixel_loss",
        "ssim",
        "psnr",
        "val_ssim",
        "val_psnr",
    ]

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(csv_header)

    print(f"  lr_g={lr_g}, lr_d={lr_d}, pixel_weight={pixel_weight}, "
          f"structure_weight={structure_weight}, gp_weight={gp_weight}")

    optimizer_G = optim.Adam(generator.parameters(), lr=lr_g, betas=betas)
    optimizer_D = optim.Adam(discriminator.parameters(), lr=lr_d, betas=betas)

    # Loss functions with label smoothing
    def discriminator_loss(pred, target):
        target = target * label_smoothing
        return F.binary_cross_entropy_with_logits(pred, target)

    criterion_pixel = nn.L1Loss()  # L1 loss instead of MSE for better stability

    # Early stopping setup
    mode = "max" if metric == "ssim" else "min"
    early_stopping = EarlyStopping(patience=patience, mode=mode)

    # Training loop
    for epoch in range(num_epochs):
        generator.train()
        discriminator.train()

        epoch_g_loss = 0
        epoch_d_loss = 0
        epoch_pixel_loss = 0
        epoch_ssim = 0
        epoch_psnr = 0
        n_batches = 0

        train_bar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for data in train_bar:
            # Get batch
            noisy_imgs = data["lr"].to(device)
            clean_imgs = data["hr"].to(device)
            batch_size = noisy_imgs.size(0)

            # Ground truths with noise
            valid = torch.ones((batch_size, 1), device=device)
            fake = torch.zeros((batch_size, 1), device=device)

            # ---------------------
            #  Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()

            # Generate denoised images
            with torch.no_grad():
                gen_output = generator(noisy_imgs)
                gen_imgs = gen_output[1]  # Using the second output as final

            # Real loss with noise
            pred_real = discriminator(clean_imgs)
            noise = torch.randn_like(pred_real) * noise_std
            loss_real = discriminator_loss(pred_real + noise, valid)

            # Fake loss with noise
            pred_fake = discriminator(gen_imgs.detach())
            noise = torch.randn_like(pred_fake) * noise_std
            loss_fake = discriminator_loss(pred_fake + noise, fake)

            # Gradient penalty
            gradient_penalty = compute_gradient_penalty(
                discriminator, clean_imgs, gen_imgs.detach(), device
            )

            # Total discriminator loss
            loss_D = (loss_real + loss_fake) / 2 + gp_weight * gradient_penalty

            loss_D.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=grad_clip)
            optimizer_D.step()

            # -----------------
            #  Train Generator
            # -----------------
            optimizer_G.zero_grad()

            # Generate denoised images
            gen_output = generator(noisy_imgs)
            gen_imgs = gen_output[1]

            # Adversarial loss
            pred_fake = discriminator(gen_imgs)
            loss_GAN = discriminator_loss(pred_fake, valid)

            # Content losses
            loss_structure = structure_consistency_loss(gen_imgs, clean_imgs)
            loss_pixel = criterion_pixel(gen_imgs, clean_imgs)

            # Total generator loss
            loss_G = (
                loss_GAN
                + pixel_weight * loss_pixel
                + structure_weight * loss_structure
            )

            loss_G.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=grad_clip)
            optimizer_G.step()

            # Calculate metrics
            with torch.no_grad():
                batch_ssim = calculate_ssim(gen_imgs, clean_imgs)
                batch_psnr = calculate_psnr(gen_imgs, clean_imgs)

            # Update epoch metrics
            epoch_g_loss += loss_G.item()
            epoch_d_loss += loss_D.item()
            epoch_pixel_loss += loss_pixel.item()
            epoch_ssim += batch_ssim.item()
            epoch_psnr += batch_psnr.item()
            n_batches += 1

            # Update progress bar
            train_bar.set_postfix(
                {
                    "G_Loss": f"{loss_G.item():.4f}",
                    "D_Loss": f"{loss_D.item():.4f}",
                    "SSIM": f"{batch_ssim.item():.4f}",
                    "PSNR": f"{batch_psnr.item():.4f}",
                }
            )

            # Early stopping for extreme loss values
            if not torch.isfinite(loss_G) or not torch.isfinite(loss_D):
                print("Training stopped due to infinite loss")
                return generator, discriminator

        # Calculate epoch averages
        epoch_g_loss /= n_batches
        epoch_d_loss /= n_batches
        epoch_pixel_loss /= n_batches
        epoch_ssim /= n_batches
        epoch_psnr /= n_batches

        # Validation phase
        generator.eval()
        val_ssim = 0
        val_psnr = 0
        val_count = 0

        with torch.no_grad():
            for data in test_loader:
                noisy_imgs = data["lr"].to(device)
                clean_imgs = data["hr"].to(device)

                gen_imgs = generator(noisy_imgs)[1]

                val_ssim += calculate_ssim(gen_imgs, clean_imgs).item()
                val_psnr += calculate_psnr(gen_imgs, clean_imgs).item()
                val_count += 1

        val_ssim /= val_count
        val_psnr /= val_count

        # Log metrics
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    epoch,
                    epoch_g_loss,
                    epoch_d_loss,
                    epoch_pixel_loss,
                    epoch_ssim,
                    val_psnr,
                    val_ssim,
                    val_psnr,
                ]
            )

        # Save best model (must happen before early_stopping updates best_loss)
        metric_value = val_ssim if metric == "ssim" else -val_psnr
        prev_best = early_stopping.best_loss
        is_best = prev_best is None or (
            (metric == "ssim" and metric_value > prev_best)
            or (metric == "mse" and metric_value < prev_best)
        )
        if is_best:
            torch.save(
                {
                    "generator_state_dict": generator.state_dict(),
                    "discriminator_state_dict": discriminator.state_dict(),
                    "optimizer_G_state_dict": optimizer_G.state_dict(),
                    "optimizer_D_state_dict": optimizer_D.state_dict(),
                    "epoch": epoch,
                    "ssim_score": val_ssim,
                    "psnr_score": val_psnr,
                },
                save_path,
            )
            print(f"Saved best model with SSIM: {val_ssim:.4f}, PSNR: {val_psnr:.4f}")

        # Early stopping check
        if early_stopping(metric_value):
            print(f"Early stopping triggered at epoch {epoch}")
            break

        print(
            f"Epoch {epoch}: G_loss: {epoch_g_loss:.4f}, D_loss: {epoch_d_loss:.4f}, "
            f"SSIM: {val_ssim:.4f}, PSNR: {val_psnr:.4f}"
        )

        if on_epoch_end is not None:
            on_epoch_end(
                epoch,
                generator,
                {
                    "g_loss": epoch_g_loss,
                    "d_loss": epoch_d_loss,
                    "pixel_loss": epoch_pixel_loss,
                    "train_ssim": epoch_ssim,
                    "train_psnr": epoch_psnr,
                    "val_ssim": val_ssim,
                    "val_psnr": val_psnr,
                },
                device,
            )

    return generator, discriminator
