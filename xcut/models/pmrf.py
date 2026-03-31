"""Posterior-Mean Rectified Flow (PMRF) for Hi-C contact matrix restoration.

PMRF improves on standard rectified flow by changing the source distribution.
Instead of flowing from LR (far from HR) to HR, the flow transports from
the posterior mean prediction (close to HR but blurry) to HR.

This achieves provably lower MSE at equivalent perceptual quality (Ohayon
et al., ICLR 2025). The key insight: optimal transport from posterior means
to ground truth is strictly better than posterior sampling.

Two-stage pipeline:
    Stage 1: A frozen pre-trained model produces posterior mean f_omega(LR).
    Stage 2: A rectified flow transports f_omega(LR) + sigma_s * noise → HR.

This module implements Stage 2 as a thin subclass of RectifiedFlowDiffusion.
The only change is the source distribution — all flow math is inherited.

References:
    - "Posterior-Mean Rectified Flow" (Ohayon et al., ICLR 2025)
    - "Flow Matching for Generative Modeling" (Lipman et al., ICLR 2023)
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import reduce

from xcut.models.rectified_flow import RectifiedFlowDiffusion


class PMRFDiffusion(RectifiedFlowDiffusion):
    """PMRF: rectified flow with posterior-mean source distribution.

    Identical to ``RectifiedFlowDiffusion`` except:
    - ``forward()`` and ``sample()`` accept an extra ``posterior_mean`` arg
    - The source distribution at t=1 is ``posterior_mean + sigma_s * noise``
      instead of the raw LR condition
    - The LR condition is still passed to the backbone for guidance

    Attributes:
        sigma_s: Noise scale added to posterior mean. Controls the
            distortion-perception tradeoff. Larger values push toward
            perceptual quality; smaller toward MSE fidelity.
    """

    def __init__(
        self,
        model: nn.Module,
        image_size: int = 128,
        n_timestep: int = 1000,
        nfe: int = 4,
        objective: Literal["velocity", "x0"] = "velocity",
        loss_type: Literal["l1", "l2"] = "l1",
        sigma_min: float = 0.0,
        sigma_s: float = 0.1,
        clip_denoise: bool = True,
        device: torch.device | str = "cuda",
    ) -> None:
        super().__init__(
            model=model,
            image_size=image_size,
            n_timestep=n_timestep,
            nfe=nfe,
            objective=objective,
            loss_type=loss_type,
            sigma_min=sigma_min,
            clip_denoise=clip_denoise,
            device=device,
        )
        self.sigma_s = sigma_s

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def forward(
        self,
        img: torch.Tensor,
        condition: torch.Tensor,
        posterior_mean: torch.Tensor,
    ) -> torch.Tensor:
        """Compute training loss with PMRF source distribution.

        Args:
            img: Target HR images ``[B, C, H, W]`` in ``[0, 1]``.
            condition: Source LR images ``[B, C, H, W]`` in ``[0, 1]``.
                Passed to the backbone as conditioning input.
            posterior_mean: Stage 1 predictions ``[B, C, H, W]`` in ``[0, 1]``.
                Used as the source distribution (replaces LR at t=1).

        Returns:
            Scalar loss.
        """
        b, c, h, w = img.shape
        assert h == self.image_size and w == self.image_size

        x0 = img  # HR — t = 0
        # PMRF source: posterior mean + small noise
        x1 = posterior_mean + self.sigma_s * torch.randn_like(posterior_mean)
        x1 = x1.clamp(0.0, 1.0)

        # Sample continuous t ∈ (0, 1)
        t_cont = torch.rand(b, device=img.device)
        t_cont = t_cont.clamp(1e-5, 1.0 - 1e-5)
        t_spatial = t_cont.view(b, 1, 1, 1)

        # Interpolate along straight-line path
        xt = self._interpolate(t_spatial, x0, x1)

        # Convert to integer timestep for backbone
        step = self._continuous_to_discrete(t_cont)

        # Forward through backbone (conditioned on LR, not posterior mean)
        pred = self.model(xt, step, condition)

        # Compute loss depending on objective
        if self.objective == "velocity":
            target = self._compute_velocity(x0, x1)
        elif self.objective == "x0":
            target = x0
        else:
            raise ValueError(f"Unknown objective: {self.objective!r}")

        if self.loss_type == "l1":
            loss = F.l1_loss(pred, target, reduction="none")
        elif self.loss_type == "l2":
            loss = F.mse_loss(pred, target, reduction="none")
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type!r}")

        loss = reduce(loss, "b ... -> b (...)", "mean")
        return loss.mean()

    def forward_with_pred(
        self,
        img: torch.Tensor,
        condition: torch.Tensor,
        posterior_mean: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute training loss and return x0 prediction.

        Same as ``forward()`` but also returns the predicted x0 for
        auxiliary losses (e.g., marginal distribution matching).

        Args:
            img: Target HR images ``[B, C, H, W]`` in ``[0, 1]``.
            condition: Source LR images ``[B, C, H, W]`` in ``[0, 1]``.
            posterior_mean: Stage 1 predictions ``[B, C, H, W]`` in ``[0, 1]``.

        Returns:
            Tuple of (loss, pred_x0).
        """
        b, c, h, w = img.shape
        assert h == self.image_size and w == self.image_size

        x0 = img
        x1 = posterior_mean + self.sigma_s * torch.randn_like(posterior_mean)
        x1 = x1.clamp(0.0, 1.0)

        t_cont = torch.rand(b, device=img.device)
        t_cont = t_cont.clamp(1e-5, 1.0 - 1e-5)
        t_spatial = t_cont.view(b, 1, 1, 1)

        xt = self._interpolate(t_spatial, x0, x1)
        step = self._continuous_to_discrete(t_cont)

        pred = self.model(xt, step, condition)

        if self.objective == "velocity":
            target = self._compute_velocity(x0, x1)
            pred_x0 = xt - t_spatial * pred
        elif self.objective == "x0":
            target = x0
            pred_x0 = pred
        else:
            raise ValueError(f"Unknown objective: {self.objective!r}")

        if self.loss_type == "l1":
            loss = F.l1_loss(pred, target, reduction="none")
        elif self.loss_type == "l2":
            loss = F.mse_loss(pred, target, reduction="none")
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type!r}")

        loss = reduce(loss, "b ... -> b (...)", "mean")
        return loss.mean(), pred_x0

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        condition: torch.Tensor,
        posterior_mean: torch.Tensor,
        return_all_timesteps: bool = False,
    ) -> torch.Tensor:
        """Generate HR predictions via Euler ODE integration from posterior mean.

        Integrates from ``t = 1`` (posterior mean + noise) to ``t = 0`` (HR).

        Args:
            batch_size: Number of samples (must match ``condition.shape[0]``).
            condition: LR images ``[B, C, H, W]`` in ``[0, 1]``.
            posterior_mean: Stage 1 predictions ``[B, C, H, W]`` in ``[0, 1]``.
            return_all_timesteps: If ``True``, return all intermediate states.

        Returns:
            Predicted HR images ``[B, C, H, W]`` in ``[0, 1]``.
        """
        nfe = self.nfe
        dt = 1.0 / nfe

        # Start from posterior mean + noise (PMRF source distribution)
        xt = posterior_mean + self.sigma_s * torch.randn_like(posterior_mean)
        xt = xt.clamp(0.0, 1.0)
        intermediates: list[torch.Tensor] = []

        for i in range(nfe):
            t_cont = 1.0 - i * dt
            t_tensor = torch.full(
                (batch_size,), t_cont, device=xt.device, dtype=xt.dtype
            )
            step = self._continuous_to_discrete(t_tensor)

            velocity = self._pred_velocity_from_model(xt, t_cont, step, condition)
            xt = xt - dt * velocity

            if self.clip_denoise:
                xt = xt.clamp(0.0, 1.0)

            if return_all_timesteps:
                intermediates.append(xt.clone())

        ret = xt.clamp(0.0, 1.0)
        if return_all_timesteps:
            ret = torch.stack(intermediates, dim=1)
        return ret

    @torch.no_grad()
    def sample_midpoint(
        self,
        batch_size: int,
        condition: torch.Tensor,
        posterior_mean: torch.Tensor,
    ) -> torch.Tensor:
        """Sample using the midpoint method (2nd-order) from posterior mean.

        Args:
            batch_size: Number of samples.
            condition: LR images ``[B, C, H, W]`` in ``[0, 1]``.
            posterior_mean: Stage 1 predictions ``[B, C, H, W]`` in ``[0, 1]``.

        Returns:
            Predicted HR images ``[B, C, H, W]`` in ``[0, 1]``.
        """
        nfe = self.nfe
        dt = 1.0 / nfe

        xt = posterior_mean + self.sigma_s * torch.randn_like(posterior_mean)
        xt = xt.clamp(0.0, 1.0)

        for i in range(nfe):
            t_cont = 1.0 - i * dt
            t_mid = t_cont - 0.5 * dt

            t_tensor = torch.full(
                (batch_size,), t_cont, device=xt.device, dtype=xt.dtype
            )
            step = self._continuous_to_discrete(t_tensor)
            v1 = self._pred_velocity_from_model(xt, t_cont, step, condition)
            x_mid = xt - 0.5 * dt * v1

            t_mid_tensor = torch.full(
                (batch_size,), t_mid, device=xt.device, dtype=xt.dtype
            )
            step_mid = self._continuous_to_discrete(t_mid_tensor)
            v_mid = self._pred_velocity_from_model(x_mid, t_mid, step_mid, condition)

            xt = xt - dt * v_mid

            if self.clip_denoise:
                xt = xt.clamp(0.0, 1.0)

        return xt.clamp(0.0, 1.0)


def create_pmrf(
    model: nn.Module,
    image_size: int = 128,
    n_timestep: int = 1000,
    nfe: int = 4,
    objective: Literal["velocity", "x0"] = "velocity",
    loss_type: Literal["l1", "l2"] = "l1",
    sigma_min: float = 0.0,
    sigma_s: float = 0.1,
    clip_denoise: bool = True,
    device: torch.device | str = "cuda",
) -> PMRFDiffusion:
    """Factory function to create a PMRF diffusion wrapper.

    Args:
        model: Backbone network. Must accept ``(x_t, timestep, condition)``.
        image_size: Spatial resolution of input patches.
        n_timestep: Discrete timestep range the backbone was designed for.
        nfe: Number of Euler steps at inference (4-10 is usually enough).
        objective: ``"velocity"`` or ``"x0"`` prediction target.
        loss_type: ``"l1"`` or ``"l2"`` pixel loss.
        sigma_min: Training noise for regularisation on the interpolation path.
        sigma_s: Noise added to posterior mean source distribution.
        clip_denoise: Clamp intermediate predictions to ``[0, 1]``.
        device: Target device.

    Returns:
        Configured ``PMRFDiffusion`` module.
    """
    return PMRFDiffusion(
        model=model,
        image_size=image_size,
        n_timestep=n_timestep,
        nfe=nfe,
        objective=objective,
        loss_type=loss_type,
        sigma_min=sigma_min,
        sigma_s=sigma_s,
        clip_denoise=clip_denoise,
        device=device,
    )
