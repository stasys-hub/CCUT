"""Rectified Flow for Hi-C contact matrix restoration.

Implements conditional rectified flow (straight-line OT paths) between
LR and HR distributions. Dramatically simpler than Schrödinger bridge
or DDPM, with competitive or superior quality at very few sampling steps.

Theory:
    The forward process defines a straight-line interpolation:
        x_t = (1 - t) * x_0 + t * x_1        where x_0 = HR, x_1 = LR

    The velocity field is constant along each path:
        v = dx/dt = x_1 - x_0

    A neural network learns to predict this velocity (or equivalently, x_0)
    from (x_t, t, condition). Sampling inverts the ODE from t=1 (LR) to t=0
    (HR) using Euler steps.

    Optionally, a small amount of noise (sigma_min) is added at training time
    to prevent the flow from overfitting to exact pairs:
        x_t = (1 - t) * x_0 + t * x_1 + sigma_min * eps

Parameterizations:
    - "velocity": Network predicts v = x_1 - x_0. Training target is the
      velocity. x_0 is recovered as x_t - t * v_pred for intermediate steps.
    - "x0": Network directly predicts x_0. The velocity is derived as
      (x_t - pred_x0) / t during sampling. Matches the convention used by
      I2SBBridgeDiffusion and HiCBridgeDiffusion for easy comparison.

Sampling:
    Euler ODE integration from t=1 → t=0 in `nfe` steps:
        x_{t-dt} = x_t - dt * v(x_t, t)

References:
    - "Flow Matching for Generative Modeling" (Lipman et al., ICLR 2023)
    - "Flow Straight and Fast" (Liu et al., ICLR 2023)
    - "ResFlow" (CVPR 2025), "PMRF" (ICLR 2025)
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import reduce


class RectifiedFlowDiffusion(nn.Module):
    """Rectified flow for conditional image restoration.

    Drop-in replacement for I2SBBridgeDiffusion / HiCBridgeDiffusion.
    Same ``forward(img, condition) → loss`` and
    ``sample(batch_size, condition) → prediction`` interface.

    Attributes:
        model: Backbone network with signature ``(x_t, timestep, condition) → output``.
        image_size: Spatial resolution of input patches.
        n_timestep: Discrete timestep range for backbone compatibility.
            Continuous ``t ∈ [0, 1]`` is mapped to integer indices via
            ``step = round(t * (n_timestep - 1))``.
        nfe: Number of function evaluations (Euler steps) during sampling.
        objective: Prediction target — ``"velocity"`` or ``"x0"``.
        loss_type: Pixel loss — ``"l1"`` or ``"l2"``.
        sigma_min: Small noise injected during training for regularisation.
        clip_denoise: Whether to clamp predictions to ``[0, 1]`` during sampling.
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
        clip_denoise: bool = True,
        device: torch.device | str = "cuda",
    ) -> None:
        super().__init__()
        self.model = model
        self.image_size = image_size
        self.n_timestep = n_timestep
        self.nfe = nfe
        self.objective = objective
        self.loss_type = loss_type
        self.sigma_min = sigma_min
        self.clip_denoise = clip_denoise
        self.device = torch.device(device)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _continuous_to_discrete(self, t: torch.Tensor) -> torch.Tensor:
        """Map continuous ``t ∈ [0, 1]`` to integer timestep for backbone.

        Args:
            t: Continuous time, shape ``(B,)`` or ``(B, 1, 1, 1)``.

        Returns:
            Integer timestep indices, shape ``(B,)``, in ``[0, n_timestep - 1]``.
        """
        t_flat = t.view(-1)
        return (t_flat * (self.n_timestep - 1)).long().clamp(0, self.n_timestep - 1)

    def _compute_velocity(self, x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        """Compute ground-truth velocity field: ``v = x_1 - x_0``.

        In the rectified flow convention, the ODE goes from ``x_0`` (HR, t=0)
        to ``x_1`` (LR, t=1).  The velocity is the direction from clean to
        degraded.  Sampling *reverses* this by integrating from t=1 → t=0.

        Args:
            x0: Target HR images, shape ``(B, C, H, W)``.
            x1: Source LR images, shape ``(B, C, H, W)``.

        Returns:
            Velocity, shape ``(B, C, H, W)``.
        """
        return x1 - x0

    def _interpolate(
        self,
        t: torch.Tensor,
        x0: torch.Tensor,
        x1: torch.Tensor,
    ) -> torch.Tensor:
        """Compute ``x_t`` along the straight-line path.

        Args:
            t: Continuous time, shape ``(B, 1, 1, 1)``.
            x0: Target HR images.
            x1: Source LR images.

        Returns:
            Interpolated samples ``x_t``, shape ``(B, C, H, W)``.
        """
        xt = (1.0 - t) * x0 + t * x1
        if self.sigma_min > 0.0:
            xt = xt + self.sigma_min * torch.randn_like(xt)
        return xt

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def forward(self, img: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """Compute training loss.

        Matches the interface of ``I2SBBridgeDiffusion.forward``.

        Args:
            img: Target HR images ``[B, C, H, W]`` in ``[0, 1]``.
            condition: Source LR images ``[B, C, H, W]`` in ``[0, 1]``.

        Returns:
            Scalar loss.
        """
        b, c, h, w = img.shape
        assert h == self.image_size and w == self.image_size

        x0 = img  # HR — t = 0
        x1 = condition  # LR — t = 1

        # Sample continuous t ∈ (0, 1) — avoid exact endpoints
        t_cont = torch.rand(b, device=img.device)
        t_cont = t_cont.clamp(1e-5, 1.0 - 1e-5)
        t_spatial = t_cont.view(b, 1, 1, 1)

        # Interpolate along straight-line path
        xt = self._interpolate(t_spatial, x0, x1)

        # Convert to integer timestep for backbone
        step = self._continuous_to_discrete(t_cont)

        # Forward through backbone
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
        self, img: torch.Tensor, condition: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute training loss and return x0 prediction.

        Same as forward() but also returns the predicted x0 for
        auxiliary losses (e.g., marginal distribution matching).

        Args:
            img: Target HR images ``[B, C, H, W]`` in ``[0, 1]``.
            condition: Source LR images ``[B, C, H, W]`` in ``[0, 1]``.

        Returns:
            Tuple of (loss, pred_x0) where pred_x0 is the model's x0 prediction.
        """
        b, c, h, w = img.shape
        assert h == self.image_size and w == self.image_size

        x0 = img
        x1 = condition

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
    # Inference helpers
    # ------------------------------------------------------------------

    def _pred_x0_from_model(
        self,
        xt: torch.Tensor,
        t_cont: float,
        step: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        """Get x_0 prediction from model output, regardless of objective.

        Args:
            xt: Current state, shape ``(B, C, H, W)``.
            t_cont: Current continuous timestep (scalar).
            step: Integer timestep for backbone, shape ``(B,)``.
            condition: LR conditioning images.

        Returns:
            Predicted ``x_0``, shape ``(B, C, H, W)``.
        """
        pred = self.model(xt, step, condition)

        if self.objective == "velocity":
            # v = x_1 - x_0, and x_t = (1-t)*x_0 + t*x_1
            # → x_0 = x_t - t * v
            pred_x0 = xt - t_cont * pred
        elif self.objective == "x0":
            pred_x0 = pred
        else:
            raise ValueError(f"Unknown objective: {self.objective!r}")

        return pred_x0

    def _pred_velocity_from_model(
        self,
        xt: torch.Tensor,
        t_cont: float,
        step: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        """Get velocity prediction from model output, regardless of objective.

        Args:
            xt: Current state, shape ``(B, C, H, W)``.
            t_cont: Current continuous timestep (scalar).
            step: Integer timestep for backbone, shape ``(B,)``.
            condition: LR conditioning images.

        Returns:
            Predicted velocity, shape ``(B, C, H, W)``.
        """
        pred = self.model(xt, step, condition)

        if self.objective == "velocity":
            return pred
        elif self.objective == "x0":
            # v = (x_1 - x_0) and x_t = (1-t)*x_0 + t*x_1
            # From x_0 prediction: v = (x_t - pred_x0) / t  (when t > 0)
            # But more stable: we know v should point from pred_x0 toward x_1
            # x_t - pred_x0 = t * (x_1 - x_0), so v = (x_t - pred_x0) / t
            t_safe = max(t_cont, 1e-5)
            return (xt - pred) / t_safe
        else:
            raise ValueError(f"Unknown objective: {self.objective!r}")

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        condition: torch.Tensor,
        return_all_timesteps: bool = False,
    ) -> torch.Tensor:
        """Generate HR predictions from LR via Euler ODE integration.

        Integrates from ``t = 1`` (LR) to ``t = 0`` (HR) in ``nfe`` Euler
        steps.  Matches the interface of ``I2SBBridgeDiffusion.sample``.

        Args:
            batch_size: Number of samples (must match ``condition.shape[0]``).
            condition: LR images ``[B, C, H, W]`` in ``[0, 1]``.
            return_all_timesteps: If ``True``, return all intermediate
                predictions stacked along dim 1.

        Returns:
            Predicted HR images ``[B, C, H, W]`` in ``[0, 1]``.
            If ``return_all_timesteps``, shape is ``[B, nfe, C, H, W]``.
        """
        nfe = self.nfe
        dt = 1.0 / nfe

        # Start at t=1 → x_1 = LR
        xt = condition.clone()
        intermediates: list[torch.Tensor] = []

        for i in range(nfe):
            t_cont = 1.0 - i * dt  # current t, going from 1 → 0
            t_tensor = torch.full(
                (batch_size,), t_cont, device=xt.device, dtype=xt.dtype
            )
            step = self._continuous_to_discrete(t_tensor)

            # Get velocity and step backward (toward t=0)
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
    ) -> torch.Tensor:
        """Sample using the midpoint method (2nd-order) for better accuracy.

        Uses two function evaluations per step — one at the current point
        and one at the half-step — for roughly 2× the accuracy of Euler at
        the same ``nfe`` cost.  Total neural network calls: ``2 * nfe``.

        Args:
            batch_size: Number of samples.
            condition: LR images ``[B, C, H, W]`` in ``[0, 1]``.

        Returns:
            Predicted HR images ``[B, C, H, W]`` in ``[0, 1]``.
        """
        nfe = self.nfe
        dt = 1.0 / nfe

        xt = condition.clone()

        for i in range(nfe):
            t_cont = 1.0 - i * dt
            t_mid = t_cont - 0.5 * dt

            # Euler half-step to get midpoint
            t_tensor = torch.full(
                (batch_size,), t_cont, device=xt.device, dtype=xt.dtype
            )
            step = self._continuous_to_discrete(t_tensor)
            v1 = self._pred_velocity_from_model(xt, t_cont, step, condition)
            x_mid = xt - 0.5 * dt * v1

            # Evaluate velocity at midpoint
            t_mid_tensor = torch.full(
                (batch_size,), t_mid, device=xt.device, dtype=xt.dtype
            )
            step_mid = self._continuous_to_discrete(t_mid_tensor)
            v_mid = self._pred_velocity_from_model(x_mid, t_mid, step_mid, condition)

            # Full step using midpoint velocity
            xt = xt - dt * v_mid

            if self.clip_denoise:
                xt = xt.clamp(0.0, 1.0)

        return xt.clamp(0.0, 1.0)


def create_rectified_flow(
    model: nn.Module,
    image_size: int = 128,
    n_timestep: int = 1000,
    nfe: int = 4,
    objective: Literal["velocity", "x0"] = "velocity",
    loss_type: Literal["l1", "l2"] = "l1",
    sigma_min: float = 0.0,
    clip_denoise: bool = True,
    device: torch.device | str = "cuda",
) -> RectifiedFlowDiffusion:
    """Factory function to create a rectified flow diffusion wrapper.

    Args:
        model: Backbone network. Must accept ``(x_t, timestep, condition)``.
        image_size: Spatial resolution of input patches.
        n_timestep: Discrete timestep range the backbone was designed for.
        nfe: Number of Euler steps at inference (4–10 is usually enough).
        objective: ``"velocity"`` (predict ``x_1 - x_0``) or ``"x0"``
            (predict clean target directly).
        loss_type: ``"l1"`` or ``"l2"`` pixel loss.
        sigma_min: Optional training noise for regularisation. ``0.0``
            disables it; ``0.01`` is a reasonable starting value.
        clip_denoise: Clamp intermediate predictions to ``[0, 1]``.
        device: Target device.

    Returns:
        Configured ``RectifiedFlowDiffusion`` module.
    """
    return RectifiedFlowDiffusion(
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
