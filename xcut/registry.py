"""Model registry for xcut.

Maps model type strings (from config YAML) to builder functions.
Each builder takes a ModelConfig and device, returns the objects needed
for training (generator, discriminator if GAN, etc.).

Adding a new model:
    1. Write a builder function that returns a dict with at least 'generator'.
    2. Register it: MODEL_REGISTRY["my_model"] = build_my_model
"""

from __future__ import annotations

from typing import Any, Callable

import torch

from xcut.config import ModelConfig

BuilderFn = Callable[[ModelConfig, torch.device], dict[str, Any]]

MODEL_REGISTRY: dict[str, BuilderFn] = {}


def register_model(name: str) -> Callable[[BuilderFn], BuilderFn]:
    """Decorator to register a model builder."""

    def decorator(fn: BuilderFn) -> BuilderFn:
        MODEL_REGISTRY[name] = fn
        return fn

    return decorator


@register_model("hinet_gan")
def build_hinet_gan(config: ModelConfig, device: torch.device) -> dict[str, Any]:
    """Build HINet generator + discriminator for GAN training."""
    from xcut.models.hinet import HINet
    from xcut.models.hinet_gan import Discriminator

    gen_params = {
        "in_chn": config.params.get("in_chn", 1),
        "wf": config.params.get("wf", 64),
        "depth": config.params.get("depth", 5),
        "relu_slope": config.params.get("relu_slope", 0.2),
    }
    disc_params = {
        "in_channels": config.params.get("in_chn", 1),
    }

    generator = HINet(**gen_params).to(device)
    discriminator = Discriminator(**disc_params).to(device)

    return {"generator": generator, "discriminator": discriminator}


@register_model("hinet")
def build_hinet(config: ModelConfig, device: torch.device) -> dict[str, Any]:
    """Build HINet generator for supervised (non-GAN) training."""
    from xcut.models.hinet import HINet

    gen_params = {
        "in_chn": config.params.get("in_chn", 1),
        "wf": config.params.get("wf", 64),
        "depth": config.params.get("depth", 5),
        "relu_slope": config.params.get("relu_slope", 0.2),
    }
    generator = HINet(**gen_params).to(device)
    return {"generator": generator}


@register_model("rectified_flow")
def build_rectified_flow(config: ModelConfig, device: torch.device) -> dict[str, Any]:
    """Build HINetI2SB backbone + RectifiedFlowDiffusion wrapper."""
    from xcut.models.hinet_i2sb import create_hinet_i2sb
    from xcut.models.rectified_flow import create_rectified_flow

    params = config.params
    backbone = create_hinet_i2sb(
        in_chn=params.get("in_chn", 1),
        wf=params.get("wf", 64),
        depth=params.get("depth", 5),
        time_dim=params.get("time_dim", 256),
        n_timestep=params.get("n_timestep", 1000),
        cond_channels=params.get("cond_channels", 1),
    ).to(device)

    diffusion = create_rectified_flow(
        model=backbone,
        image_size=params.get("image_size", 64),
        n_timestep=params.get("n_timestep", 1000),
        nfe=params.get("nfe", 4),
        objective=params.get("objective", "velocity"),
        loss_type=params.get("loss_type", "l1"),
        sigma_min=params.get("sigma_min", 0.0),
        clip_denoise=params.get("clip_denoise", True),
        device=device,
    ).to(device)

    return {"generator": diffusion, "backbone": backbone}


def build_model(config: ModelConfig, device: torch.device) -> dict[str, Any]:
    """Build model(s) from config using the registry."""
    if config.type not in MODEL_REGISTRY:
        available = ", ".join(sorted(MODEL_REGISTRY.keys()))
        raise KeyError(f"Unknown model type '{config.type}'. Available: {available}")
    return MODEL_REGISTRY[config.type](config, device)
