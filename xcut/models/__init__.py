"""xcut.models - Neural network architectures for Hi-C enhancement."""

from xcut.models.hinet import HINet
from xcut.models.hinet_gan import Discriminator, train_gan
from xcut.models.hinet_i2sb import HINetI2SB, create_hinet_i2sb
from xcut.models.losses import (
    HiCLoss,
    distance_decay_loss,
    insulation_loss,
    structure_consistency_loss,
)
from xcut.models.pmrf import PMRFDiffusion, create_pmrf
from xcut.models.rectified_flow import RectifiedFlowDiffusion, create_rectified_flow

__all__ = [
    "HINet",
    "HINetI2SB",
    "create_hinet_i2sb",
    "Discriminator",
    "train_gan",
    # Losses (model-agnostic)
    "HiCLoss",
    "structure_consistency_loss",
    "insulation_loss",
    "distance_decay_loss",
    "RectifiedFlowDiffusion",
    "create_rectified_flow",
    "PMRFDiffusion",
    "create_pmrf",
]
