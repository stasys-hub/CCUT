"""HINet with timestep conditioning for I2SB diffusion.

Extends the base HINet architecture with:
- Sinusoidal timestep embeddings
- Time conditioning via adaptive instance normalization (AdaIN) or simple injection

The network takes (x_t, t) and predicts the noise/direction to recover x_0.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from xcut.models.hinet import (
    HINet,
    SAM,
    UNetConvBlock,
    UNetUpBlock,
    conv3x3,
    conv,
    conv_down,
)


class SinusoidalPosEmb(nn.Module):
    """Sinusoidal positional embedding for timesteps.

    As used in DDPM and most diffusion models.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Embed timestep.

        Args:
            t: Timestep tensor, shape (B,)

        Returns:
            Embedding, shape (B, dim)
        """
        device = t.device
        half_dim = self.dim // 2
        emb_scale = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb_scale)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


class TimeEmbedding(nn.Module):
    """Time embedding projection with activation."""

    def __init__(self, time_dim: int, embed_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(time_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(t)


class TimeCondConv(nn.Module):
    """Convolution with time conditioning via scale/shift (FiLM-like)."""

    def __init__(self, in_ch: int, out_ch: int, time_dim: int, kernel_size: int = 3):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, padding=kernel_size // 2)
        self.time_mlp = nn.Linear(time_dim, out_ch * 2)
        nn.init.zeros_(self.time_mlp.weight)
        nn.init.zeros_(self.time_mlp.bias)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        scale_shift = self.time_mlp(t_emb)
        scale, shift = scale_shift.chunk(2, dim=-1)
        scale = scale.unsqueeze(-1).unsqueeze(-1)
        shift = shift.unsqueeze(-1).unsqueeze(-1)
        return x * (1 + scale) + shift


class UNetConvBlockTime(nn.Module):
    """U-Net convolution block with time conditioning and optional HIN/CSFF."""

    def __init__(
        self,
        in_size: int,
        out_size: int,
        downsample: bool,
        relu_slope: float,
        time_dim: int,
        use_csff: bool = False,
        use_HIN: bool = False,
    ):
        super().__init__()
        self.downsample = downsample
        self.identity = nn.Conv2d(in_size, out_size, 1, 1, 0)
        self.use_csff = use_csff

        self.conv_1 = TimeCondConv(in_size, out_size, time_dim, kernel_size=3)
        self.relu_1 = nn.LeakyReLU(relu_slope, inplace=False)
        self.conv_2 = TimeCondConv(out_size, out_size, time_dim, kernel_size=3)
        self.relu_2 = nn.LeakyReLU(relu_slope, inplace=False)

        if downsample and use_csff:
            self.csff_enc = nn.Conv2d(out_size, out_size, 3, 1, 1)
            self.csff_dec = nn.Conv2d(out_size, out_size, 3, 1, 1)

        if use_HIN:
            self.norm = nn.InstanceNorm2d(out_size // 2, affine=True)
        self.use_HIN = use_HIN

        if downsample:
            self.downsample_conv = conv_down(out_size, out_size, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        t_emb: torch.Tensor,
        enc: torch.Tensor | None = None,
        dec: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        out = self.conv_1(x, t_emb)

        if self.use_HIN:
            out_1, out_2 = torch.chunk(out, 2, dim=1)
            out = torch.cat([self.norm(out_1), out_2], dim=1)
        out = self.relu_1(out)
        out = self.relu_2(self.conv_2(out, t_emb))

        out = out + self.identity(x)
        if enc is not None and dec is not None:
            out = out + self.csff_enc(enc) + self.csff_dec(dec)
        if self.downsample:
            out_down = self.downsample_conv(out)
            return out_down, out
        return out


class UNetUpBlockTime(nn.Module):
    """U-Net upsampling block with time conditioning."""

    def __init__(self, in_size: int, out_size: int, relu_slope: float, time_dim: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(
            in_size, out_size, kernel_size=2, stride=2, bias=True
        )
        self.conv_block = UNetConvBlockTime(
            in_size, out_size, False, relu_slope, time_dim
        )

    def forward(
        self, x: torch.Tensor, bridge: torch.Tensor, t_emb: torch.Tensor
    ) -> torch.Tensor:
        up = self.up(x)
        out = torch.cat([up, bridge], 1)
        out = self.conv_block(out, t_emb)
        return out


class HINetI2SB(nn.Module):
    """HINet with timestep conditioning for I2SB diffusion.

    Two-stage architecture with:
    - Sinusoidal time embeddings
    - Time conditioning at each convolution block
    - Cross-stage feature fusion
    - Optional LR conditioning via channel concatenation

    Input: (x_t, t, cond=x_1) where x_t is noisy sample, t is timestep, x_1 is LR condition
    Output: Prediction for recovering x_0 from x_t
    """

    def __init__(
        self,
        in_chn: int = 1,
        wf: int = 64,
        depth: int = 5,
        relu_slope: float = 0.2,
        hin_position_left: int = 0,
        hin_position_right: int = 4,
        time_dim: int = 256,
        n_timestep: int = 1000,
        cond_channels: int = 1,
    ):
        super().__init__()
        self.depth = depth
        self.time_dim = time_dim
        self.in_chn = in_chn
        self.cond_channels = cond_channels
        self.use_cond = cond_channels > 0

        total_in_chn = in_chn + cond_channels if self.use_cond else in_chn

        # Time embedding
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim // 4),
            TimeEmbedding(time_dim // 4, time_dim),
        )

        self.down_path_1 = nn.ModuleList()
        self.down_path_2 = nn.ModuleList()
        self.conv_01 = nn.Conv2d(total_in_chn, wf, 3, 1, 1)
        self.conv_02 = nn.Conv2d(total_in_chn, wf, 3, 1, 1)

        # Time injection at input level
        self.time_to_input = nn.Linear(time_dim, wf)

        prev_channels = wf
        for i in range(depth):
            use_HIN = hin_position_left <= i <= hin_position_right
            downsample = (i + 1) < depth
            self.down_path_1.append(
                UNetConvBlockTime(
                    prev_channels,
                    (2**i) * wf,
                    downsample,
                    relu_slope,
                    time_dim,
                    use_HIN=use_HIN,
                )
            )
            self.down_path_2.append(
                UNetConvBlockTime(
                    prev_channels,
                    (2**i) * wf,
                    downsample,
                    relu_slope,
                    time_dim,
                    use_csff=downsample,
                    use_HIN=use_HIN,
                )
            )
            prev_channels = (2**i) * wf

        self.up_path_1 = nn.ModuleList()
        self.up_path_2 = nn.ModuleList()
        self.skip_conv_1 = nn.ModuleList()
        self.skip_conv_2 = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            self.up_path_1.append(
                UNetUpBlockTime(prev_channels, (2**i) * wf, relu_slope, time_dim)
            )
            self.up_path_2.append(
                UNetUpBlockTime(prev_channels, (2**i) * wf, relu_slope, time_dim)
            )
            self.skip_conv_1.append(nn.Conv2d((2**i) * wf, (2**i) * wf, 3, 1, 1))
            self.skip_conv_2.append(nn.Conv2d((2**i) * wf, (2**i) * wf, 3, 1, 1))
            prev_channels = (2**i) * wf

        self.sam12 = SAM(prev_channels, out_chn=in_chn)
        self.cat12 = nn.Conv2d(prev_channels * 2, prev_channels, 1, 1, 0)
        self.last = conv3x3(prev_channels, in_chn, bias=True)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor | None = None,
        return_both_stages: bool = False,
    ) -> torch.Tensor | list[torch.Tensor]:
        """Forward pass with timestep and optional LR conditioning.

        Args:
            x: Input tensor (x_t), shape (B, C, H, W)
            t: Timestep, shape (B,) - integer indices from 0 to n_timestep-1
            cond: Optional conditioning tensor (LR/x_1), shape (B, cond_channels, H, W)
            return_both_stages: If True, return [stage1, stage2] like original HINet

        Returns:
            Prediction, shape (B, C, H, W)
        """
        # Concat LR conditioning if provided
        if cond is not None and self.use_cond:
            image = torch.cat([x, cond], dim=1)
        else:
            image = x

        # Store original x_t for SAM and residual (SAM expects single channel)
        x_t_original = x

        # Time embedding
        t_emb = self.time_mlp(t)
        x1 = self.conv_01(image)
        # Add time embedding to input features
        x1 = x1 + self.time_to_input(t_emb).unsqueeze(-1).unsqueeze(-1)

        encs = []
        decs = []
        for i, down in enumerate(self.down_path_1):
            if (i + 1) < self.depth:
                x1, x1_up = down(x1, t_emb)
                encs.append(x1_up)
            else:
                x1 = down(x1, t_emb)

        for i, up in enumerate(self.up_path_1):
            x1 = up(x1, self.skip_conv_1[i](encs[-i - 1]), t_emb)
            decs.append(x1)

        # SAM uses original x_t (single channel) for image output
        sam_feature, out_1 = self.sam12(x1, x_t_original)

        x2 = self.conv_02(image)
        x2 = self.cat12(torch.cat([x2, sam_feature], dim=1))
        blocks = []
        for i, down in enumerate(self.down_path_2):
            if (i + 1) < self.depth:
                x2, x2_up = down(x2, t_emb, encs[i], decs[-i - 1])
                blocks.append(x2_up)
            else:
                x2 = down(x2, t_emb)

        for i, up in enumerate(self.up_path_2):
            x2 = up(x2, self.skip_conv_2[i](blocks[-i - 1]), t_emb)

        out_2 = self.last(x2)
        # Residual connection: model learns to predict the "correction" to x_t
        # This is consistent with original HINet and works for diffusion too
        out_2 = out_2 + x_t_original

        if return_both_stages:
            return [out_1, out_2]
        return out_2


def create_hinet_i2sb(
    in_chn: int = 1,
    wf: int = 64,
    depth: int = 5,
    time_dim: int = 256,
    n_timestep: int = 1000,
    cond_channels: int = 1,
) -> HINetI2SB:
    """Factory function to create HINet for I2SB.

    Args:
        in_chn: Number of input channels
        wf: Width factor for feature channels
        depth: Number of U-Net levels
        time_dim: Dimension of time embedding
        n_timestep: Number of diffusion timesteps
        cond_channels: Number of conditioning channels (LR input). Set to 0 to disable.

    Returns:
        HINetI2SB model
    """
    return HINetI2SB(
        in_chn=in_chn,
        wf=wf,
        depth=depth,
        time_dim=time_dim,
        n_timestep=n_timestep,
        cond_channels=cond_channels,
    )
