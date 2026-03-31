import torch
import torch.nn as nn


def conv3x3(in_chn: int, out_chn: int, bias: bool = True) -> nn.Conv2d:
    """Create a 3x3 convolution with padding."""
    return nn.Conv2d(in_chn, out_chn, kernel_size=3, stride=1, padding=1, bias=bias)


def conv_down(in_chn: int, out_chn: int, bias: bool = False) -> nn.Conv2d:
    """Create a downsampling convolution (4x4, stride 2)."""
    return nn.Conv2d(in_chn, out_chn, kernel_size=4, stride=2, padding=1, bias=bias)


def conv(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    bias: bool = False,
    stride: int = 1,
) -> nn.Conv2d:
    """Create a convolution with padding."""
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size,
        padding=(kernel_size // 2),
        bias=bias,
        stride=stride,
    )


class SAM(nn.Module):
    """Supervised Attention Module."""

    def __init__(
        self, n_feat: int, kernel_size: int = 3, bias: bool = True, out_chn: int = 3
    ):
        """Initialize SAM.

        Args:
            n_feat: Number of input features.
            kernel_size: Convolution kernel size.
            bias: Whether to use bias in convolutions.
            out_chn: Number of output channels.
        """
        super().__init__()
        self.conv1 = conv(n_feat, n_feat, kernel_size, bias=bias)
        self.conv2 = conv(n_feat, out_chn, kernel_size, bias=bias)
        self.conv3 = conv(out_chn, n_feat, kernel_size, bias=bias)

    def forward(
        self, x: torch.Tensor, x_img: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass of SAM.

        Args:
            x: Feature map.
            x_img: Input image.

        Returns:
            Tuple of (attention-enhanced features, output image).
        """
        x1 = self.conv1(x)
        img = self.conv2(x) + x_img
        x2 = torch.sigmoid(self.conv3(img))
        x1 = x1 * x2
        x1 = x1 + x
        return x1, img


class UNetConvBlock(nn.Module):
    """U-Net convolution block with optional HIN and CSFF."""

    def __init__(
        self,
        in_size: int,
        out_size: int,
        downsample: bool,
        relu_slope: float,
        use_csff: bool = False,
        use_HIN: bool = False,
    ):
        """Initialize UNetConvBlock.

        Args:
            in_size: Input channel size.
            out_size: Output channel size.
            downsample: Whether to downsample.
            relu_slope: Slope for LeakyReLU.
            use_csff: Whether to use cross-stage feature fusion.
            use_HIN: Whether to use Half Instance Normalization.
        """
        super().__init__()
        self.downsample = downsample
        self.identity = nn.Conv2d(in_size, out_size, 1, 1, 0)
        self.use_csff = use_csff

        self.conv_1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_1 = nn.LeakyReLU(relu_slope, inplace=False)
        self.conv_2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1, bias=True)
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
        enc: torch.Tensor | None = None,
        dec: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: Input tensor.
            enc: Encoder features for CSFF (optional).
            dec: Decoder features for CSFF (optional).

        Returns:
            Output tensor or (downsampled_output, output) if downsample.
        """
        out = self.conv_1(x)

        if self.use_HIN:
            out_1, out_2 = torch.chunk(out, 2, dim=1)
            out = torch.cat([self.norm(out_1), out_2], dim=1)
        out = self.relu_1(out)
        out = self.relu_2(self.conv_2(out))

        out += self.identity(x)
        if enc is not None and dec is not None:
            out = out + self.csff_enc(enc) + self.csff_dec(dec)
        if self.downsample:
            out_down = self.downsample_conv(out)
            return out_down, out
        return out


class UNetUpBlock(nn.Module):
    """U-Net upsampling block."""

    def __init__(self, in_size: int, out_size: int, relu_slope: float):
        """Initialize UNetUpBlock.

        Args:
            in_size: Input channel size.
            out_size: Output channel size.
            relu_slope: Slope for LeakyReLU.
        """
        super().__init__()
        self.up = nn.ConvTranspose2d(
            in_size, out_size, kernel_size=2, stride=2, bias=True
        )
        self.conv_block = UNetConvBlock(in_size, out_size, False, relu_slope)

    def forward(self, x: torch.Tensor, bridge: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor.
            bridge: Skip connection from encoder.

        Returns:
            Output tensor.
        """
        up = self.up(x)
        out = torch.cat([up, bridge], 1)
        out = self.conv_block(out)
        return out


class HINet(nn.Module):
    """Half Instance Normalization Network for image restoration.

    Two-stage architecture with cross-stage feature fusion.
    """

    def __init__(
        self,
        in_chn: int = 1,
        wf: int = 64,
        depth: int = 5,
        relu_slope: float = 0.2,
        hin_position_left: int = 0,
        hin_position_right: int = 4,
    ):
        """Initialize HINet.

        Args:
            in_chn: Number of input channels.
            wf: Width factor for feature channels.
            depth: Number of U-Net levels.
            relu_slope: Slope for LeakyReLU.
            hin_position_left: First level to apply HIN.
            hin_position_right: Last level to apply HIN.
        """
        super().__init__()
        self.depth = depth
        self.down_path_1 = nn.ModuleList()
        self.down_path_2 = nn.ModuleList()
        self.conv_01 = nn.Conv2d(in_chn, wf, 3, 1, 1)
        self.conv_02 = nn.Conv2d(in_chn, wf, 3, 1, 1)

        prev_channels = wf
        for i in range(depth):
            use_HIN = hin_position_left <= i <= hin_position_right
            downsample = (i + 1) < depth
            self.down_path_1.append(
                UNetConvBlock(
                    prev_channels, (2**i) * wf, downsample, relu_slope, use_HIN=use_HIN
                )
            )
            self.down_path_2.append(
                UNetConvBlock(
                    prev_channels,
                    (2**i) * wf,
                    downsample,
                    relu_slope,
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
            self.up_path_1.append(UNetUpBlock(prev_channels, (2**i) * wf, relu_slope))
            self.up_path_2.append(UNetUpBlock(prev_channels, (2**i) * wf, relu_slope))
            self.skip_conv_1.append(nn.Conv2d((2**i) * wf, (2**i) * wf, 3, 1, 1))
            self.skip_conv_2.append(nn.Conv2d((2**i) * wf, (2**i) * wf, 3, 1, 1))
            prev_channels = (2**i) * wf

        self.sam12 = SAM(prev_channels, out_chn=in_chn)
        self.cat12 = nn.Conv2d(prev_channels * 2, prev_channels, 1, 1, 0)
        self.last = conv3x3(prev_channels, in_chn, bias=True)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Forward pass.

        Args:
            x: Input tensor.

        Returns:
            List of [stage1_output, stage2_output].
        """
        image = x
        x1 = self.conv_01(image)
        encs = []
        decs = []
        for i, down in enumerate(self.down_path_1):
            if (i + 1) < self.depth:
                x1, x1_up = down(x1)
                encs.append(x1_up)
            else:
                x1 = down(x1)

        for i, up in enumerate(self.up_path_1):
            x1 = up(x1, self.skip_conv_1[i](encs[-i - 1]))
            decs.append(x1)

        sam_feature, out_1 = self.sam12(x1, image)

        x2 = self.conv_02(image)
        x2 = self.cat12(torch.cat([x2, sam_feature], dim=1))
        blocks = []
        for i, down in enumerate(self.down_path_2):
            if (i + 1) < self.depth:
                x2, x2_up = down(x2, encs[i], decs[-i - 1])
                blocks.append(x2_up)
            else:
                x2 = down(x2)

        for i, up in enumerate(self.up_path_2):
            x2 = up(x2, self.skip_conv_2[i](blocks[-i - 1]))

        out_2 = self.last(x2)
        out_2 = out_2 + image
        return [out_1, out_2]
