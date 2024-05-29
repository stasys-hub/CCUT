import torch
import torch.nn as nn
from .layers import RRDB, ConvBlock2
from .basemodel import BaseModel


class UNetRRDB2(BaseModel):
    def __init__(self, in_channels=3, out_channels=3, features=[64, 128, 256, 512]):
        """
        Initialize the U-Net architecture.

        Args:
            in_channels (int): Number of input channels. Default is 3 (for RGB images).
            out_channels (int): Number of output channels. Default is 3 (for RGB images).
            features (list of int): Number of features in each block. Default is [64, 128, 256, 512].
        """
        super(UNetRRDB2, self).__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # Add RRDB blocks separately, matching the features
        self.features = features
        self.in_channels = in_channels
        # Initialize RRDB blocks for upsampling layers
        self.rrdbs_up = nn.ModuleList(
            [RRDB(feature * 2, reps=5) for feature in reversed(features[:-1])]
        )
        self.rrdbs_up.append(
            RRDB(features[0], reps=5)
        )  # The last RRDB does not have skip connection doubling

        # Initialize downsampling layers with RRDB blocks
        for feature in features:
            self.downs.append(RRDB(in_channels))
            self.downs.append(ConvBlock2(in_channels, feature))
            in_channels = feature

        # Up part of UNet
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.ups.append(ConvBlock2(feature * 2, feature))

        self.bottleneck = ConvBlock2(features[-1], features[-1] * 2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        """
        Forward pass of the U-Net.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        skip_connections = []
        in_channels = self.in_channels

        for idx, down in enumerate(self.downs):
            if idx % 2 == 0:  # RRDB block
                x = down(x)
            else:  # ConvBlock
                x = down(x)
                skip_connections.append(x)
                x = self.pool(x)
                in_channels = self.features[idx // 2]

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        # Upsampling path
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)  # ConvTranspose2d
            x = self.rrdbs_up[idx // 2](x)  # Apply RRDB after upsampling

            skip_connection = skip_connections[idx // 2]
            if x.shape != skip_connection.shape:
                x = nn.functional.interpolate(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](concat_skip)  # ConvBlock

        return self.final_conv(x)
