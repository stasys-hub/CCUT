import torch
import torch.nn as nn
import torch.nn.functional
import torch.utils.checkpoint as checkpoint
from timm.layers import DropPath, to_2tuple, trunc_normal_

# Data Augmentation

import torch
import torch.nn as nn


class SymmetricalLayer(nn.Module):
    """
    A custom PyTorch layer that symmetrizes a given tensor along its last two dimensions.

    The layer computes the average of the input tensor and its transpose along the last two dimensions
    (typically corresponding to height and width in an image tensor). This operation enforces a kind of
    symmetry, which might be useful in certain types of neural network architectures, especially those
    dealing with spatial data.
    """

    def __init__(self):
        super(SymmetricalLayer, self).__init__()
        # Initialization can be expanded if needed.

    def forward(self, x):
        """
        Forward pass of the SymmetricalLayer.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: A tensor which is the element-wise average of the input and its transpose
            along the last two dimensions.
        """
        # Transpose the input along the last two dimensions (height and width).
        # This assumes the input has a shape where the last two dimensions correspond to spatial dimensions.
        x_transposed = torch.transpose(x, -1, -2)

        # Take the average of the original input and the transposed input.
        # This step enforces the symmetrical property.
        output = (x + x_transposed) / 2

        return output


#################### CBAM ####################
class ChannelAttention(nn.Module):
    """
    Channel Attention Module as part of the CBAM (Convolutional Block Attention Module).
    This module computes attention weights based on the channel-wise statistics of the input feature map.

    Args:
        in_channels (int): Number of input channels.
        down_ratio (int): Reduction ratio for the intermediate dense layer. Default is 8.
    """

    def __init__(self, in_channels, down_ratio=8):
        super().__init__()

        # Adaptive pooling layers to reduce spatial dimensions to 1x1, focusing on channel-wise statistics.
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # MLP with one hidden layer to capture channel-wise dependencies.
        # The number of neurons in the hidden layer is reduced by 'down_ratio' to decrease model complexity.
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels // down_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // down_ratio, in_channels, bias=False),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Apply average pooling and max pooling, followed by the MLP.
        # This process generates two different channel-wise attention maps.
        x1 = self.avg_pool(x).squeeze(-1).squeeze(-1)
        x1 = self.mlp(x1)
        x2 = self.max_pool(x).squeeze(-1).squeeze(-1)
        x2 = self.mlp(x2)

        # Combine the two attention maps and apply sigmoid activation.
        # The result is a channel-wise attention map.
        out = self.sigmoid(x1 + x2).unsqueeze(-1).unsqueeze(-1)

        # Apply the attention map to the input feature map.
        return out * x


class SpatialAttention(nn.Module):
    """
    Spatial Attention Module as part of the CBAM.
    This module computes attention weights based on the spatial distribution of the input feature map.

    Args:
        kernel_size (int): The size of the kernel to use in the convolutional layer. Default is 7.
    """

    def __init__(self, kernel_size=7):
        super().__init__()

        # Convolution layer to generate a spatial attention map.
        # It takes in 2 channels (mean and max of the input) and outputs 1 channel.
        self.conv = nn.Conv2d(
            2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Calculate the mean and max along the channel dimension and concatenate.
        x1 = torch.mean(x, dim=1, keepdim=True)
        x2, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([x1, x2], dim=1)

        # Apply convolution and sigmoid activation to get the spatial attention map.
        out = self.conv(out)
        out = self.sigmoid(out)

        # Apply the attention map to the input feature map.
        return out * x


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module (CBAM).
    This module sequentially infers attention maps along two separate dimensions - channel and spatial,
    and then multiplies them to the input feature map for adaptive feature refinement.

    Args:
        in_channels (int): Number of input channels.
        down_ratio (int): Reduction ratio for the intermediate dense layer in channel attention. Default is 8.
        kernel_size (int): The size of the kernel in the spatial attention module. Default is 7.
    """

    def __init__(self, in_channels, down_ratio=8, kernel_size=7):
        super().__init__()

        # Initialize channel and spatial attention modules.
        self.ca = ChannelAttention(in_channels=in_channels, down_ratio=down_ratio)
        self.sa = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        # Apply channel attention followed by spatial attention.
        x = self.ca(x)
        x = self.sa(x)
        return x


##### --------------------------- RDBD CORE LAYERS --------------------------- ####


class ConvBlock2(nn.Module):
    def __init__(self, in_channels, out_channels):
        """
        A convolutional block used in the U-Net, consisting of two convolutional layers
        with a ReLU activation and batch normalization.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
        """
        super(ConvBlock2, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """
        Forward pass of the ConvBlock.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        x = self.conv1(x)
        x = self.relu(x)
        return x


# Core Layers
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_act, **kwargs):
        super().__init__()

        # Define a convolutional layer with specified input and output channels,
        # and additional keyword arguments passed in **kwargs.
        # Set bias to True for the convolutional layer.
        self.cnn = nn.Conv2d(
            in_channels,
            out_channels,
            **kwargs,
            bias=True,
        )

        # If use_act is True, initialize an instance of LeakyReLU activation function
        # with a negative slope of 0.2, otherwise use the identity function.
        self.act = nn.LeakyReLU(0.2, inplace=True) if use_act else nn.Identity()

    def forward(self, x):
        # Apply convolution followed by activation function to the input tensor x.
        return self.act(self.cnn(x))


import torch
import torch.nn as nn


class DenseResidualBlock(nn.Module):
    def __init__(self, in_channels, channels=32, residual_beta=0.2):
        """
        Initializes a DenseResidualBlock module.

        Args:
            in_channels (int): Number of input channels.
            channels (int): Number of internal channels for ConvBlock layers (default: 32).
            residual_beta (float): Scaling factor for the residual connection (default: 0.2).
        """
        super().__init__()
        self.residual_beta = residual_beta
        self.blocks = nn.ModuleList()

        # Create a sequence of ConvBlock layers and append to the module list
        for i in range(5):
            self.blocks.append(
                ConvBlock(
                    in_channels + channels * i,
                    channels if i <= 3 else in_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    use_act=True if i <= 3 else False,
                )
            )

    def forward(self, x):
        """
        Forward pass through the DenseResidualBlock.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        new_inputs = x
        for block in self.blocks:
            out = block(new_inputs)
            new_inputs = torch.cat([new_inputs, out], dim=1)
        return self.residual_beta * out + x


class RRDB(nn.Module):
    def __init__(self, in_channels, residual_beta=0.2, reps: int = 3):
        """
        Initializes a RRDB (Residual-in-Residual Dense Block) module.

        Args:
            in_channels (int): Number of input channels.
            residual_beta (float): Scaling factor for the residual connection (default: 0.2).
        """
        super().__init__()
        self.residual_beta = residual_beta
        self.rrdb = nn.Sequential(
            *[DenseResidualBlock(in_channels) for _ in range(reps)]
        )

    def forward(self, x):
        """
        Forward pass through the RRDB.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        return self.rrdb(x) * self.residual_beta + x


##### --------------------------- SWIN CORE LAYERS --------------------------- ####


# Define a Multi-layer Perceptron (MLP) module
class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        """
        Initialize the Mlp module.

        Parameters:
            in_features (int): Number of input features.
            hidden_features (int, optional): Number of hidden features. Defaults to `in_features`.
            out_features (int, optional): Number of output features. Defaults to `in_features`.
            act_layer (torch.nn.Module, optional): Activation layer. Defaults to nn.GELU.
            drop (float, optional): Dropout rate. Defaults to 0.
        """
        # Initialize the parent class, nn.Module
        super().__init__()

        # Initialize parameters if not provided
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        # Define the first fully connected layer
        self.fc1 = nn.Linear(in_features, hidden_features)

        # Define the activation function
        self.act = act_layer()

        # Define the second fully connected layer
        self.fc2 = nn.Linear(hidden_features, out_features)

        # Define the dropout layer
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        """
        Forward pass through the MLP.

        Parameters:
            x (torch.Tensor): The input tensor.

        Returns:
            x (torch.Tensor): The output tensor.
        """

        # Pass input through the first fully connected layer
        x = self.fc1(x)

        # Apply the activation function
        x = self.act(x)

        # Apply dropout
        x = self.drop(x)

        # Pass through the second fully connected layer
        x = self.fc2(x)

        # Apply dropout again
        x = self.drop(x)

        return x


def window_partition(x, window_size):
    """
    Partition a 4D tensor into non-overlapping windows.

    Args:
        x (torch.Tensor): Input tensor of shape (Batch size, Height, Width, Channels).
        window_size (int): The size of each square window.

    Returns:
        windows (torch.Tensor): Partitioned windows, reshaped into
                                 (num_windows * Batch size, window_size, window_size, Channels).
    """
    # Extract the shape of the input tensor
    B, H, W, C = x.shape

    # Reshape the input tensor to prepare for window partitioning
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)

    # Permute the dimensions to isolate windows and flatten into the output shape
    windows = (
        x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    )

    return windows


def window_reverse(windows, window_size, H, W):
    """
    Reverse the partitioning of a 4D tensor into non-overlapping windows to reconstruct the original tensor.

    Args:
        windows (torch.Tensor): Partitioned windows, with shape
                                (num_windows * Batch size, window_size, window_size, Channels).
        window_size (int): The size of each square window.
        H (int): The height of the original image.
        W (int): The width of the original image.

    Returns:
        x (torch.Tensor): Reconstructed tensor of shape (Batch size, Height, Width, Channels).
    """

    # Calculate the batch size based on the shape of the windows and the original dimensions
    B = int(windows.shape[0] / (H * W / window_size / window_size))

    # Reshape the windows into the original tensor shape but partitioned
    x = windows.view(
        B, H // window_size, W // window_size, window_size, window_size, -1
    )

    # Reverse the permute and reshape operations done during partitioning
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)

    return x


class WindowAttention(nn.Module):
    """
    Window-based Multi-Head Self-Attention (W-MSA) module with relative position bias.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): Dimensions (height and width) of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default is True.
        qk_scale (float, optional): Override the default qk scale of head_dim ** -0.5 if set.
        attn_drop (float, optional): Dropout rate for attention weights. Default is 0.0.
        proj_drop (float, optional): Dropout rate for the output. Default is 0.0.
    """

    def __init__(
        self,
        dim,
        window_size,
        num_heads,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()

        self.dim = dim  # Number of input channels
        self.window_size = window_size  # Dimensions of the window (Height x Width)
        self.num_heads = num_heads  # Number of attention heads
        head_dim = dim // num_heads  # Dimension of each attention head
        self.scale = qk_scale or head_dim**-0.5  # Scaling factor for the query and key

        # Define a parameter table for the relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads)
        )

        # Initialize relative position indices
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(
            torch.meshgrid([coords_h, coords_w])
        )  # 2, window_height, window_width
        coords_flatten = torch.flatten(
            coords, 1
        )  # Flatten to get pair-wise combinations
        # Calculate pair-wise relative position offsets
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(
            1, 2, 0
        ).contiguous()  # Reshape to get relative position indices
        relative_coords[:, :, 0] += self.window_size[0] - 1  # Zero-center coordinates
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)

        self.register_buffer("relative_position_index", relative_position_index)

        # Linear layers for Q, K, V projections
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        # Dropout layer for attention scores
        self.attn_drop = nn.Dropout(attn_drop)

        # Output projection layer
        self.proj = nn.Linear(dim, dim)

        # Output dropout layer
        self.proj_drop = nn.Dropout(proj_drop)

        # Initialize parameters
        trunc_normal_(self.relative_position_bias_table, std=0.02)

        # Softmax activation for the attention scores
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Forward pass for the WindowAttention module.

        Args:
            x (Tensor): Input features with shape (num_windows * B, N, C).
            mask (Tensor, optional): A mask tensor with shape (num_windows, Wh * Ww, Wh * Ww). Defaults to None.

        Returns:
            Tensor: The output tensor after applying window-based multi-head self-attention.
        """

        B_, N, C = x.shape  # Extract dimensions

        # Apply the QKV linear layer and reshape.
        qkv = (
            self.qkv(x)
            .reshape(B_, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )

        # Split into individual Q, K, V tensors
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Scale the query tensor
        q = q * self.scale

        # Compute attention scores (Q @ K')
        attn = q @ k.transpose(-2, -1)

        # Add relative position bias
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1],
            -1,
        )
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn += relative_position_bias.unsqueeze(0)

        # Apply mask, if provided
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(
                1
            ).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        # Apply softmax to the attention scores
        attn = self.softmax(attn)

        # Apply dropout to attention
        attn = self.attn_drop(attn)

        # Compute the output tensor
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)

        # Apply output projection and dropout
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}"

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops


class SwinTransformerBlock(nn.Module):
    r"""Swin Transformer Block.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(
        self,
        dim,
        input_resolution,
        num_heads,
        window_size=7,
        shift_size=0,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()

        # Initialize instance variables.
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        # Validate and adjust window and shift sizes.
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size

        # Initialize normalization and attention layers.
        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim,
            window_size=to_2tuple(self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        # Initialize drop path.
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        # Initialize another normalization and MLP layers.
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

        # Calculate attention mask if shift_size is greater than 0.
        if self.shift_size > 0:
            attn_mask = self.calculate_mask(self.input_resolution)
        else:
            attn_mask = None

        # Register attention mask as buffer.
        self.register_buffer("attn_mask", attn_mask)

    def calculate_mask(self, x_size):
        # calculate attention mask for SW-MSA
        H, W = x_size
        img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
        h_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        w_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(
            img_mask, self.window_size
        )  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(
            attn_mask == 0, float(0.0)
        )

        return attn_mask

    def calculate_mask(self, x_size):
        # Extract height and width from the input resolution.
        H, W = x_size

        # Initialize a zero mask with dimensions 1, H, W, 1.
        img_mask = torch.zeros((1, H, W, 1))

        # Define slices for rolling shift along height and width.
        h_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )
        w_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None),
        )

        # Counter for filling in the img_mask.
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                # Fill in the mask using slices and counter.
                img_mask[:, h, w, :] = cnt
                cnt += 1

        # Partition the image mask into windows.
        mask_windows = window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)

        # Calculate attention mask by comparing each pair of mask windows.
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(
            attn_mask == 0, float(0.0)
        )

        return attn_mask

    def forward(self, x, x_size):
        # Main forward pass for the block.
        H, W = x_size
        B, L, C = x.shape

        # Normalize input and prepare for window-based attention.
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # Apply cyclic shift if required.
        if self.shift_size > 0:
            shifted_x = torch.roll(
                x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2)
            )
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(
            shifted_x, self.window_size
        )  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(
            -1, self.window_size * self.window_size, C
        )  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA (to be compatible for testing on images whose shapes are the multiple of window size
        if self.input_resolution == x_size:
            attn_windows = self.attn(
                x_windows, mask=self.attn_mask
            )  # nW*B, window_size*window_size, C
        else:
            attn_windows = self.attn(
                x_windows, mask=self.calculate_mask(x_size).to(x.device)
            )

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(
                shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2)
            )
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class PatchMerging(nn.Module):
    # Initialize Patch Merging layer
    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution  # Resolution of input feature
        self.dim = dim  # Number of input channels
        self.reduction = nn.Linear(
            4 * dim, 2 * dim, bias=False
        )  # Linear layer for dimension reduction
        self.norm = norm_layer(4 * dim)  # Normalization layer

    def forward(self, x):
        # Input x has shape (B, H*W, C)
        H, W = self.input_resolution
        B, L, C = x.shape

        # Validate input dimensions
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        # Reshape x for merging patches
        x = x.view(B, H, W, C)

        # Split feature map into four parts
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]

        # Concatenate patches
        x = torch.cat([x0, x1, x2, x3], -1)
        x = x.view(B, -1, 4 * C)

        # Apply normalization and dimension reduction
        x = self.norm(x)
        x = self.reduction(x)

        return x  # Output tensor

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"


class BasicLayer(nn.Module):
    """A basic Swin Transformer layer for one stage.
    This class aggregates a sequence of Swin Transformer blocks into a single module.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(
        self,
        dim,
        input_resolution,
        depth,
        num_heads,
        window_size,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        downsample=None,
        use_checkpoint=False,
    ):
        super().__init__()
        self.dim = dim  # Embedding dimension
        self.input_resolution = input_resolution  # Input resolution (Height x Width)
        self.depth = depth  # Number of Swin Transformer blocks
        self.use_checkpoint = use_checkpoint  # Whether to use gradient checkpointing

        # Build Swin Transformer blocks
        self.blocks = nn.ModuleList(
            [
                SwinTransformerBlock(
                    dim=dim,
                    input_resolution=input_resolution,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=0 if (i % 2 == 0) else window_size // 2,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[i]
                    if isinstance(drop_path, list)
                    else drop_path,
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )

        # Optional downsample layer
        if downsample is not None:
            self.downsample = downsample(
                input_resolution, dim=dim, norm_layer=norm_layer
            )
        else:
            self.downsample = None

    def forward(self, x, x_size):
        """Forward pass through the layer"""
        for blk in self.blocks:
            # Optionally use gradient checkpointing for saving memory
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, x_size)
            else:
                x = blk(x, x_size)

        # Optional downsampling at the end of the layer
        if self.downsample is not None:
            x = self.downsample(x)
        return x  # Output tensor

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"


##### --------------------------- TRANSFORMER CORE LAYERS --------------------------- ####


class PatchEmbed(nn.Module):
    r"""Image to Patch Embedding
    This class is responsible for converting an input image to a sequence of flattened patches and
    performing an optional normalization.Image to Patch Embedding
    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(
        self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None
    ):
        super().__init__()
        img_size = to_2tuple(img_size)  # Ensure the image size is a tuple
        patch_size = to_2tuple(patch_size)  # Ensure the patch size is a tuple
        patches_resolution = [
            img_size[0] // patch_size[0],
            img_size[1] // patch_size[1],
        ]  # Calculate patches resolution
        self.img_size = img_size  # Original image size
        self.patch_size = patch_size  # Size of each patch
        self.patches_resolution = (
            patches_resolution  # Resolution after dividing image into patches
        )
        self.num_patches = (
            patches_resolution[0] * patches_resolution[1]
        )  # Total number of patches

        self.in_chans = in_chans  # Number of input channels (e.g., 3 for RGB images)
        self.embed_dim = embed_dim  # Dimension of the embedding space

        # Optional normalization layer
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        """Forward pass
        Args:
            x: The input image tensor of shape [B, C, H, W].
        Returns:
            x: The patch embeddings of shape [B, num_patches, embed_dim].
        """
        # Flatten each patch and transpose dimensions
        x = x.flatten(2).transpose(1, 2)  # Resulting shape: [B, num_patches, embed_dim]

        # Apply normalization if specified
        if self.norm is not None:
            x = self.norm(x)
        return x  # Output patch embeddings


class PatchUnEmbed(nn.Module):
    """Patch to Image Unembedding
    This class is responsible for converting a sequence of flattened patches back to an image.
    It performs the reverse operation of PatchEmbed.
    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(
        self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None
    ):
        super().__init__()
        img_size = to_2tuple(img_size)  # Ensure the image size is a tuple
        patch_size = to_2tuple(patch_size)  # Ensure the patch size is a tuple
        patches_resolution = [
            img_size[0] // patch_size[0],
            img_size[1] // patch_size[1],
        ]  # Calculate patches resolution
        self.img_size = img_size  # Original image size
        self.patch_size = patch_size  # Size of each patch
        self.patches_resolution = (
            patches_resolution  # Resolution after dividing image into patches
        )
        self.num_patches = (
            patches_resolution[0] * patches_resolution[1]
        )  # Total number of patches

        self.in_chans = in_chans  # Number of input channels (e.g., 3 for RGB images)
        self.embed_dim = embed_dim  # Dimension of the embedding space

    def forward(self, x, x_size):
        """Forward pass
        Args:
            x: The input patch embeddings of shape [B, num_patches, embed_dim].
            x_size: A tuple indicating the spatial dimensions (Height, Width) to reshape the image to.
        Returns:
            x: The reconstructed image of shape [B, C, H, W].
        """
        B, HW, C = x.shape  # Batch size, Height x Width, Channels

        # Reshape and transpose dimensions back to original image
        x = x.transpose(1, 2).view(B, self.embed_dim, x_size[0], x_size[1])

        return x  # Output reconstructed image


class RSTB(nn.Module):
    """Residual Swin Transformer Block (RSTB).
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        img_size: Input image size.
        patch_size: Patch size.
        resi_connection: The convolutional block before residual connection.
    """

    def __init__(
        self,
        dim,
        input_resolution,
        depth,
        num_heads,
        window_size,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        norm_layer=nn.LayerNorm,
        downsample=None,
        use_checkpoint=False,
        img_size=224,
        patch_size=4,
        resi_connection="1conv",
    ):
        super(RSTB, self).__init__()

        self.dim = dim
        self.input_resolution = input_resolution

        # Initialize a group of Swin Transformer blocks
        self.residual_group = BasicLayer(
            dim=dim,
            input_resolution=input_resolution,
            depth=depth,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop,
            attn_drop=attn_drop,
            drop_path=drop_path,
            norm_layer=norm_layer,
            downsample=downsample,
            use_checkpoint=use_checkpoint,
        )

        if resi_connection == "1conv":
            self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        elif resi_connection == "3conv":
            # to save parameters and memory
            self.conv = nn.Sequential(
                nn.Conv2d(dim, dim // 4, 3, 1, 1),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim // 4, 1, 1, 0),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim, 3, 1, 1),
            )
        # Initialize patch embedding layer
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=0,
            embed_dim=dim,
            norm_layer=None,
        )
        # Initialize patch unembedding layer
        self.patch_unembed = PatchUnEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=0,
            embed_dim=dim,
            norm_layer=None,
        )

    def forward(self, x, x_size):
        # Apply Swin Transformer blocks to the input
        x_transformed = self.residual_group(x, x_size)

        # Convert the transformed patches back to image form
        x_unembed = self.patch_unembed(x_transformed, x_size)

        # Apply convolutional layer(s) to the image
        x_conv = self.conv(x_unembed)

        # Convert the image back into patch form
        x_embed = self.patch_embed(x_conv)

        # Add original input to the output (Residual Connection)
        out = x_embed + x

        return out

    # Original, but not really comprehensible
    # def forward(self, x, x_size):
    #     return self.patch_embed(self.conv(self.patch_unembed(self.residual_group(x, x_size), x_size))) + x


class Upsample(nn.Sequential):
    """Upsample module.
    Args:
        scale (int): Scale factor. Supported scales are powers of 2 and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat):
        m = []  # List to store layers for upscaling

        # Check if the scale factor is a power of 2
        if (scale & (scale - 1)) == 0:
            # Create multiple upscaling blocks if scale = 2^n (n>1)
            for _ in range(int(math.log(scale, 2))):
                # Conv2D layer to increase the number of channels
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))

                # PixelShuffle layer to upscale the feature map by a factor of 2
                m.append(nn.PixelShuffle(2))

        # Check if the scale factor is 3
        elif scale == 3:
            # Conv2D layer to increase the number of channels
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))

            # PixelShuffle layer to upscale the feature map by a factor of 3
            m.append(nn.PixelShuffle(3))

        # If the scale factor is neither a power of 2 nor 3, raise an error
        else:
            raise ValueError(
                f"scale {scale} is not supported. Supported scales: 2^n and 3."
            )

        # Initialize the nn.Sequential with the created layers
        super(Upsample, self).__init__(*m)


class UpsampleOneStep(nn.Sequential):
    """UpsampleOneStep module (the difference with Upsample is that it always only has 1conv + 1pixelshuffle)
       Used in lightweight SR to save parameters.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
        num_out_ch (int): Channel number of output features.
        input_resolution (tuple, optional): The resolution of the input feature map (H, W).
    """

    def __init__(self, scale, num_feat, num_out_ch, input_resolution=None):
        # Number of channels in the input feature map
        self.num_feat = num_feat

        # Input resolution of the feature map
        self.input_resolution = input_resolution

        m = []  # List to store layers for upscaling

        # Conv2D layer to increase the number of channels
        m.append(nn.Conv2d(num_feat, (scale**2) * num_out_ch, 3, 1, 1))

        # PixelShuffle layer to upscale the feature map by the specified scale factor
        m.append(nn.PixelShuffle(scale))

        # Initialize the nn.Sequential with the created layers
        super(UpsampleOneStep, self).__init__(*m)


# ----- Addtional Layers ------- #


# Adapted from: https://github.com/JavierGurrola/RDUNet
class DenoisingBlock(nn.Module):
    """
    A DenoisingBlock for reducing noise in image-like data.
    ...
    """

    def __init__(self, in_channels, inner_channels, out_channels):
        super(DenoisingBlock, self).__init__()

        # Define a single activation function to be reused
        self.actv = nn.ReLU(inplace=True)

        # Define convolutional layers
        self.conv_0 = nn.Conv2d(in_channels, inner_channels, 3, padding=1)
        self.conv_1 = nn.Conv2d(
            in_channels + inner_channels, inner_channels, 3, padding=1
        )
        self.conv_2 = nn.Conv2d(
            in_channels + 2 * inner_channels, inner_channels, 3, padding=1
        )
        self.conv_3 = nn.Conv2d(
            in_channels + 3 * inner_channels, out_channels, 3, padding=1
        )

    def forward(self, x):
        # Apply convolutional layers and shared activation function
        out_0 = self.actv(self.conv_0(x))
        out_0 = torch.cat([x, out_0], 1)

        out_1 = self.actv(self.conv_1(out_0))
        out_1 = torch.cat([out_0, out_1], 1)

        out_2 = self.actv(self.conv_2(out_1))
        out_2 = torch.cat([out_1, out_2], 1)

        out_3 = self.actv(self.conv_3(out_2))

        return out_3 + x


class CombineTensorsLayer(nn.Module):
    def __init__(self):
        # Initialize the parent class, nn.Module
        super(CombineTensorsLayer, self).__init__()

    # Define the forward pass
    def forward(self, tensor1, tensor2):
        """
        Combines two tensors by concatenating them along the channel dimension.

        Parameters:
            tensor1: The first tensor.
            tensor2: The second tensor.

        Returns:
            result_tensor: The concatenated tensor.
        """

        # Embed the elements of tensor2 along its diagonal and add a channel dimension
        tensor2 = torch.diag_embed(tensor2)
        tensor2 = tensor2.unsqueeze(1)

        # Concatenate tensor1 and tensor2 along the channel dimension (dim=1)
        result_tensor = torch.cat((tensor1, tensor2), dim=1)

        # Return the concatenated tensor
        return result_tensor
