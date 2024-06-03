from nn.esrgan import Generator
from nn.rrdbunet import UNetRRDB2
from nn.swinir import SwinIR
import torch


# Test the forward pass of ConvBlock
def test_forward_pass_gen():
    batch_size = 16
    input_channels = 1
    input_height = 32
    input_width = 32
    esr = Generator(input_channels)
    unet = UNetRRDB2(in_channels=1, out_channels=1, features=[64, 128, 256, 512, 1024])
    swin = SwinIR(
        upscale=1,
        img_size=(input_height, input_width),
        window_size=8,
        img_range=1.0,
        depths=[6, 6, 6, 6],
        embed_dim=60,
        num_heads=[6, 6, 6, 6],
        mlp_ratio=2,
        in_chans=1,
    )

    input_tensor = torch.randn(batch_size, input_channels, input_height, input_width)

    output = esr(input_tensor)
    output = unet(input_tensor)
    output = swin(input_tensor)

    # Ensure the output tensor has the correct shape
    assert output.shape == (batch_size, input_channels, input_height, input_width)


def main():
    test_forward_pass_gen()


# Run the tests using pytest
if __name__ == "__main__":
    main()
