import torch
from nn.layers import (
    ConvBlock2,
    ConvBlock,
    DenseResidualBlock,
    RRDB,
    CombineTensorsLayer,
)
import pytest


# Sample input tensor for testing
@pytest.fixture
def sample_input():
    return torch.rand(1, 3, 64, 64)  # Batch size 1, 3 channels, 64x64 image


# Tests for ConvBlock2
def test_conv_block_2(sample_input):
    conv_block_2 = ConvBlock2(3, 16)  # In: 3 channels, Out: 16 channels
    output = conv_block_2(sample_input)
    assert output.shape == (1, 16, 64, 64)


# Tests for ConvBlock
def test_conv_block(sample_input):
    conv_block = ConvBlock(3, 16, use_act=True, kernel_size=3, stride=1, padding=1)
    output = conv_block(sample_input)
    assert output.shape == (1, 16, 64, 64)


# Tests for DenseResidualBlock
def test_dense_residual_block(sample_input):
    dense_residual_block = DenseResidualBlock(3)
    output = dense_residual_block(sample_input)
    assert output.shape == (1, 3, 64, 64)


# Tests for RRDB
def test_rrdb(sample_input):
    rrdb = RRDB(3)
    output = rrdb(sample_input)
    assert output.shape == (1, 3, 64, 64)
