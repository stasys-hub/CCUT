import pytest
import torch
from nn.losses import SSIMLoss, CombinedLoss


# Helper function to create test images
def create_test_images(size=(1, 1, 11, 11), value=0.0):
    return torch.full(size, value, dtype=torch.float32), torch.full(
        size, value, dtype=torch.float32
    )


@pytest.fixture
def identical_images():
    return create_test_images()


@pytest.fixture
def different_images():
    img1, _ = create_test_images()
    img2 = torch.full((1, 1, 11, 11), 1.0, dtype=torch.float32)  # Different image
    return img1, img2


def test_ssim_loss_identical_images(identical_images):
    loss_fn = SSIMLoss()
    loss = loss_fn(*identical_images)
    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-5)


def test_ssim_loss_different_images(different_images):
    loss_fn = SSIMLoss()
    loss = loss_fn(*different_images)
    assert loss > 0


def test_combined_loss_identical_images(identical_images):
    loss_fn = CombinedLoss()
    loss = loss_fn(*identical_images)
    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-5)


def test_combined_loss_different_images(different_images):
    loss_fn = CombinedLoss()
    loss = loss_fn(*different_images)
    assert loss > 0
