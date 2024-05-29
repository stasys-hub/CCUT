from abc import ABC
import matplotlib.pyplot as plt
import torch
from utils.metrics import compare_images
import numpy as np
import torch.nn.functional as F


# Abstract class for loss functions
class LossFunction(torch.nn.Module, ABC):  # Inherits from both nn.Module and ABC
    def __init__(self):
        super(LossFunction, self).__init__()

    def forward(self, outputs, targets):
        """
        Compute the loss based on the model outputs and the targets.

        Parameters:
            outputs: The output from the model
            targets: The ground truth targets

        Returns:
            Loss value computed from the outputs and targets
        """
        pass


class SSIMLoss(LossFunction):
    """
    Implements the Structural Similarity Index (SSIM) as a loss function.

    SSIM measures the structural similarity between two images. The SSIM score lies
    between -1 and 1, with 1 indicating two identical images. By subtracting the SSIM
    score from 1, it is converted into a loss, where a value of 0 indicates perfect similarity.

    Parameters:
    - C1 (float): a constant to stabilize the division with weak denominator.
    - C2 (float): a constant to stabilize the division with weak denominator.
    - window_size (int): size of the window used to compute SSIM.

    Returns:
    - loss (torch.Tensor): SSIM-based loss between the two input images.
    """

    def __init__(self, C1=0.01**2, C2=0.03**2, window_size: int = 11):
        super(SSIMLoss, self).__init__()
        self.C1 = C1
        self.C2 = C2
        self.window_size = window_size

    def forward(self, img1, img2):
        # Convert numpy arrays to tensors
        if isinstance(img1, np.ndarray):
            img1 = torch.from_numpy(img1).float()
        if isinstance(img2, np.ndarray):
            img2 = torch.from_numpy(img2).float()

        # Compute means of images using averaging window
        mu1 = F.avg_pool2d(img1, self.window_size)
        mu2 = F.avg_pool2d(img2, self.window_size)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        # Compute image variances
        sigma1_sq = F.avg_pool2d(img1 * img1, self.window_size) - mu1_sq
        sigma2_sq = F.avg_pool2d(img2 * img2, self.window_size) - mu2_sq
        sigma12 = F.avg_pool2d(img1 * img2, self.window_size) - mu1_mu2

        # Calculate SSIM score's numerator and denominator
        SSIM_num = (2 * mu1_mu2 + self.C1) * (2 * sigma12 + self.C2)
        SSIM_den = (mu1_sq + mu2_sq + self.C1) * (sigma1_sq + sigma2_sq + self.C2)
        SSIM = SSIM_num / SSIM_den

        # Trun into loss, by substracting
        return 1 - SSIM.mean()


class CombinedLoss(LossFunction):
    """
    Implements a combined loss function using both MSE and SSIM.

    Parameters:
    - alpha (float): Weight for the MSE loss.
    - beta (float): Weight for the SSIM loss.
    - C1, C2, window_size: Parameters for the SSIM loss.

    Returns:
    - loss (torch.Tensor): Combined loss between the two input images, where both terms are scaled to [0, 1].
    """

    def __init__(
        self, alpha=1.0, beta=1.0, C1=0.01**2, C2=0.03**2, window_size: int = 11
    ):
        super(CombinedLoss, self).__init__()
        self.mse_loss = torch.nn.MSELoss(reduction="mean")
        self.ssim_loss = SSIMLoss(C1, C2, window_size)
        self.alpha = alpha
        self.beta = beta

    def forward(self, outputs, targets):
        # Get the scaled MSE loss
        mse_val = self.mse_loss(
            outputs, targets
        )  # assuming images are in [0,1] this value should resemble the ,ax value of yoput images
        ssim_val = self.ssim_loss(outputs, targets)

        combined_loss = self.alpha * mse_val + self.beta * ssim_val

        return combined_loss
