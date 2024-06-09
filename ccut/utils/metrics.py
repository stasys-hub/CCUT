import math

import numpy as np
import torch
import torch.nn.functional as F

from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr, spearmanr

# TODO Adapt for Gry Scale images
# define a function for peak signal-to-noise ratio (PSNR)


def ssim(img1, img2, window_size=11, size_average=True):
    """
    Compute the Structural Similarity Index (SSIM) between two images.

    Args:
    - img1 (np.ndarray): The first image array.
    - img2 (np.ndarray): The second image array.
    - window_size (int): The size of the window to use, defaults to 11.
    - size_average (bool): If True, returns the mean SSIM over the image, otherwise returns the full SSIM map.

    Returns:
    - torch.Tensor: The computed SSIM value if size_average is True, otherwise the SSIM map.
    """
    # Constants for SSIM calculation
    C1 = 0.01**2
    C2 = 0.03**2

    # Convert numpy arrays to tensors
    if isinstance(img1, np.ndarray):
        img1 = torch.from_numpy(img1).float()
    if isinstance(img2, np.ndarray):
        img2 = torch.from_numpy(img2).float()

    # Ensure the images are 2D and expand dimensions for compatibility with pooling functions
    if len(img1.shape) == 2:
        img1 = img1.unsqueeze(0).unsqueeze(0)
        img2 = img2.unsqueeze(0).unsqueeze(0)

    # Compute mean intensity for each window for both images
    mu1 = F.avg_pool2d(img1, window_size)
    mu2 = F.avg_pool2d(img2, window_size)

    # Compute the square of the mean intensity for each window
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    # Compute variance and covariance for the window
    sigma1_sq = F.avg_pool2d(img1 * img1, window_size) - mu1_sq
    sigma2_sq = F.avg_pool2d(img2 * img2, window_size) - mu2_sq
    sigma12 = F.avg_pool2d(img1 * img2, window_size) - mu1_mu2

    # Calculate the numerator and denominator of the SSIM formula
    SSIM_num = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    SSIM_den = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    SSIM = SSIM_num / SSIM_den

    if size_average:
        return SSIM.mean()
    else:
        return SSIM


def psnr(target, ref):
    """
    Calculates the Peak Signal-to-Noise Ratio (PSNR) between two RGB images.

    The PSNR is used as a quality measurement between the original and a 
    compressed image. The higher the PSNR, the better the quality of the 
    compressed or reconstructed image.

    Parameters:
    ----------
    target : np.ndarray
        The target image (e.g., the compressed image) as a NumPy array.
    
    ref : np.ndarray
        The reference image (e.g., the original image) as a NumPy array.

    Returns:
    -------
    float
        The PSNR value between the target and reference images.

    Notes:
    -----
    - Both input images should have the same dimensions and should be in RGB format.
    - The pixel values should be in the range [0, 255].
    """
    # Assume RGB image
    target_data = target.astype(float)
    ref_data = ref.astype(float)

    diff = ref_data - target_data
    diff = diff.flatten()

    rmse = math.sqrt(np.mean(diff**2.0))

    return 20 * math.log10(ref.max() / rmse)



def mse(target, ref):
    """
    Calculates the Mean Squared Error (MSE) between two images.

    The MSE is the average of the squared differences between the corresponding 
    pixels of the target and reference images.

    Parameters:
    ----------
    target : np.ndarray
        The target image as a NumPy array.
    
    ref : np.ndarray
        The reference image as a NumPy array.

    Returns:
    -------
    float
        The Mean Squared Error between the target and reference images.

    Notes:
    -----
    - Both input images should have the same dimensions.
    - The pixel values should be in the same range for meaningful results.
    """
    err = np.sum((target.astype("float") - ref.astype("float")) ** 2)
    err /= float(target.shape[0] * target.shape[1])

    return err

def mae(target, ref):
    """
    Calculates the Mean Absolute Error (MAE) between two images.

    The MAE is the average of the absolute differences between the corresponding 
    pixels of the target and reference images.

    Parameters:
    ----------
    target : np.ndarray
        The target image as a NumPy array.
    
    ref : np.ndarray
        The reference image as a NumPy array.

    Returns:
    -------
    float
        The Mean Absolute Error between the target and reference images.

    Notes:
    -----
    - Both input images should have the same dimensions.
    - The pixel values should be in the same range for meaningful results.
    """
    err = np.mean(np.abs(target.astype("float") - ref.astype("float")))

    return err


def compare_images(target, ref):
    """
    Compute and return several image quality metrics between two images.

    This function calculates the Peak Signal-to-Noise Ratio (PSNR), Mean Squared Error (MSE),
    Mean Absolute Error (MAE), and Structural Similarity Index (SSIM) between a target image
    and a reference image.

    Parameters:
    ----------
    target : np.ndarray
        The target image (e.g., the compressed or altered image) as a NumPy array.
    
    ref : np.ndarray
        The reference image (e.g., the original image) as a NumPy array.

    Returns:
    -------
    list
        A list containing the computed values of PSNR, MSE, MAE, and SSIM in that order.

    Notes:
    -----
    - Both input images should have the same dimensions and should be in a comparable format (e.g., both RGB).
    - The pixel values should be in the same range for meaningful results.
    """
    scores = []
    scores.append(psnr(target, ref))
    scores.append(mse(target, ref))
    scores.append(mae(target, ref))
    scores.append(ssim(target, ref))

    return scores



def GDLoss(generated, ground_truth, lambda_gdl=0.05):
    # Compute gradients
    gradient_x_generated, gradient_y_generated = compute_gradients(generated)
    gradient_x_ground_truth, gradient_y_ground_truth = compute_gradients(ground_truth)

    # Compute squared differences of gradients
    loss_x = F.mse_loss(gradient_x_generated, gradient_x_ground_truth)
    loss_y = F.mse_loss(gradient_y_generated, gradient_y_ground_truth)

    # Return sum of losses
    return lambda_gdl * (loss_x + loss_y)


# Function to compute gradients
def compute_gradients(image):
    sobel_x = torch.tensor(
        [[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=image.dtype, device=image.device
    ).view(1, 1, 3, 3)
    sobel_y = torch.tensor(
        [[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=image.dtype, device=image.device
    ).view(1, 1, 3, 3)

    with torch.cuda.amp.autocast(enabled=False):  # Disable autocast for computation
        gradient_x = F.conv2d(image, sobel_x, padding=1)
        gradient_y = F.conv2d(image, sobel_y, padding=1)

    return gradient_x, gradient_y


def multi_scale_loss(generated, ground_truth, scales=[0.5, 1, 2.0]):
    loss = 0
    for scale in scales:
        generated_scaled = F.interpolate(
            generated, scale_factor=scale, mode="bicubic", align_corners=False
        )
        ground_truth_scaled = F.interpolate(
            ground_truth, scale_factor=scale, mode="bicubic", align_corners=False
        )
        loss += F.mse_loss(generated_scaled, ground_truth_scaled)
    return loss / len(scales)



def calculate_insulation_score(matrix, window_size):
    """
    Calculate the insulation score for a given contact matrix.

    The insulation score measures the degree of insulation for each bin in a contact matrix,
    which is a common metric used in Hi-C data analysis.

    Parameters:
    ----------
    matrix : np.ndarray
        A 2D NumPy array representing the contact matrix.
    
    window_size : int
        The size of the window used to calculate the insulation score.

    Returns:
    -------
    np.ndarray
        A 1D NumPy array containing the insulation scores for each bin.

    Notes:
    -----
    - The input contact matrix should be square (n_bins x n_bins).
    - The window size should be chosen appropriately based on the resolution of the matrix.
    - Bins at the edges of the matrix may have insulation scores calculated with smaller windows,
      as they do not have sufficient neighbors.
    """
    # Determine the number of bins
    n_bins = matrix.shape[0]

    # Initialize the insulation score array
    insulation_scores = np.zeros(n_bins)

    # Loop over each bin to calculate the insulation score
    for i in range(n_bins):
        # Define the inner window limits
        inner_start = max(i - window_size, 0)
        inner_end = min(i + window_size + 1, n_bins)

        # Define the outer window limits
        outer_start = max(i - 2 * window_size, 0)
        outer_end = min(i + 2 * window_size + 1, n_bins)

        # Calculate the sum of interactions within the inner window
        inner_sum = np.sum(matrix[inner_start:inner_end, inner_start:inner_end])

        # Calculate the sum of interactions within the outer window, excluding the inner window
        outer_sum = (
            np.sum(matrix[outer_start:outer_end, outer_start:outer_end]) - inner_sum
        )

        # Calculate the insulation score
        if outer_sum != 0:
            insulation_scores[i] = inner_sum / outer_sum
        else:
            insulation_scores[i] = 0  # Handle division by zero

    return insulation_scores

def compare_signals(signal1, signal2):
    """
    Compute various metrics between two signal arrays.

    This function calculates the Mean Squared Error (MSE), Mean Absolute Error (MAE),
    Root Mean Squared Error (RMSE), Pearson correlation coefficient, and Spearman 
    correlation coefficient between two input signals.

    Parameters:
    ----------
    signal1 : np.ndarray
        The first signal array.
    
    signal2 : np.ndarray
        The second signal array.

    Returns:
    -------
    dict
        A dictionary containing the computed metrics:
        - "mse": Mean Squared Error
        - "mae": Mean Absolute Error
        - "rmse": Root Mean Squared Error
        - "pearson_correlation": Pearson correlation coefficient
        - "spearman_correlation": Spearman correlation coefficient

    Notes:
    -----
    - Both input signals should be 1D arrays of the same length.
    """
    mse = mean_squared_error(signal1, signal2)
    mae = mean_absolute_error(signal1, signal2)
    rmse = np.sqrt(mse)
    pearson_corr, _ = pearsonr(signal1, signal2)
    spearman_corr, _ = spearmanr(signal1, signal2)

    return {
        "mse": mse,
        "mae": mae,
        "rmse": rmse,
        "pearson_correlation": pearson_corr,
        "spearman_correlation": spearman_corr,
    }
