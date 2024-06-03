import numpy as np
import torch
from typing import Union


def norm_ccmat(cc_mat: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    """
    ARGS
        cc_mat: torch/numpy array 2d
    RETURN
        returns a matrix scaled between 0 - 1
    """
    if cc_mat is type(torch.Tensor):
        cc_mat = cc_mat.numpy()

    cc_mat = np.float128(cc_mat)
    if np.max(cc_mat) == 0:
        return np.zeros(cc_mat.shape)
    else:
        return (cc_mat - np.min(cc_mat)) / (np.max(cc_mat) - np.min(cc_mat))


# def log_transform(cc_mat: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
#     min_value = np.min(cc_mat)

#     positive_offset = abs(min_value) + 1e-6
#     pos_mat = cc_mat + positive_offset


#     return np.log10(pos_mat)
def log_transform(cc_mat: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    return np.nan_to_num(np.log2(cc_mat + 1, where=cc_mat != 0), nan=0)


def upsample_matrix(mat: Union[np.ndarray, torch.Tensor], factor=2) -> np.ndarray:
    """
    ARGS
        mat: numpy/torch array 2d
        factor: factor to upsample matrix dims e.g. factor 2 original (28,28) -> upsampled (56,56)
    RETURN
        np.array with upsampled matrix
    """
    return np.repeat(mat, factor, axis=1).repeat(factor, axis=0)


def upsample_matrix_4x(mat: Union[np.ndarray, torch.Tensor], factor=4) -> np.ndarray:
    """
    ARGS
        mat: numpy/torch array 2d
        factor: factor to upsample matrix dims e.g. factor 2 original (28,28) -> upsampled (56,56)
    RETURN
        np.array with upsampled matrix
    """
    return np.repeat(mat, factor, axis=1).repeat(factor, axis=0)


def upsample_matrix_5x(mat: Union[np.ndarray, torch.Tensor], factor=5) -> np.ndarray:
    """
    ARGS
        mat: numpy/torch array 2d
        factor: factor to upsample matrix dims e.g. factor 2 original (28,28) -> upsampled (56,56)
    RETURN
        np.array with upsampled matrix
    """
    return np.repeat(mat, factor, axis=1).repeat(factor, axis=0)


def get_padded_dims(dimensions: tuple[int, int], patch_size: int) -> tuple[int, int]:
    """
    Calculate the padding needed to make the image dimensions divisible by the patch size.

    Parameters:
    dimensions (tuple[int, int]): A tuple containing the height and width of the image.
    patch_size (int): The size of the patch to which dimensions should be divisible.

    Returns:
    tuple[int, int]: A tuple containing the padding for height and width.
    """
    if patch_size <= 0:
        raise ValueError("patch_size must be a positive integer")
    height, width = dimensions
    if height < 0 or width < 0:
        raise ValueError("height and width must be non-negative integers")

    # Calculate padding needed to make the image dimensions divisible by patch_size
    pad_height = (patch_size - height % patch_size) % patch_size
    pad_width = (patch_size - width % patch_size) % patch_size

    return (pad_height, pad_width)


def pad_matrix(matrix: np.ndarray, padding: tuple[int, int]) -> np.ndarray:
    """
    Pad a matrix with zeros to the specified dimensions.

    Parameters:
    matrix (np.ndarray): The input matrix to be padded.
    padding (tuple[int, int]): A tuple containing the padding to be added to the height and width of the matrix.

    Returns:
    np.ndarray: The padded matrix.
    """
    pad_height, pad_width = padding

    if pad_height < 0 or pad_width < 0:
        raise ValueError("pad_height and pad_width must be non-negative integers")
    if matrix.ndim != 2:
        raise ValueError("matrix must be a 2-dimensional array")

    return np.pad(
        matrix, ((0, pad_height), (0, pad_width)), mode="constant", constant_values=0
    )
