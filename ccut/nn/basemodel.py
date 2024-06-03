import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from abc import ABC, abstractmethod
from utils.transforms import get_padded_dims, pad_matrix


class BaseModel(nn.Module, ABC):
    def __init__(self):
        super(BaseModel, self).__init__()

        # Can be triggered to stop training
        self.stop_training = False

    @abstractmethod
    def forward(self, x):
        """
        Forward pass of the model. Must be implemented by all subclasses.
        """
        pass

    def plot_sample(self, data):
        """
        Example method for plotting a sample. Can be overridden by subclasses.
        """
        pass

    def save_model(self, path, optimizer=None, epoch=None, additional_info=None):
        """
        Save the model state along with additional training information.
        """
        save_dict = {"model_state_dict": self.state_dict()}
        if optimizer:
            save_dict["optimizer_state_dict"] = optimizer.state_dict()
        if epoch is not None:
            save_dict["epoch"] = epoch
        if additional_info:
            save_dict.update(additional_info)

        torch.save(save_dict, path)

    def load_model(self, path, optimizer=None, map_location="cuda:0"):
        """
        Load the model state. Optionally load optimizer state.

        Parameters:
            path (str): The file path to load the model from.
            optimizer (torch.optim.Optimizer, optional): The optimizer to load the state into.
            map_location (str or torch.device, optional): The device to map the model to. Defaults to 'cuda'.
        """
        checkpoint = torch.load(path, map_location=map_location)
        self.load_state_dict(checkpoint["model_state_dict"])
        if optimizer and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        return checkpoint

    def load(self, model_dict_path):
        self.load_state_dict(torch.load(model_dict_path))

    def predict_patch(self, patch: np.ndarray, device=None) -> np.ndarray:
        """
        Predicts the output for a single patch using the model.

        Parameters:
        patch (np.ndarray): The input patch.
        device (torch.device): The device to perform computations on.

        Returns:
        np.ndarray: The predicted patch.
        """
        # Infer the device from the model
        if not device:
            device = next(self.parameters()).device
        patch_tensor = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0).to(device)
        pred = self(patch_tensor)
        if device == "cpu":
            pred_numpy = pred.detach().squeeze(0).squeeze(0).numpy()
        else:
            pred_numpy = pred.detach().cpu().squeeze(0).squeeze(0).numpy()
        return pred_numpy

    def reconstruct_matrix(
        self, lr: np.ndarray, patch_size: int, device=None
    ) -> np.ndarray:
        """
        Reconstructs an image from patches using the given model.

        Parameters:
        lr (np.ndarray): The low-resolution image.
        patch_size (int): The size of each patch.
        device (torch.device): The device to perform computations on.

        Returns:
        np.ndarray: The reconstructed image.
        """
        padding = get_padded_dims(dimensions=lr.shape, patch_size=patch_size)
        padded_lr = pad_matrix(lr, padding=padding)

        # Infer the device from the model
        if not device:
            device = next(self.parameters()).device

        reconstructed_matrix = np.zeros_like(padded_lr)
        padded_height, padded_width = padded_lr.shape
        for i in tqdm(range(0, padded_height, patch_size)):
            for j in range(0, padded_width, patch_size):
                patch = padded_lr[i : i + patch_size, j : j + patch_size]
                pred_numpy = self.predict_patch(patch, device)
                reconstructed_matrix[i : i + patch_size, j : j + patch_size] = (
                    pred_numpy
                )

        # Remove padding to match the original dimensions
        height, width = lr.shape
        final_reconstructed_matrix = np.clip(
            reconstructed_matrix[:height, :width], 0, 1
        )
        final_reconstructed_matrix[final_reconstructed_matrix < 0.001] = 0
        return final_reconstructed_matrix
