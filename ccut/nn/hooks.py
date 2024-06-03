import matplotlib.pyplot as plt
import torch
from utils.metrics import compare_images
import numpy as np


# Class for hooks to be called during training
class Hook:
    def before_epoch(self, **kwargs):
        """
        Function to be called before the start of each epoch.

        Parameters:
            **kwargs: A dictionary containing any relevant information (e.g., model, data_loader, epoch, etc.)
        """
        pass

    def after_epoch(self, **kwargs):
        """
        Function to be called after the end of each epoch.

        Parameters:
            **kwargs: A dictionary containing any relevant information (e.g., model, data_loader, epoch, metrics, etc.)
        """
        pass

    def after_step(self, **kwargs):
        """
        Function to be called after the end of each step.

        Parameters:
            **kwargs: A dictionary containing any relevant information (e.g., model, data_loader, epoch, metrics, etc.)
        """
        pass

    def before_step(self, **kwargs):
        """
        Function to be called befire each step.

        Parameters:
            **kwargs: A dictionary containing any relevant information (e.g., model, data_loader, epoch, metrics, etc.)
        """
        pass


class MonitorLoss(Hook):  # pragma: no cover
    def __init__(self, monitor="epoch_loss"):
        """
        Parameters:
            patience: Number of epochs to wait for an improvement before stopping the training.
            min_delta: Minimum change in monitored value to qualify as an improvement.
            monitor: The metric name to be monitored (e.g., 'val_loss', 'val_accuracy').
        """
        self.monitor = monitor

    def after_epoch(self, epoch, model, data_loader, metrics):
        """
        Called after each epoch ends.

        Parameters:
            epoch: Current epoch.
            model: Model being trained.
            data_loader: DataLoader used for training.
            metrics: Dictionary containing training and validation metrics.
        """
        current_loss = metrics.get(self.monitor)
        if current_loss is None:
            print(f"Warning: {self.monitor} not found in metrics.")
            return
        else:
            print(f"Epoch {epoch} - Total Loss: {current_loss}")


class PLotAfterEpoch(Hook):  # pragma: no cover
    def __init__(self, df, path) -> None:
        self.df = df
        self.path = path

    def before_epoch(self, model, **kwargs):
        self.plot_examples(model)

    def plot_examples(self, model):
        model.eval()
        for i in [360, 350, 160, 50, 36]:
            test_lr3 = self.df[i]["lr"]
            torch_lr3 = torch.from_numpy(test_lr3).to("cuda")
            test_hr3 = self.df[i]["hr"]
            y_hat3 = model(torch_lr3.unsqueeze(0))
            # plot data
            mtr = compare_images(test_hr3, y_hat3.cpu().detach().numpy())
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 5))
            ax1.set_title("Low resolution")
            ax2.set_title("High resolution")
            ax2.set_xlabel(f"SSIR: {mtr[0]:.3f}, MSE: {mtr[1]:.3f}")
            ax3.set_title("Network Prediction")
            ax3.imshow(
                np.squeeze(y_hat3.cpu().detach().numpy()),
                cmap="hot",
                interpolation="nearest",
            )
            ax2.imshow(np.squeeze(test_hr3), cmap="hot", interpolation="nearest")
            ax1.imshow(np.squeeze(test_lr3), cmap="hot", interpolation="nearest")
            plt.savefig(f"{self.path}/model_loc_{i}.png")
            plt.close()

        model.train()


class PLotAfterStep(Hook):  # pragma: no cover
    def __init__(self, df, path) -> None:
        self.df = df
        self.path = path

    def after_step(self, model, **kwargs):
        self.plot_examples(model)

    def plot_examples(self, model):
        model.eval()
        for i in [360, 350, 160, 50, 36]:
            test_lr3 = self.df[i]["lr"]
            torch_lr3 = torch.from_numpy(test_lr3).to("cuda")
            test_hr3 = self.df[i]["hr"]
            y_hat3 = model(torch_lr3.unsqueeze(0))
            # plot data
            mtr = compare_images(test_hr3, y_hat3.cpu().detach().numpy())
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 5))
            ax1.set_title("Low resolution")
            ax2.set_title("High resolution")
            ax2.set_xlabel(f"SSIR: {mtr[0]:.3f}, MSE: {mtr[1]:.3f}")
            ax3.set_title("Network Prediction")
            ax3.imshow(
                np.squeeze(y_hat3.cpu().detach().numpy()),
                cmap="hot",
                interpolation="nearest",
            )
            ax2.imshow(np.squeeze(test_hr3), cmap="hot", interpolation="nearest")
            ax1.imshow(np.squeeze(test_lr3), cmap="hot", interpolation="nearest")
            plt.savefig(f"{self.path}/model_loc_{i}.png")
            plt.close()

        model.train()


class LossTracker(Hook):  # pragma: no cover
    def __init__(self):
        self.epoch_losses = []

    def after_epoch(self, epoch, model, data_loader, metrics, **kwargs):
        current_loss = metrics.get("train_loss")
        if current_loss is not None:
            self.epoch_losses.append(current_loss)
            print(f"Epoch {epoch} - Loss added to LossTracker: {current_loss}")


class StepLossLoggingHook(Hook):  # pragma: no cover
    def after_step(self, idx, epoch, step_loss, **kwargs):
        # logger = logging.getLogger(__name__)
        # logger.info(f"Epoch {epoch}, Step {idx}, Step Loss: {step_loss:.6f}")
        print(f"Epoch {epoch}, Step {idx}, Step Loss: {step_loss:.6f}")


class EarlyStopping(Hook):  # pragma: no cover
    def __init__(self, monitor="epoch_loss", patience=3, min_delta=0.0001, mode="min"):
        """
        Args:
            monitor (str): The metric name to monitor.
            patience (int): Number of epochs with no improvement after which training will be stopped.
            min_delta (float): Minimum change in the monitored metric to qualify as an improvement.
            mode (str): One of {'min', 'max'}. If `min`, then the training will stop when the monitored metric has
                        stopped decreasing; if `max`, it will stop when the monitored metric has stopped increasing.
        """
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_score = None
        self.epochs_since_improvement = 0
        if self.mode == "min":
            self.min_delta *= -1
        elif self.mode != "max":
            raise ValueError("mode should be either 'min' or 'max'")

    def after_epoch(self, model=None, metrics=None, epoch=None, **kwargs):
        """
        Called after each epoch ends. Evaluates if the training process should be stopped early.

        Parameters:
            model (torch.nn.Module): The model being trained.
            metrics (dict): A dictionary containing metrics collected during the epoch.
            epoch (int): The current epoch number.
            **kwargs: Additional keyword arguments. This can be used to pass extra information if needed.
        """

        # Retrieve the current value of the monitored metric from the metrics dictionary
        current_score = metrics.get(self.monitor)

        # Initialize the best score if it's the first epoch or update the count of epochs since improvement
        if self.best_score is None:
            self.best_score = current_score
        elif (
            self.mode == "min" and current_score < self.best_score + self.min_delta
        ) or (self.mode == "max" and current_score > self.best_score - self.min_delta):
            # Improvement is seen; reset the counter and update the best score
            self.best_score = current_score
            self.epochs_since_improvement = 0
        else:
            # No improvement; increment the counter
            self.epochs_since_improvement += 1

        # Check if the number of epochs without improvement has reached the patience limit
        if self.epochs_since_improvement >= self.patience:
            # Log the early stopping event
            print(
                f"Early stopping triggered at epoch {epoch}. Best {self.monitor}: {self.best_score:.6f}."
            )
            # Signal the model to stop training
            model.stop_training = True


class TestHook(Hook):
    def __init__(self, test_string, some_value="") -> None:
        self.test_string = test_string
        self.some_value = some_value

    def before_epoch(self, **kwargs):
        print(f"This is a Test {self.test_string}, {self.some_value}")
