from utils.helpers import get_device
import torch
import logging


class Trainer:
    def __init__(
        self,
        model,
        loss_function,
        optim_class,
        optim_params,
        grad_scaler=None,
        mixed_precision=False,
        hooks=[],
        hook_args={},
        device="auto",
    ):
        # Initialize loss function, gradient scaler, and hooks
        self.loss_function = loss_function
        self.gs = grad_scaler
        self.hooks = hooks
        self.mixed_precision = mixed_precision
        self.hook_args = hook_args
        self.metrics = {}

        # Initialize optimizer
        self.optim = optim_class(model.parameters(), **optim_params)

        # Determine the device (CPU or GPU)
        if device is not None and device == "auto":
            self.device = get_device()
        else:
            self.device = device

        # Move the model to the device
        self.model = model.to(self.device)

        # Intitialize Hooks
        self.hooks = []
        for hook_class in hooks:
            args = hook_args.get(hook_class.__name__, {})
            hook_instance = hook_class(**args)
            self.hooks.append(hook_instance)

        # Initialize Logger
        self.logger = logging.getLogger(__name__)

        # Init metric dicts
        self.step_metrics = {}

    def step(self, lr, hr, idx):
        # Forward pass
        pred = self.model(lr)

        # Compute loss
        loss = self.loss_function(pred, hr)

        # Check for NaN values in the loss
        if not torch.isnan(loss):
            # Zero out the gradients
            self.model.zero_grad()

            # Perform backward pass and optimizer step
            if self.gs:
                # With gradient scaling to prevent underflow
                self.gs.scale(loss).backward()
                self.gs.step(self.optim)
                self.gs.update()
            else:
                # Without gradient scaling
                loss.backward()
                self.optim.step()
        else:
            self.logger.warning(
                "Training is not running as expected! You have NaN-values in your loss. Value set to 0!"
            )
            loss = 0  # set loss to 0 if NaN to ensure train_loss doesn't become NaN

        # Print the average training loss so far
        self.logger.info(f"{loss.item()/(idx+1):.6f}")

        self.step_metrics["step_loss"] = loss.item()

        return loss.item()  # Return the computed loss for this batch

    def train(self, data_loader, epochs):
        # Set the model to training mode
        self.logger.info("Starting training")
        self.model.train()

        # Execute any hooks before the epoch starts
        if self.check_hooks(state="before_epoch"):
            return

        # Loop through each epoch
        for epoch in range(epochs):
            # Initialize the training loss
            train_loss = 0.0

            # Loop through each batch in the data loader
            for idx, data in enumerate(data_loader):
                # Move the data to the device
                lr, hr = data["lr"].to(self.device), data["hr"].to(self.device)

                if self.check_hooks(state="before_step"):
                    return

                # perform training step w/wo mixed precision and accumulate loss
                if self.mixed_precision:
                    with torch.amp.autocast(self.device):
                        batch_loss = self.step(lr, hr, idx)
                else:
                    batch_loss = self.step(lr, hr, idx)

                train_loss += batch_loss  # Accumulate the loss for the epoch
                # Execute any hooks after the step ends
                if self.check_hooks(state="after_step"):
                    return

            epoch_loss = train_loss / len(data_loader)
            self.metrics = {"epoch_loss": epoch_loss, "epoch": epoch}

            self.logger.info(f"Epoch {epoch} Loss: {epoch_loss:.6f}")

            # Execute any hooks after the epoch ends
            if self.check_hooks(state="after_epoch"):
                return

    def check_hooks(self, state: str):
        """
        Executes the appropriate hooks based on the current training state.

        This method is designed to invoke different sets of hooks at various points
        in the training process (before/after each epoch and step). It handles any
        exceptions raised by hooks and logs them appropriately.

        Parameters:
        state (str): The current state of the training process. Expected values are
                    'before_epoch', 'after_epoch', 'before_step', 'after_step'.

        Returns:
        bool: True if training should be stopped (based on the model's stop_training attribute),
            False otherwise.
        """

        try:
            # Match the state and execute the corresponding hooks
            match state:
                case "before_epoch":
                    # Execute all 'before_epoch' hooks
                    for hook in self.hooks:
                        args = self.hook_args.get(hook.__class__.__name__, {})
                        hook.before_epoch(
                            model=self.model, metrics=self.metrics, **args
                        )
                case "after_epoch":
                    # Execute all 'after_epoch' hooks
                    for hook in self.hooks:
                        args = self.hook_args.get(hook.__class__.__name__, {})
                        hook.after_epoch(model=self.model, metrics=self.metrics, **args)
                case "before_step":
                    # Execute all 'before_step' hooks
                    for hook in self.hooks:
                        args = self.hook_args.get(hook.__class__.__name__, {})
                        hook.before_step(model=self.model, metrics=self.metrics, **args)
                case "after_step":
                    # Execute all 'after_step' hooks
                    for hook in self.hooks:
                        args = self.hook_args.get(hook.__class__.__name__, {})
                        hook.after_step(model=self.model, metrics=self.metrics, **args)
                case _:
                    # Log an error if an unrecognized state is provided
                    self.logger.error(
                        f"Check Hooks was provided with a wrong state argument: {state}"
                    )
        except Exception as e:
            # Log any exceptions raised by hooks
            self.logger.error(f"Error occurred in {state} hooks: {e}", exc_info=True)

        # Check if the stop_training attribute is set, indicating early stopping
        if self.model.stop_training:
            self.logger.info("Stop Training Attribute triggered!")
            return True

        return False

    def get_optimizer(self):
        """
        Returns the optimizer used in the trainer.
        """
        return self.optim
