import torch
from nn.rrdbunet import UNetRRDB2
from utils.dataloader import DatasetConfig, CC_Dataset
from nn.trainer import Trainer
from nn.hooks import EarlyStopping
from nn.losses import CombinedLoss
from utils.transforms import clip_and_norm
import logging
from functools import partial


def main():
    # Set the logging level: INFO, WARNING,...
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s-%(levelname)s: %(message)s",
        datefmt="(%d/%m/%Y)-%H:%M",  # Custom time format
    )
    # setup params
    batch_size = 1

    # setup model
    unet = UNetRRDB2(in_channels=1, out_channels=1, features=[64, 128, 256, 512, 1024])

    # Use partial to provide parameters to functions passed to CC_Dataset
    transform_clip_and_norm = partial(clip_and_norm, clip_value=300)

    # Setup Data
    df = DatasetConfig("../data/datasets.json")  # only to map data
    ds = CC_Dataset(
        df, transform_x=[transform_clip_and_norm], transform_y=[transform_clip_and_norm]
    )  # you can supply it with a list of functions which will be applied per sample
    train_loader = torch.utils.data.DataLoader(
        ds, batch_size=batch_size, num_workers=0, shuffle=True
    )

    # Initialize your hooks
    hook_classes = [EarlyStopping]

    # Define hook arguments
    hook_args = {}

    # Set pytorch loss function
    loss = CombinedLoss(window_size=15)

    trainer = Trainer(
        model=unet,
        loss_function=loss,
        # set a grad scaler (optional)
        grad_scaler=torch.cuda.amp.GradScaler(),
        # Hooks and arguments for them can be provided -> custom implementations are possible based on nn.hooks
        hooks=hook_classes,
        hook_args=hook_args,
        # optimizer as SGD, RMSE or ADAM and paramters
        optim_class=torch.optim.Adam,
        optim_params={"lr": 5e-5, "betas": (0.0, 0.9), "weight_decay": 1e-4},
        # Use mixed precision?
        mixed_precision=True,
    )

    trainer.train(train_loader, 10)
    optim = trainer.get_optimizer()

    unet.save_model(
        "/home/keetz/DEV/CCUT/checkpoints/unet-rrdb-combinedloss4M-16x.pth",
        optimizer=optim,
    )


if __name__ == "__main__":
    main()
