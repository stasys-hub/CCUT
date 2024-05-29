import torch
from nn.rrdbunet import UNetRRDB2
from utils.dataloader import DatasetConfig, CC_Dataset
from nn.trainer import Trainer
from nn.hooks import EarlyStopping
from nn.losses import CombinedLoss
from utils.transforms import norm_ccmat
import logging

# Profiling
# import cProfile
# import pstats


def main():
    # Set the logging level
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    # setup params
    batch_size = 1

    # setup model
    unet = UNetRRDB2(in_channels=1, out_channels=1, features=[64, 128, 256, 512, 1024])

    # Setup Data
    df = DatasetConfig("../data/multi4M.json")
    ds = CC_Dataset(df, transform_x=norm_ccmat, transform_y=norm_ccmat)
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
        grad_scaler=torch.cuda.amp.GradScaler(),
        hooks=hook_classes,
        hook_args=hook_args,
        optim_class=torch.optim.Adam,
        optim_params={"lr": 5e-5, "betas": (0.0, 0.9), "weight_decay": 1e-4},
        mixed_precision=True,
    )

    trainer.train(train_loader, 10)
    optim = trainer.get_optimizer()

    unet.save_model("/home/keetz/DEV/CCUT/checkpoints/unet-rrdb-combinedloss4M-16x.pth", optimizer=optim)


if __name__ == "__main__":
    main()
