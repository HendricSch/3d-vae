import torch
import torch.nn as nn
import lightning
from lightning.pytorch.callbacks import LearningRateFinder
import yaml

from data.era5 import ERA5DataModule
from models.autoencoder import Autoencoder


def main():

    torch.set_float32_matmul_precision("medium")

    # set seed
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    with open("configs/autoencoder/kl-f8.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    data = ERA5DataModule(config)

    autoencoder = Autoencoder(config)

    lr_finder = LearningRateFinder(
        min_lr=1e-7,
        max_lr=1e-2,
        num_training=200,
        mode="exponential",
        early_stop_threshold=None,
        update_attr=True
    )

    trainer = lightning.Trainer(
        max_epochs=config["config"]["training"]["epochs"],
        precision="16-mixed",
        callbacks=[lr_finder],
    )

    trainer.fit(autoencoder, datamodule=data)


if __name__ == '__main__':
    main()
