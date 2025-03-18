import torch
import torch.nn as nn
import lightning
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateFinder
import yaml

from data.era5 import ERA5DataModule
from models.autoencoder import Autoencoder


def main():

    torch.set_float32_matmul_precision("medium")

    with open("configs/autoencoder_kl_f8.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    data = ERA5DataModule(config)

    autoencoder = Autoencoder(config)

    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints/",
        filename=config["config"]["general"]["name"] + "-{step}",
        every_n_train_steps=100,

    )

    lr_finder = LearningRateFinder(mode="linear")

    trainer = lightning.Trainer(
        max_epochs=config["config"]["training"]["epochs"],
        precision="16-mixed",
        callbacks=[checkpoint_callback, lr_finder],
        accumulate_grad_batches=4,
    )

    trainer.fit(autoencoder, datamodule=data)


if __name__ == '__main__':
    main()
