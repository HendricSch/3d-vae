import torch
import torch.nn as nn
import lightning
from lightning.pytorch.callbacks import ModelCheckpoint
import yaml

from data.era5 import ERA5DataModule
from models.autoencoder import Autoencoder


def main():

    torch.set_float32_matmul_precision("medium")

    with open("configs/autoencoder_kl_f16_attention.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    data = ERA5DataModule(config)

    autoencoder = Autoencoder(config)

    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints/",
        filename=config["config"]["general"]["name"] + "-{step}",
        every_n_train_steps=500,

    )

    trainer = lightning.Trainer(
        max_epochs=config["config"]["training"]["epochs"],
        precision="bf16-true",
        callbacks=[checkpoint_callback],
        accelerator="tpu",
        devices="auto",
    )

    trainer.fit(autoencoder, datamodule=data)


if __name__ == '__main__':
    main()
