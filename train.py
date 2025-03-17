import torch
import torch.nn as nn
import lightning
import yaml

from data.era5 import ERA5DataModule
from models.autoencoder import Autoencoder


def main():

    with open("configs/autoencoder_kl_f8.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    data = ERA5DataModule(config)

    autoencoder = Autoencoder(config)

    trainer = lightning.Trainer(
        max_epochs=config["config"]["training"]["max_epochs"],
        precision="16-mixed",
    )

    trainer.fit(autoencoder, datamodule=data)


if __name__ == '__main__':
    main()
