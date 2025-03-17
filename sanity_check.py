import yaml

import torch
from tqdm import tqdm
import lightning as pl

from models.autoencoder import Autoencoder
from data.dummy_data import DummyDataModule


def main():

    # print device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load a yaml file
    with open("configs/autoencoder_kl_f8.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    autoencoder = Autoencoder(config)

    # Load the data module
    data_module = DummyDataModule(config, 100000)

    # Test Train

    trainer = pl.Trainer()

    trainer.fit(autoencoder, data_module)


if __name__ == '__main__':
    main()
