import yaml

import torch
from lightning.pytorch.utilities.model_summary import ModelSummary

from models.autoencoder import Autoencoder


def main():

    # print device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load a yaml file
    with open("configs/autoencoder_kl_f8.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    autoencoder = Autoencoder(config)

    # Print the model summary
    model_summary = ModelSummary(autoencoder, max_depth=1)
    print(model_summary)


if __name__ == '__main__':
    main()
