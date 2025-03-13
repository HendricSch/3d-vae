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
    model_summary = ModelSummary(autoencoder, max_depth=-1)
    print(model_summary)

    # dummy input
    x = torch.zeros(
        config["config"]["training"]["batch_size"],
        config["config"]["data"]["in_channels"],
        config["config"]["data"]["x"],
        config["config"]["data"]["y"]
    )

    rec, posterior = autoencoder.forward(x)
    z = posterior.sample()

    loss = autoencoder.loss(x, rec, posterior)

    print("-------------------")
    print(f"Input shape: {x.shape}")
    print(f"Reconstruction shape: {rec.shape}")
    print(f"Posterior shape: {z.shape}")
    print(
        f"KL: {posterior.kl().mean() * config['config']['loss']['kl_weight']}")
    print(f"Loss: {loss}")
    print("-------------------")


if __name__ == '__main__':
    main()
