import yaml

import torch
from tqdm import tqdm
from lightning.pytorch.utilities.model_summary import ModelSummary

from models.autoencoder import Autoencoder


def main():

    # print device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load a yaml file
    with open("configs/autoencoder_kl_f8.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    autoencoder = Autoencoder(config).to(device)

    # Print the model summary
    model_summary = ModelSummary(autoencoder, max_depth=2)
    print(model_summary)

    # dummy input
    x = torch.zeros(
        config["config"]["training"]["batch_size"],
        config["config"]["data"]["in_channels"],
        config["config"]["data"]["x"],
        config["config"]["data"]["y"],
        device=device
    )

    rec, posterior = autoencoder.forward(x)
    z = posterior.sample()

    loss = autoencoder.loss(x, rec, posterior)

    print("-------------------")
    print(f"Input shape: {x.shape}")
    print(f"Reconstruction shape: {rec.shape}")
    print(f"Latent shape: {z.shape}")
    print(f"Loss: {loss}")
    print("-------------------")

    del x, rec, z, loss

    # test loop

    for _ in tqdm(range(100)):
        x = torch.randn(
            config["config"]["training"]["batch_size"],
            config["config"]["data"]["in_channels"],
            config["config"]["data"]["x"],
            config["config"]["data"]["y"],
            device=device
        )

        rec, posterior = autoencoder.forward(x)
        z = posterior.sample()

        loss_rec, loss_kl = autoencoder.loss(x, rec, posterior)

        loss = loss_rec + loss_kl

        loss.backward()

        print(f"Loss: {loss}")

        del x, rec, z, loss

    print("Done!")


if __name__ == '__main__':
    main()
