import torch
from matplotlib import pyplot as plt
import yaml

from models.autoencoder import Autoencoder
from data.era5 import ERA5Dataset, gen_bgen


def main():

    with open("configs/autoencoder_kl_f8.yaml", "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    model = Autoencoder.load_from_checkpoint(
        "checkpoints/vae-kl-f8-1440x720-69c-step=23400.ckpt")

    model.eval()

    bgen = gen_bgen(train=False)
    dataset = ERA5Dataset(bgen)

    sample = dataset[200].unsqueeze(0)

    with torch.no_grad():
        sample = sample.to(model.device)

        rec, posterior = model.forward(sample)

    sample = sample.squeeze(0).cpu().numpy()
    rec = rec.squeeze(0).cpu().numpy()

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    ax[0].imshow(sample[0].T, cmap="turbo", origin="lower")
    ax[0].set_title("Input")
    ax[0].axes.get_xaxis().set_ticks([])
    ax[0].axes.get_yaxis().set_ticks([])

    ax[1].imshow(rec[0].T, cmap="turbo", origin="lower")
    ax[1].set_title("Reconstruction")
    ax[1].axes.get_xaxis().set_ticks([])
    ax[1].axes.get_yaxis().set_ticks([])
    plt.title("t2m")

    plt.savefig("image/t2m.png")

    ax[0].imshow(sample[1].T, cmap="turbo", origin="lower")
    ax[0].set_title("Input")
    ax[0].axes.get_xaxis().set_ticks([])
    ax[0].axes.get_yaxis().set_ticks([])

    ax[1].imshow(rec[1].T, cmap="turbo", origin="lower")
    ax[1].set_title("Reconstruction")
    ax[1].axes.get_xaxis().set_ticks([])
    ax[1].axes.get_yaxis().set_ticks([])
    plt.title("wind 10m u")

    plt.savefig("image/wind_u_10m.png")

    ax[0].imshow(sample[2].T, cmap="turbo", origin="lower")
    ax[0].set_title("Input")
    ax[0].axes.get_xaxis().set_ticks([])
    ax[0].axes.get_yaxis().set_ticks([])

    ax[1].imshow(rec[2].T, cmap="turbo", origin="lower")
    ax[1].set_title("Reconstruction")
    ax[1].axes.get_xaxis().set_ticks([])
    ax[1].axes.get_yaxis().set_ticks([])
    plt.title("wind 10m v")

    plt.savefig("image/wind_v_10m.png")

    ax[0].imshow(sample[3].T, cmap="turbo", origin="lower")
    ax[0].set_title("Input")
    ax[0].axes.get_xaxis().set_ticks([])
    ax[0].axes.get_yaxis().set_ticks([])

    ax[1].imshow(rec[3].T, cmap="turbo", origin="lower")
    ax[1].set_title("Reconstruction")
    ax[1].axes.get_xaxis().set_ticks([])
    ax[1].axes.get_yaxis().set_ticks([])
    plt.title("mean sea pressure")

    plt.savefig("image/mean_sea_pressure.png")

    for i in range(13):
        ax[0].imshow(sample[4 + i].T, cmap="turbo", origin="lower")
        ax[0].set_title("Input")
        ax[0].axes.get_xaxis().set_ticks([])
        ax[0].axes.get_yaxis().set_ticks([])

        ax[1].imshow(rec[4 + i].T, cmap="turbo", origin="lower")
        ax[1].set_title("Reconstruction")
        ax[1].axes.get_xaxis().set_ticks([])
        ax[1].axes.get_yaxis().set_ticks([])

        plt.title(f"temp {i}")

        plt.savefig(f"image/temp_{i}.png")

    for i in range(13):
        ax[0].imshow(sample[17 + i].T, cmap="turbo", origin="lower")
        ax[0].set_title("Input")
        ax[0].axes.get_xaxis().set_ticks([])
        ax[0].axes.get_yaxis().set_ticks([])

        ax[1].imshow(rec[17 + i].T, cmap="turbo", origin="lower")
        ax[1].set_title("Reconstruction")
        ax[1].axes.get_xaxis().set_ticks([])
        ax[1].axes.get_yaxis().set_ticks([])

        plt.title(f"wind u {i}")

        plt.savefig(f"image/wind_u_{i}.png")

    for i in range(13):
        ax[0].imshow(sample[30 + i].T, cmap="turbo", origin="lower")
        ax[0].set_title("Input")
        ax[0].axes.get_xaxis().set_ticks([])
        ax[0].axes.get_yaxis().set_ticks([])

        ax[1].imshow(rec[30 + i].T, cmap="turbo", origin="lower")
        ax[1].set_title("Reconstruction")
        ax[1].axes.get_xaxis().set_ticks([])
        ax[1].axes.get_yaxis().set_ticks([])

        plt.title(f"wind v {i}")

        plt.savefig(f"image/wind_v_{i}.png")

    for i in range(13):
        ax[0].imshow(sample[43 + i].T, cmap="turbo", origin="lower")
        ax[0].set_title("Input")
        ax[0].axes.get_xaxis().set_ticks([])
        ax[0].axes.get_yaxis().set_ticks([])

        ax[1].imshow(rec[43 + i].T, cmap="turbo", origin="lower")
        ax[1].set_title("Reconstruction")
        ax[1].axes.get_xaxis().set_ticks([])
        ax[1].axes.get_yaxis().set_ticks([])

        plt.title(f"geopotential {i}")

        plt.savefig(f"image/geopotential_{i}.png")

    for i in range(13):
        ax[0].imshow(sample[56 + i].T, cmap="turbo", origin="lower")
        ax[0].set_title("Input")
        ax[0].axes.get_xaxis().set_ticks([])
        ax[0].axes.get_yaxis().set_ticks([])

        ax[1].imshow(rec[56 + i].T, cmap="turbo", origin="lower")
        ax[1].set_title("Reconstruction")
        ax[1].axes.get_xaxis().set_ticks([])
        ax[1].axes.get_yaxis().set_ticks([])

        plt.title(f"specific humidity {i}")

        plt.savefig(f"image/spec_hum_{i}.png")


if __name__ == '__main__':
    main()
