import torch
import torch.nn as nn

from typing import Literal

from models.discriminator import PatchGANDiscriminator
from models.blocks.distributions import DiagonalGaussianDistribution


class RecKLDiscriminatorLoss(nn.Module):
    def __init__(self, config: dict):
        super(RecKLDiscriminatorLoss, self).__init__()

        self.kl_weight = config["config"]["loss"]["kl_weight"]
        self.discriminator_weight = config["config"]["loss"]["discriminator_weight"]
        self.discriminator_start_steps = config["config"]["loss"]["discriminator_start_steps"]

        self.discriminator = PatchGANDiscriminator(
            in_channels=config["config"]["data"]["in_channels"],
            out_channels=config["config"]["data"]["out_channels"],
            channels=64
        )

        if config["config"]["loss"]["reconstruction_loss"] == "l1":
            self.reconstruction_loss_fn = torch.nn.L1Loss()
        elif config["config"]["loss"]["reconstruction_loss"] == "mse":
            self.reconstruction_loss_fn = torch.nn.MSELoss()
        else:
            raise ValueError(
                f"Invalid reconstruction loss: {config['config']['loss']['reconstruction_loss']}! Must be one of ['l1', 'mse']")

    def forward(
        self,
        inputs: torch.Tensor,
        reconstructions: torch.Tensor,
        posteriors: DiagonalGaussianDistribution,
        global_step: int,
        optimizer: Literal["Generator", "Discriminator"],
    ) -> tuple[torch.Tensor, dict]:

        if optimizer == "Generator":

            # Reconstruction loss
            rec_loss = self.reconstruction_loss_fn(reconstructions, inputs)

            # KL loss
            kl_loss = posteriors.kl().mean()
            kl_loss = kl_loss * self.kl_weight

            # Discriminator loss
            logits_fake = self.discriminator(reconstructions)
            g_loss = -torch.mean(logits_fake)
            g_loss = g_loss * self.discriminator_weight

            if global_step < self.discriminator_start_steps:
                g_loss = 0

            # Total loss
            loss = rec_loss + kl_loss + g_loss

            log = {
                "total_loss": loss,
                "rec_loss": rec_loss,
                "kl_loss": kl_loss,
                "g_loss": g_loss
            }

            return loss, log

        elif optimizer == "Discriminator":

            # Discriminator loss
            logits_real = self.discriminator(inputs)
            logits_fake = self.discriminator(reconstructions)

            d_loss = 0.5 * (
                torch.mean(torch.nn.functional.softplus(-logits_real)) +
                torch.mean(torch.nn.functional.softplus(logits_fake))
            )

            log = {
                "d_loss": d_loss
            }

            return d_loss, log

        else:
            raise ValueError(
                f"Invalid optimizer mode: {optimizer}! Must be one of ['Generator', 'Discriminator']")
