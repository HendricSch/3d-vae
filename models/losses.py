import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Literal

from models.discriminator import PatchGANDiscriminator
from models.blocks.distributions import DiagonalGaussianDistribution


def adopt_weight(weight: float, global_step: int, threshold: int = 0, value: float = 0.0) -> float:
    if global_step < threshold:
        weight = value
    return weight


def hinge_d_loss(logits_real: torch.Tensor, logits_fake: torch.Tensor) -> torch.Tensor:
    loss_real = torch.mean(F.relu(1. - logits_real))
    loss_fake = torch.mean(F.relu(1. + logits_fake))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def vanilla_d_loss(logits_real: torch.Tensor, logits_fake: torch.Tensor) -> torch.Tensor:
    d_loss = 0.5 * (
        torch.mean(torch.nn.functional.softplus(-logits_real)) +
        torch.mean(torch.nn.functional.softplus(logits_fake)))
    return d_loss


class RecKLDiscriminatorLoss(nn.Module):
    def __init__(self, config: dict):
        super(RecKLDiscriminatorLoss, self).__init__()

        self.kl_weight = config["config"]["loss"]["kl_weight"]
        self.discriminator_weight = config["config"]["loss"]["discriminator_weight"]
        self.disc_factor = 1.0
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

    def calculate_adaptive_weight(self, rec_loss, g_loss, last_layer):

        if last_layer is None:
            raise ValueError(
                "Last layer is required for adaptive weight calculation!")

        rec_grads = torch.autograd.grad(
            rec_loss, last_layer, retain_graph=True)[0]
        g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]

        d_weight = torch.norm(rec_grads) / (torch.norm(g_grads) + 1e-4)
        print(d_weight)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        print(d_weight)
        d_weight = d_weight * self.discriminator_weight
        print(d_weight)

        return d_weight

    def forward(
        self,
        inputs: torch.Tensor,
        reconstructions: torch.Tensor,
        posteriors: DiagonalGaussianDistribution,
        global_step: int,
        optimizer: Literal["Generator", "Discriminator"],
        last_layer: torch.Tensor,
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

            try:
                d_weight = self.calculate_adaptive_weight(
                    rec_loss, g_loss, last_layer=last_layer)
            except RuntimeError:
                assert not self.training
                d_weight = torch.tensor(0.0)

            disc_factor = adopt_weight(
                weight=self.disc_factor,
                global_step=global_step,
                threshold=self.discriminator_start_steps,
                value=0.0
            )

            # Total loss
            loss = rec_loss + kl_loss + d_weight * disc_factor * g_loss

            log = {
                "total_loss": loss.clone().detach().mean(),
                "kl_loss": kl_loss.detach().mean(),
                "rec_loss": rec_loss.detach().mean(),
                "d_weight": d_weight.detach(),
                "disc_factor": torch.tensor(disc_factor),
                "g_loss": g_loss.detach().mean()
            }

            return loss, log

        elif optimizer == "Discriminator":

            # Discriminator loss
            logits_real = self.discriminator(inputs.detach())
            logits_fake = self.discriminator(reconstructions.detach())

            disc_factor = adopt_weight(
                weight=self.disc_factor,
                global_step=global_step,
                threshold=self.discriminator_start_steps,
                value=0.0
            )

            d_loss = disc_factor * hinge_d_loss(logits_real, logits_fake)

            log = {
                "d_loss": d_loss.clone().detach().mean(),
                "logits_real": logits_real.detach().mean(),
                "logits_fake": logits_fake.detach().mean()
            }

            return d_loss, log

        else:
            raise ValueError(
                f"Invalid optimizer mode: {optimizer}! Must be one of ['Generator', 'Discriminator']")
