import torch
import lightning as pl

from models.blocks.encoder import Encoder
from models.blocks.decoder import Decoder

from models.blocks.distributions import DiagonalGaussianDistribution


class Autoencoder(pl.LightningModule):
    def __init__(self, config: dict):
        super().__init__()

        ### Load the configuration ###

        self.config = config["config"]

        # data
        self.x = self.config["data"]["x"]
        self.y = self.config["data"]["y"]
        self.in_channels = self.config["data"]["in_channels"]
        self.out_channels = self.config["data"]["out_channels"]

        # model
        self.z_channels = self.config["model"]["z_channels"]
        self.embed_dim = self.config["model"]["embed_dim"]
        self.channels = self.config["model"]["channels"]
        self.channel_mult = self.config["model"]["channel_mult"]
        self.num_res_blocks = self.config["model"]["num_res_blocks"]
        self.attention = self.config["model"]["attention"]

        # loss
        self.reconstruction_loss_fn = self.config["loss"]["reconstruction_loss"]
        self.kl_weight = self.config["loss"]["kl_weight"]

        # training
        self.learning_rate = self.config["training"]["learning_rate"]
        self.epochs = self.config["training"]["epochs"]
        self.batch_size = self.config["training"]["batch_size"]
        self.lr_scheduler = self.config["training"]["lr_scheduler"]

        # lightning
        self.save_hyperparameters()
        self.lr = self.learning_rate
        self.example_input_array = torch.zeros(
            self.batch_size, self.in_channels, self.x, self.y
        )

        ### Initialize the model ###

        # Encoder and decoder
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)

        # Quantization layers
        self.quant_conv = torch.nn.Conv2d(
            2 * self.z_channels, 2 * self.embed_dim, 1
        )

        # Post-quantization layers
        self.post_quant_conv = torch.nn.Conv2d(
            self.embed_dim, self.z_channels, 1)

    def encode(self, x: torch.Tensor) -> DiagonalGaussianDistribution:

        h = self.encoder(x)
        moments = self.quant_conv(h)

        posterior = DiagonalGaussianDistribution(moments)

        return posterior

    def decode(self, z: torch.Tensor) -> torch.Tensor:

        z = self.post_quant_conv(z)
        rec = self.decoder(z)

        return rec

    def forward(self, input, sample_posterior=True) -> tuple[torch.Tensor, DiagonalGaussianDistribution]:

        posterior = self.encode(input)

        if sample_posterior:
            z = posterior.sample()

        else:
            z = posterior.mode()

        dec = self.decode(z)

        return dec, posterior

    def loss(self, inputs: torch.Tensor, reconstructions: torch.Tensor, posterior: DiagonalGaussianDistribution) -> tuple[torch.Tensor, torch.Tensor]:

        rec_loss_fn = None

        if self.reconstruction_loss_fn == "l1":
            rec_loss_fn = torch.nn.functional.l1_loss

        elif self.reconstruction_loss_fn == "mse":
            rec_loss_fn = torch.nn.functional.mse_loss

        else:
            raise ValueError(
                f"Invalid reconstruction loss: {self.reconstruction_loss_fn}")

        rec_loss = rec_loss_fn(reconstructions, inputs)

        kl_loss = posterior.kl().mean()
        kl_loss = kl_loss * self.kl_weight

        return rec_loss, kl_loss

    def training_step(self, batch: torch.Tensor, batch_idx):

        inputs = batch
        rec, posterior = self.forward(inputs)

        rec_loss, kl_loss = self.loss(inputs, rec, posterior)

        loss = rec_loss + kl_loss

        self.log("loss", loss, prog_bar=True,
                 logger=True, on_step=True, on_epoch=True)

        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx):

        inputs = batch
        rec, posterior = self.forward(inputs)

        rec_loss, kl_loss = self.loss(inputs, rec, posterior)

        loss = rec_loss + kl_loss

        self.log("val/rec_loss", rec_loss)

        return loss

    def configure_optimizers(self):

        lr = self.learning_rate

        opt_ae = torch.optim.Adam(list(self.encoder.parameters()) +
                                  list(self.decoder.parameters()) +
                                  list(self.quant_conv.parameters()) +
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr, betas=(0.5, 0.9))

        if self.lr_scheduler is not None:
            raise NotImplementedError(
                "Learning rate scheduler not implemented")

        return opt_ae

    @torch.no_grad()
    def log_images(self, batch):
        pass
