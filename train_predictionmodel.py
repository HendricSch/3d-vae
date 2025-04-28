import torch
import torch.nn as nn
import lightning
from lightning.pytorch.callbacks import ModelCheckpoint

from data.latents import LatentsDataModule
from models.predictionnet import PredictionModel, DummyModel


def main():

    torch.set_float32_matmul_precision("medium")

    data = LatentsDataModule(batch_size=2)

    model = PredictionModel()
    # model = DummyModel()

    # checkpoint_callback = ModelCheckpoint(
    #     dirpath="checkpoints/",
    #     filename=config["config"]["general"]["name"] + "-{step}",
    #     every_n_train_steps=2500,

    # )

    trainer = lightning.Trainer(
        max_epochs=-1,
        precision="16-mixed",
    )

    trainer.fit(model, datamodule=data)

    # save the model
    trainer.save_checkpoint("checkpoints/prediction_model_moments.ckpt")


if __name__ == '__main__':
    main()
