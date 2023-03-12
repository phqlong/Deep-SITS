from typing import Any, List

import torch
from pytorch_lightning import LightningModule
from torchmetrics import MeanMetric, R2Score


class SITSLitModule(LightningModule):
    """ 
    A LightningModule organizes your PyTorch code into 6 sections:
        - Computations (init)
        - Train loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Prediction Loop (predict_step)
        - Optimizers and LR Schedulers (configure_optimizers)

    Docs:
        https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # Save model to calculate self.parameters() in configure_optimizers
        self.model = model

        # loss function
        self.criterion = torch.nn.MSELoss()

        # R2 Score metric
        self.r2_score = R2Score()

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()
        self.train_r2 = MeanMetric()
        self.val_r2 = MeanMetric()
        self.test_r2 = MeanMetric()

    def forward(self, x: torch.Tensor):
        return self.model(x)

    def model_step(self, batch: Any):
        x, y = batch
        preds = self.forward(x)
        loss = self.criterion(preds, y.float())
        r2 = self.r2_score(preds, y.float())
        return loss, r2

    def training_step(self, batch: Any, batch_idx: int):
        loss, r2 = self.model_step(batch)

        # update and log metrics
        self.train_loss(loss)
        self.train_r2(r2)
        self.log("train/loss", self.train_loss.compute(), on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/r2", self.train_r2.compute(), on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def training_epoch_end(self, outputs: List[Any]):
        self.train_loss.reset()
        self.train_r2.reset()

    def validation_step(self, batch: Any, batch_idx: int):
        loss, r2 = self.model_step(batch)

        # update and log metrics
        val_loss = self.val_loss(loss)
        val_r2 = self.val_r2(r2)
        self.log("val/loss", val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/r2", val_r2, on_step=False, on_epoch=True, prog_bar=True)

    def validation_epoch_end(self, outputs: List[Any]):
        self.val_loss.reset()
        self.val_r2.reset()

    def test_step(self, batch: Any, batch_idx: int):
        loss, r2 = self.model_step(batch)

        # update and log metrics
        test_loss = self.test_loss(loss)
        test_r2 = self.test_r2(r2)
        self.log("test/loss", test_loss, on_step=False,
                 on_epoch=True, prog_bar=True)
        self.log("test/r2", test_r2,
                 on_step=False, on_epoch=True, prog_bar=True)

    def test_epoch_end(self, outputs: List[Any]):
        self.test_loss.reset()
        self.test_r2.reset()

    def predict_step(self, batch: Any, batch_idx: int):
        x, y = batch
        preds = self.forward(x)
        return preds

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        # Lr and weight_decay are partially initialized in hydra.utils.instantiate(cfg.model)
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}


if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    cfg = omegaconf.OmegaConf.load(root / "configs" / "model" / "default.yaml")
    _ = hydra.utils.instantiate(cfg)
