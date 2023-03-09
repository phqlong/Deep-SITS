import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

import datetime
import torch
import pandas as pd
from typing import List, Tuple

import hydra
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.loggers import LightningLoggerBase

from src import utils

log = utils.get_pylogger(__name__)


def create_submission(predictions: torch.Tensor, template_file_path: str, submission_dir: str) -> None:
    """Creates submission file from predictions.
    Args:
        predictions (Tensor): Tensor of predictions. Shape: (N, )
    """     
    # Reading the submission file template
    test_file = pd.read_csv(template_file_path)

    # Combining the results into dataframe
    test_file['Predicted Rice Yield (kg/ha)'] = (predictions*1000).long().tolist()
    
    # Dumping the predictions into a csv file.
    now = datetime.datetime.now()
    test_file.to_csv(f"{submission_dir}/{now.strftime('%Y%m%d_%H%M%S')}.csv", index=False)
    return

@utils.task_wrapper
def inference(cfg: DictConfig) -> Tuple[dict, dict]:
    """Inference given checkpoint on a datamodule testset.

    This method is wrapped in optional @task_wrapper decorator which applies extra utilities
    before and after the call.

    Args:
        cfg (DictConfig): Configuration composed by Hydra.

    Returns:
        Tuple[dict, dict]: Dict with metrics and dict with all instantiated objects.
    """

    assert cfg.ckpt_path

    log.info(f"Instantiating datamodule <{cfg.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.datamodule)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "trainer": trainer,
    }

    log.info("Starting inference!")
    predictions = trainer.predict(model=model, datamodule=datamodule, ckpt_path=cfg.ckpt_path)

    # Create submission file from predictions
    create_submission(predictions[0][:100], cfg.template_file_path, cfg.submission_dir)

    metric_dict = trainer.callback_metrics

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="inference.yaml")
def main(cfg: DictConfig) -> None:
    inference(cfg)


if __name__ == "__main__":
    main()
