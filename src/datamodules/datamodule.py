import os
import pandas as pd
from typing import Any, Dict, Optional, Tuple
from sklearn.model_selection import  KFold, GroupKFold, StratifiedKFold, StratifiedGroupKFold

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.transforms import transforms

from src.datamodules.components import SITSDataset

class SITSDataModule(LightningDataModule):
    """LightningDataModule for SITS dataset.

    A DataModule implements 5 key methods:

        def prepare_data(self):
            # things to do on 1 GPU/TPU (not on every GPU/TPU in DDP)
            # download data, pre-process, split, save to disk, etc...
        def setup(self, stage):
            # things to do on every process in DDP
            # load data, set variables, etc...
        def train_dataloader(self):
            # return train dataloader
        def val_dataloader(self):
            # return validation dataloader
        def test_dataloader(self):
            # return test dataloader
        def teardown(self):
            # called on every process in DDP
            # clean up after fit or test

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/data/datamodule.html
    """

    def __init__(
        self,
        data_dir: str = "data/",
        metadata_train_path: str = "data/Crop_Yield_Data_challenge_2.csv",
        metadata_test_path: str = "data/Challenge_2_submission_template.csv",
        data_train_path1: str = "data/train_sentinel1_normalized1.nc",
        data_train_path2: str = "data/train_sentinel1_normalized2.nc",
        data_test_path: str = "data/test_sentinel1_normalized2.nc",
        train_dataset_path: str = "data/train_dataset.pt",
        val_dataset_path: str = "data/val_dataset.pt",
        test_dataset_path: str = "data/test_dataset.pt",
        train_val_test_split: Tuple[float, float, float] = (0.9, 0.05, 0.05),
        bands: list = ["vv", "vh", "vv_by_vh", "vv_add_vh", "DOP", "RVI"],
        batch_size: int = 32,
        num_workers: int = 2,
        pin_memory: bool = False,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # self.metadata_train: Optional[pd.DataFrame] = None
        # self.metadata_test: Optional[pd.DataFrame] = None

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    def prepare_data(self):
        """
        Download data if needed and save for setup.
        Do not use it to assign state (self.x = y).
        """
        metadata_train = self.read_metadata(self.hparams.metadata_train_path)
        metadata_test = self.read_metadata(self.hparams.metadata_test_path, train=False)

        # load and split datasets only if not loaded already
        if not os.path.exists(self.hparams.train_dataset_path) and not os.path.exists(self.hparams.val_dataset_path) and not os.path.exists(self.hparams.test_dataset_path):
            # Split crop_yield_data into train, val, test b KFold
            data_train, data_val, data_test = self.split_dataset(metadata_train)

            # Load datasets from netcdf files
            train_ds = SITSDataset(crop_yield_data=data_train, 
                                          path1=self.hparams.data_train_path1,
                                          path2=self.hparams.data_train_path2,
                                          bands=self.hparams.bands)
            val_ds = SITSDataset(crop_yield_data=data_val, 
                                          path1=self.hparams.data_train_path1,
                                          path2=self.hparams.data_train_path2,
                                          bands=self.hparams.bands)
            test_ds = SITSDataset(crop_yield_data=data_test, 
                                          path1=self.hparams.data_train_path1,
                                          path2=self.hparams.data_train_path2,
                                          bands=self.hparams.bands)

            # Save datasets
            torch.save(train_ds, self.hparams.train_dataset_path)
            torch.save(val_ds, self.hparams.val_dataset_path)
            torch.save(test_ds, self.hparams.test_dataset_path)


    def setup(self, stage: Optional[str] = None):
        """
        Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """
        # self.metadata_train = self.read_metadata(self.hparams.metadata_train_path)
        # self.metadata_test = self.read_metadata(self.hparams.metadata_test_path, train=False)

        # load and split datasets only if not loaded already
        if not self.data_train and not self.data_val and not self.data_test:
            self.data_train = torch.load(self.hparams.train_dataset_path)
            self.data_val = torch.load(self.hparams.val_dataset_path)
            self.data_test = torch.load(self.hparams.test_dataset_path)

    def read_metadata(self, metadata_path, train=True):
        """Read metadata from csv file."""
        crop_yield_data = pd.read_csv(metadata_path)
        if train: 
            crop_yield_data = crop_yield_data.reset_index(drop=False)
        crop_yield_data["Date of Harvest"] = pd.to_datetime(crop_yield_data["Date of Harvest"], format="%d-%m-%Y")
        crop_yield_data.columns = ['ID',
                                    'District', 
                                    'Latitude',
                                    'Longitude',
                                    'Season',
                                    'Intensity',
                                    'HarvestDate',
                                    'FieldSize', 
                                    'RiceYield']
        return crop_yield_data

    def split_dataset(self, metadata):
        """Split dataset into train and validation sets."""
        gkf = KFold(n_splits=10, shuffle=True, random_state=42)
        # gkf = GroupKFold(n_splits=CFG.n_fold, shuffle=True, random_state=CFG.seed)
        # gkf = StratifiedKFold(n_splits=CFG.n_fold, shuffle=True, random_state=CFG.seed)
        # gkf = StratifiedGroupKFold(n_splits=CFG.n_fold, shuffle=True, random_state=CFG.seed)

        for fold, (train_id, val_id) in enumerate(gkf.split(X=metadata)):
            # For all row in val_id list => create kfold column value
            metadata.loc[val_id , "kfold"] = fold

        metadata.kfold = metadata.kfold.astype(int)

        data_train = metadata[metadata.kfold >= 2]
        data_val = metadata[metadata.kfold == 0]
        data_test = metadata[metadata.kfold == 1]
        return data_train, data_val, data_test

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def predict_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )
    
    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass


if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    cfg = omegaconf.OmegaConf.load(root / "configs" / "datamodule" / "default.yaml")
    cfg.data_dir = str(root / "data")
    _ = hydra.utils.instantiate(cfg)
