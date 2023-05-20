from typing import List, Optional

import hydra
import pyrootutils
import pytorch_lightning as pl
from omegaconf import DictConfig
from pytorch_lightning import Callback, LightningDataModule, LightningModule, Trainer
from pytorch_lightning.loggers import Logger

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

from a3d.utils.hydra_utils import (  # noqa: E402
    instantiate_callbacks,
    instantiate_loggers,
)


def train(cfg: DictConfig) -> None:
    # set seed for random number generators in pytorch, numpy and python.random
    if cfg.get("seed"):
        pl.seed_everything(cfg.seed, workers=True)

    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.datamodule)
    model: LightningModule = hydra.utils.instantiate(cfg.model)
    callbacks: List[Callback] = instantiate_callbacks(cfg.get("callbacks"))
    logger: List[Logger] = instantiate_loggers(cfg.get("logger"))
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)
    trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))


@hydra.main(version_base="1.3", config_path="../conf", config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    # train the model
    train(cfg)
    return None


if __name__ == "__main__":
    main()
