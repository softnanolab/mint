import re
from pathlib import Path
import datetime
import os
from dotenv import load_dotenv, find_dotenv


import hydra
from omegaconf import DictConfig, OmegaConf, ListConfig
import lightning as pl
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.utilities import rank_zero_only

from mint.data.mint import PseduoMMDataModule
from mint.model.mint import MINT

MINT_PATH = Path(__file__).parent
CONFIG_PATH = str(MINT_PATH / "configs")

# Load environment variables from .env file
load_dotenv(find_dotenv())


# Register the 'now' resolver to save hydra logs indexed by datetime in the experiment_dir
OmegaConf.register_new_resolver("now", lambda pattern: datetime.datetime.now().strftime(pattern))

def upgrade_state_dict(state_dict):
    """Removes prefixes 'model.encoder.sentence_encoder.' and 'model.encoder.'."""
    prefixes = ["encoder.sentence_encoder.", "encoder."]
    pattern = re.compile("^" + "|".join(prefixes))
    state_dict = {pattern.sub("", name): param for name, param in state_dict.items()}
    return state_dict


@hydra.main(version_base=None, config_path=CONFIG_PATH, config_name="main")
def main(cfg: DictConfig):

    # Set matmul precision
    if cfg.meta.matmul_precision is not None:
        torch.set_float32_matmul_precision(cfg.meta.matmul_precision)

    # load model and data module
    model = MINT(cfg)
    data_module = PseduoMMDataModule(cfg.data)

    # Set up trainer
    strategy = "auto"
    if (isinstance(cfg.trainer.devices, int) and cfg.trainer.devices > 1) or (
        isinstance(cfg.trainer.devices, (list, ListConfig)) and len(cfg.trainer.devices) > 1
    ):
        strategy = DDPStrategy(find_unused_parameters=True)

    loggers = []
    wdb_logger = WandbLogger(
        name=cfg.wandb.name,
        group=cfg.wandb.name,
        save_dir=cfg.meta.experiment_dir,
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        log_model=False,
    )
    loggers.append(wdb_logger)
    # Save the config to wandb

    @rank_zero_only
    def save_config_to_wandb() -> None:
        config_out = Path(wdb_logger.experiment.dir) / "run.yaml"
        with Path.open(config_out, "w") as f:
            OmegaConf.save(cfg, f)
        wdb_logger.experiment.save(str(config_out))

    save_config_to_wandb()

    # load the trainer first
    trainer = pl.Trainer(
        default_root_dir=cfg.meta.experiment_dir,
        strategy=strategy,
        num_sanity_val_steps=2,
        enable_progress_bar=True,
        enable_checkpointing=True,
        callbacks=[
            # TODO: add monitor, save_every_n_train_steps, mode
            ModelCheckpoint(
                save_top_k=cfg.meta.save_top_k,
                save_last=True,
                every_n_epochs=1,
            )
        ],
        logger=loggers,
        **cfg.trainer,
    )

    trainer.fit(model=model, datamodule=data_module, ckpt_path=cfg.meta.resume)


if __name__ == "__main__":
    main()
