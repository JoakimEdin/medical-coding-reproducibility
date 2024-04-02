import re
from pathlib import Path
from typing import Optional

import torch
from omegaconf import OmegaConf
import wandb

from src.settings import EXPERIMENT_DIR

FORMATTING_PATTERN = r"\[([^\]]+)\]"
FORMATTING_REGEX = re.compile(FORMATTING_PATTERN)

def length_of_formatting(string: str):
    return sum(
        len(s) + 2 for s in FORMATTING_REGEX.findall(string)
    )  # plus 2 for parenthesis


def length_without_formatting(string: str):
    return len(string) - length_of_formatting(string)


def source_string(source):
    return f"{source[:18]}.." if len(source) > 20 else f"{source}"


class BaseCallback:
    def __init__(self):
        pass

    def on_initialisation_end(self, trainer=None):
        pass

    def on_train_begin(self, trainer=None):
        pass

    def on_train_end(self, trainer=None):
        pass

    def on_val_begin(self, trainer=None):
        pass

    def on_val_end(self, trainer=None):
        pass

    def on_epoch_begin(self, trainer=None):
        pass

    def on_epoch_end(self, trainer=None):
        pass

    def on_batch_begin(self, trainer=None):
        pass

    def on_batch_end(self, trainer=None):
        pass

    def on_fit_begin(self, trainer=None):
        pass

    def on_fit_end(self, trainer=None):
        pass

    def log_dict(
        self, nested_dict: dict[str, dict[str, torch.Tensor]], epoch: int
    ) -> None:
        pass

    def on_end(self, trainer=None):
        pass


class WandbCallback(BaseCallback):
    def __init__(self, config: OmegaConf):
        super().__init__()
        self.config = config

    def extract_tags(self, trainer) -> list[str]:
        tags = []
        tags.append(trainer.config.model.name)
        tags.append(Path(trainer.config.data.data_filename).stem)
        tags += trainer.config.data.code_column_names
        return tags

    def on_initialisation_end(self, trainer=None):
        wandb_cfg = OmegaConf.to_container(
            trainer.config, resolve=True, throw_on_missing=True
        )
        tags = self.extract_tags(trainer)
        if trainer.config.debug:
            mode = "disabled"
        else:
            mode = "online"

        wandb_cfg["num_parameters"] = sum(p.numel() for p in trainer.model.parameters())
        wandb_cfg["num_trainable_parameters"] = sum(
            p.numel() for p in trainer.model.parameters() if p.requires_grad
        )
        if trainer.lookups is not None:
            wandb_cfg["data_info"] = trainer.lookups.data_info
            OmegaConf.save(trainer.config, trainer.experiment_path / "config.yaml")

        wandb.init(
            config=wandb_cfg,
            settings=wandb.Settings(start_method="thread"),
            tags=tags,
            name=trainer.config.name,
            mode=mode,
            dir=EXPERIMENT_DIR,
            **self.config,
        )
        wandb.watch(trainer.model)
        if not trainer.config.debug:
            # make a folder for the model where configs and model weights are saved
            trainer.experiment_path = (
                Path(EXPERIMENT_DIR) / wandb.run.id
            )
            trainer.experiment_path.mkdir(exist_ok=False)

    def log_dict(
        self,
        nested_dict: dict[str, dict[str, torch.Tensor]],
        epoch: Optional[int] = None,
    ) -> None:
        nested_dict["epoch"] = epoch
        wandb.log(nested_dict)

    def on_end(self, trainer=None):
        wandb.finish()


class SaveBestModelCallback(BaseCallback):
    def __init__(self, config: OmegaConf):
        super().__init__()
        self.config = config
        self.prev_best = None
        self.split_name = config.split
        self.target_name = config.target
        self.metric_name = config.metric

    def on_epoch_end(self, trainer=None):
        best_metric = trainer.metric_collections[self.split_name][
            self.target_name
        ].get_best_metric(self.metric_name)
        if self.prev_best is None or best_metric != self.prev_best:
            self.prev_best = best_metric
            trainer.save_checkpoint("best_model.pt")
            print("Saved best model")

    def on_fit_end(self, trainer=None):
        trainer.load_checkpoint("best_model.pt")
        print("Loaded best model")


class EarlyStoppingCallback(BaseCallback):
    def __init__(self, config: OmegaConf):
        super().__init__()
        self.config = config
        self.split_name = config.split
        self.target_name = config.target
        self.metric_name = config.metric
        self.patience = config.patience
        self.counter = 0
        self.prev_best = None

    def on_epoch_end(self, trainer=None):
        """On the end of each epoch, test if the validation metric has improved. If it hasn't improved for self.patience epochs, stop training.

        Args:
            trainer (Trainer, optional): Trainer class. Defaults to None.
        """
        best_metric = trainer.metric_collections[self.split_name][
            self.target_name
        ].get_best_metric(self.metric_name)
        if self.prev_best is None or best_metric > self.prev_best:
            self.prev_best = best_metric
            self.counter = 0
        else:
            self.counter += 1
        if self.counter >= self.patience:
            trainer.stop_training = True
            print(
                f"Early stopping: {self.counter} epochs without improvement for {self.metric_name}"
            )
