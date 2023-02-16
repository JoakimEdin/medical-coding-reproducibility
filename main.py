import logging
import math
import os
from pathlib import Path

import hydra
from omegaconf import OmegaConf
from rich.pretty import pprint

from src.data.data_pipeline import data_pipeline
from src.factories import (
    get_callbacks,
    get_dataloaders,
    get_datasets,
    get_lookups,
    get_lr_scheduler,
    get_metric_collections,
    get_model,
    get_optimizer,
    get_text_encoder,
    get_transform,
)
from src.trainer.trainer import Trainer
from src.utils.seed import set_seed

LOGGER = logging.getLogger(name=__file__)
LOGGER.setLevel(logging.INFO)


def deterministic() -> None:
    """Run experiment deterministically. There will still be some randomness in the backward pass of the model."""
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

    import torch

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: OmegaConf) -> None:
    if cfg.deterministic:
        deterministic()
    else:
        import torch

    set_seed(cfg.seed)

    # Check if CUDA_VISIBLE_DEVICES is set
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        if cfg.gpu != -1 and cfg.gpu is not None and cfg.gpu != "":
            os.environ["CUDA_VISIBLE_DEVICES"] = (
                ",".join([str(gpu) for gpu in cfg.gpu])
                if isinstance(cfg.gpu, list)
                else str(cfg.gpu)
            )

        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = ""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pprint(f"Device: {device}")
    pprint(f"CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")

    data = data_pipeline(config=cfg.data)

    text_encoder = get_text_encoder(
        config=cfg.text_encoder, data_dir=cfg.data.dir, texts=data.get_train_documents
    )
    label_transform = get_transform(
        config=cfg.label_transform,
        targets=data.all_targets,
        load_transform_path=cfg.load_model,
    )
    text_transform = get_transform(
        config=cfg.text_transform,
        texts=data.get_train_documents,
        text_encoder=text_encoder,
        load_transform_path=cfg.load_model,
    )
    data.truncate_text(cfg.data.max_length)
    data.transform_text(text_transform.batch_transform)

    lookups = get_lookups(
        config=cfg.lookup,
        data=data,
        label_transform=label_transform,
        text_transform=text_transform,
    )

    model = get_model(
        config=cfg.model, data_info=lookups.data_info, text_encoder=text_encoder
    )
    model.to(device)

    # print data info
    pprint(lookups.data_info)

    metric_collections = get_metric_collections(
        config=cfg.metrics,
        number_of_classes=lookups.data_info["num_classes"],
        code_system2code_indices=lookups.code_system2code_indices,
        split2code_indices=lookups.split2code_indices,
    )
    datasets = get_datasets(
        config=cfg.dataset,
        data=data,
        text_transform=text_transform,
        label_transform=label_transform,
        lookups=lookups,
    )

    dataloaders = get_dataloaders(config=cfg.dataloader, datasets_dict=datasets)
    optimizer = get_optimizer(config=cfg.optimizer, model=model)
    accumulate_grad_batches = int(
        max(cfg.dataloader.batch_size / cfg.dataloader.max_batch_size, 1)
    )
    num_training_steps = (
        math.ceil(len(dataloaders["train"]) / accumulate_grad_batches)
        * cfg.trainer.epochs
    )
    lr_scheduler = get_lr_scheduler(
        config=cfg.lr_scheduler,
        optimizer=optimizer,
        num_training_steps=num_training_steps,
    )
    callbacks = get_callbacks(config=cfg.callbacks)

    trainer = Trainer(
        config=cfg,
        data=data,
        model=model,
        optimizer=optimizer,
        dataloaders=dataloaders,
        metric_collections=metric_collections,
        callbacks=callbacks,
        lr_scheduler=lr_scheduler,
        lookups=lookups,
        accumulate_grad_batches=accumulate_grad_batches,
    ).to(device)

    if cfg.load_model:
        trainer.experiment_path = Path(cfg.load_model)

    trainer.fit()


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
