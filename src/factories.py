from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from omegaconf import OmegaConf
from rich.pretty import pprint
from torch.utils.data import DataLoader

import src.data.batch_sampler as batch_samplers
import src.data.datasets as datasets
import src.data.transform as transform
import src.metrics as metrics
import src.models as models
import src.text_encoders as text_encoders
import src.trainer.callbacks as callbacks
from src.data.datatypes import Data, Lookups
from src.lookups import load_lookups
from src.text_encoders.base_text_encoder import BaseTextEncoder


def get_lookups(
    config: OmegaConf,
    data: Data,
    label_transform: transform.Transform,
    text_transform: transform.Transform,
) -> Lookups:
    return load_lookups(
        config=config,
        data=data,
        label_transform=label_transform,
        text_transform=text_transform,
    )


def get_model(
    config: OmegaConf, data_info: dict, text_encoder: Optional[Any] = None
) -> models.BaseModel:
    model_class = getattr(models, config.name)
    return model_class(text_encoder=text_encoder, **data_info, **config.configs)


def get_optimizer(config: OmegaConf, model: models.BaseModel) -> torch.optim.Optimizer:
    optimizer_class = getattr(torch.optim, config.name)
    return optimizer_class(model.parameters(), **config.configs)


def get_lr_scheduler(
    config: OmegaConf,
    optimizer: torch.optim.Optimizer,
    num_training_steps: int,
) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    if config.name is None:
        return None
    if hasattr(torch.optim.lr_scheduler, config.name):
        lr_scheduler_class = getattr(torch.optim.lr_scheduler, config.name)
        return lr_scheduler_class(optimizer, **config.configs)
    from transformers import get_scheduler

    return get_scheduler(
        name=config.name,
        optimizer=optimizer,
        num_training_steps=num_training_steps,
        **config.configs,
    )


def get_text_encoder(
    config: OmegaConf, data_dir: str, texts: list[str]
) -> text_encoders.BaseTextEncoder:
    if not hasattr(config, "name"):
        return None

    path = Path(data_dir) / config.file_name
    text_encoder_class = getattr(text_encoders, config.name)
    if path.exists() and config.load_model:
        return text_encoder_class.load(path)

    text_encoder = text_encoder_class(config=config.configs)
    text_encoder.fit(texts)
    text_encoder.save(path)
    return text_encoder


def get_transform(
    config: OmegaConf,
    targets: Optional[set[str]] = None,
    texts: Optional[list[str]] = None,
    text_encoder: Optional[BaseTextEncoder] = None,
    load_transform_path: Optional[str] = None,
) -> transform.Transform:
    """Get transform class

    Args:
        config (OmegaConf): Config for the transform
        targets (Optional[set[str]], optional): Groundtruth targets. Defaults to None.
        texts (Optional[list[str]], optional): Text. If text is not provided, the transform will be fitted on the targets. Defaults to None.
        text_encoder (Optional[BaseTextEncoder], optional): Text encoder. If not none, the tokenmap will be loaded from the text encoder. Defaults to None.
        load_transform_path (Optional[str], optional): Path of where the transform is saved. Will be loaded if not none. Defaults to None.

    Raises:
        ValueError: Error if either texts, targets, text_encoder or load_transform_path are not provided.

    Returns:
        transform.Transform: _description_
    """
    transform_class = getattr(transform, config.name)(**config.configs)
    if load_transform_path:
        transform_class.load(load_transform_path)
        print("loaded transform")

    elif text_encoder:
        transform_class.set_tokenmap(
            token2index=text_encoder.token2index, index2token=text_encoder.index2token
        )

    elif texts:
        transform_class.fit(texts)

    elif targets:
        transform_class.fit(targets)

    else:
        raise ValueError(
            "Provide set of labels, a text encoder or texts of tokens to perform fit transformation"
        )

    return transform_class


def get_datasets(
    config: OmegaConf,
    data: Data,
    text_transform: transform.Transform,
    label_transform: transform.Transform,
    lookups: Lookups,
) -> dict[str, datasets.BaseDataset]:
    dataset_class = getattr(datasets, config.name)
    datasets_dict = {}

    train_data = data.train
    val_data = data.val
    test_data = data.test
    del data.df

    datasets_dict["train"] = dataset_class(
        train_data,
        split_name="train",
        text_transform=text_transform,
        label_transform=label_transform,
        lookups=lookups,
        **config.configs,
    )
    datasets_dict["val"] = dataset_class(
        val_data,
        split_name="val",
        text_transform=text_transform,
        label_transform=label_transform,
        lookups=lookups,
        **config.configs,
    )
    datasets_dict["test"] = dataset_class(
        test_data,
        split_name="test",
        text_transform=text_transform,
        label_transform=label_transform,
        lookups=lookups,
        **config.configs,
    )
    return datasets_dict


def get_dataloaders(
    config: OmegaConf, datasets_dict: dict[str, datasets.BaseDataset]
) -> dict[str, DataLoader]:
    dataloaders = {}
    train_batch_size = min(config.batch_size, config.max_batch_size)
    pprint(f"Train batch size: {train_batch_size}")
    if config.batch_sampler.name:
        batch_sampler_class = getattr(batch_samplers, config.batch_sampler.name)
        batch_sampler = batch_sampler_class(
            dataset=datasets_dict["train"],
            batch_size=train_batch_size,
            drop_last=config.drop_last,
            **config.batch_sampler.configs,
        )
        dataloaders["train"] = DataLoader(
            datasets_dict["train"],
            batch_sampler=batch_sampler,
            collate_fn=datasets_dict["train"].collate_fn,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
        )
    else:
        dataloaders["train"] = DataLoader(
            datasets_dict["train"],
            shuffle=True,
            batch_size=train_batch_size,
            drop_last=config.drop_last,
            collate_fn=datasets_dict["train"].collate_fn,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
        )
    dataloaders["train_val"] = DataLoader(
        datasets_dict["train"],
        batch_size=config.max_batch_size,
        shuffle=False,
        collate_fn=datasets_dict["val"].collate_fn,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )

    dataloaders["val"] = DataLoader(
        datasets_dict["val"],
        batch_size=config.max_batch_size,
        shuffle=False,
        collate_fn=datasets_dict["val"].collate_fn,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )

    dataloaders["test"] = DataLoader(
        datasets_dict["test"],
        batch_size=config.max_batch_size,
        shuffle=False,
        collate_fn=datasets_dict["test"].collate_fn,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )

    return dataloaders


def get_metric_collection(
    config: OmegaConf,
    number_of_classes: int,
    code_system_code_indices: Optional[torch.Tensor] = None,
    split_code_indices: Optional[torch.Tensor] = None,
    code_system_name: Optional[str] = None,
) -> metrics.MetricCollection:
    metric_list = []
    for metric in config:
        metric_class = getattr(metrics, metric.name)
        metric_list.append(
            metric_class(number_of_classes=number_of_classes, **metric.configs)
        )

    if code_system_code_indices is not None and split_code_indices is not None:
        # Get overlapping indices
        code_indices = torch.tensor(
            np.intersect1d(code_system_code_indices, split_code_indices)
        )
    elif code_system_code_indices is not None:
        code_indices = code_system_code_indices.clone()
    elif split_code_indices is not None:
        code_indices = split_code_indices.clone()
    else:
        code_indices = None

    return metrics.MetricCollection(
        metrics=metric_list,
        code_indices=code_indices,
        code_system_name=code_system_name,
    )


def get_metric_collections(
    config: OmegaConf,
    number_of_classes: int,
    split_names: list[str] = ["train", "train_val", "val", "test"],
    splits_with_multiple_code_systems: set[str] = {"train_val", "val", "test"},
    code_system2code_indices: dict[str, torch.Tensor] = None,
    split2code_indices: dict[str, torch.Tensor] = None,
) -> dict[str, dict[str, metrics.MetricCollection]]:
    metric_collections = defaultdict(dict)
    for split_name in split_names:
        if split2code_indices is not None:
            split_code_indices = split2code_indices.get(split_name)
        else:
            split_code_indices = None

        metric_collections[split_name]["all"] = get_metric_collection(
            config=config,
            number_of_classes=number_of_classes,
            split_code_indices=split_code_indices,
        )

        if split_name not in splits_with_multiple_code_systems:
            continue

        if code_system2code_indices is None:
            continue

        for (
            code_system_name,
            code_system_code_indices,
        ) in code_system2code_indices.items():
            metric_collections[split_name][code_system_name] = get_metric_collection(
                config=config,
                number_of_classes=number_of_classes,
                code_system_code_indices=code_system_code_indices,
                split_code_indices=split_code_indices,
                code_system_name=code_system_name,
            )

    return metric_collections


def get_callbacks(config: OmegaConf) -> list[callbacks.BaseCallback]:
    callbacks_list = []
    for callback in config:
        callback_class = getattr(callbacks, callback.name)
        callbacks_list.append(callback_class(config=callback.configs))
    return callbacks_list
