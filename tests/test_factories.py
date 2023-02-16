from omegaconf import OmegaConf
import torch

from src.factories import (
    get_model,
    get_optimizer,
    get_metric_collection,
    get_metric_collections,
)
from src.models import BaseModel

data_info = {"num_classes": 10, "vocab_size": 100, "padding_idx": 0}
model_config = OmegaConf.create({"name": "BaseModel", "configs": {}})


def test_get_model() -> None:
    model = get_model(data_info=data_info, config=model_config)
    assert isinstance(model, BaseModel)


def test_get_optimizer() -> None:
    config = OmegaConf.create({"name": "Adam", "configs": {"lr": 0.001}})
    model = get_model(data_info=data_info, config=model_config)
    model.fc = torch.nn.Linear(10, 10)
    optimizer = get_optimizer(config, model)
    assert optimizer.__class__.__name__ == "Adam"
    assert optimizer.defaults["lr"] == 0.001


def test_get_metric_collection() -> None:
    config = OmegaConf.create([{"name": "Precision", "configs": {"average": "macro"}}])
    number_of_classes = 10
    metric_collection = get_metric_collection(config, number_of_classes)
    assert metric_collection.metrics[0].name == "precision_macro"
    assert metric_collection.metrics[0].number_of_classes == number_of_classes


def test_get_metric_collections() -> None:
    config = OmegaConf.create(
        [
            {"name": "Precision", "configs": {"average": "micro"}},
            {"name": "F1Score", "configs": {"average": "macro"}},
        ]
    )
    number_of_classes = 4
    splits_with_multiple_targets = set(["val", "test"])
    code_system2code_indices = {
        "icd9_proc": torch.tensor([0, 2]),
        "icd9_diag": torch.tensor([1, 3]),
    }
    split2code_indices = {"train": torch.tensor([0, 1]), "val": torch.tensor([2, 3])}
    metric_collections = get_metric_collections(
        config,
        number_of_classes,
        code_system2code_indices=code_system2code_indices,
        splits_with_multiple_code_systems=splits_with_multiple_targets,
    )

    assert metric_collections["train"]["all"].metrics[0].name == "precision_micro"
    assert metric_collections["train"]["all"].metrics[1].name == "f1_macro"
    assert (
        metric_collections["train"]["all"].metrics[0].number_of_classes
        == number_of_classes
    )
    assert metric_collections["val"]["all"].metrics[0].name == "precision_micro"
    assert metric_collections["val"]["all"].metrics[1].name == "f1_macro"
    assert (
        metric_collections["val"]["all"].metrics[0].number_of_classes
        == number_of_classes
    )
    assert metric_collections["test"]["all"].metrics[0].name == "precision_micro"
    assert metric_collections["test"]["all"].metrics[1].name == "f1_macro"
    assert (
        metric_collections["test"]["all"].metrics[0].number_of_classes
        == number_of_classes
    )

    assert metric_collections["val"]["icd9_proc"].metrics[0].name == "precision_micro"
    assert metric_collections["val"]["icd9_proc"].metrics[1].name == "f1_macro"
    assert metric_collections["val"]["icd9_proc"].metrics[0].number_of_classes == 2
    assert metric_collections["val"]["icd9_diag"].metrics[0].name == "precision_micro"
    assert metric_collections["val"]["icd9_diag"].metrics[1].name == "f1_macro"
    assert metric_collections["val"]["icd9_diag"].metrics[0].number_of_classes == 2

    assert metric_collections["test"]["icd9_proc"].metrics[0].name == "precision_micro"
    assert metric_collections["test"]["icd9_proc"].metrics[1].name == "f1_macro"
    assert metric_collections["test"]["icd9_proc"].metrics[0].number_of_classes == 2
    assert metric_collections["test"]["icd9_diag"].metrics[0].name == "precision_micro"
    assert metric_collections["test"]["icd9_diag"].metrics[1].name == "f1_macro"
    assert metric_collections["test"]["icd9_diag"].metrics[0].number_of_classes == 2

    assert "icd9_proc" not in metric_collections["train"]
    assert "icd9_diag" not in metric_collections["train"]

    logits = torch.tensor([[0.5, 0.6, 0.8, 0.4], [0.1, 0.56, 0.9, 0.3]]).to(
        device="cuda"
    )
    targets = torch.tensor([[0, 1, 1, 1], [0, 1, 1, 0]]).to(device="cuda")
    loss = torch.tensor([0.5, 0.6]).to(device="cuda")
    batch = {"logits": logits, "targets": targets, "loss": loss}

    for split in metric_collections:
        for target in metric_collections[split]:
            metric_collections[split][target].to(device="cuda")

    metric_collections["val"]["all"].update(batch)
    metric_collections["val"]["icd9_proc"].update(batch)
    metric_collections["val"]["icd9_diag"].update(batch)
    metric_collections["val"]["all"].update(batch)
    metric_collections["val"]["icd9_proc"].update(batch)
    metric_collections["val"]["icd9_diag"].update(batch)
    metric_collections["val"]["all"].update(batch)
    metric_collections["val"]["icd9_proc"].update(batch)
    metric_collections["val"]["icd9_diag"].update(batch)
    metric_collections["val"]["all"].update(batch)
    metric_collections["val"]["icd9_proc"].update(batch)
    metric_collections["val"]["icd9_diag"].update(batch)

    split = "val"
    targets = ["all", "icd9_proc", "icd9_diag"]
    for target in targets:
        print(target)
        metric_collections[split][target].update(batch)

    split = "val"
    targets = list(metric_collections[split].keys())
    for target in targets:
        print(target)
        metric_collections[split][target].update(batch)

    number_of_classes = 4
    splits_with_multiple_targets = set(["val", "test"])
    code_system2code_indices = {
        "icd9_proc": torch.tensor([0, 2]),
        "icd9_diag": torch.tensor([1, 3]),
    }
    split2code_indices = {"train": torch.tensor([0, 1]), "val": torch.tensor([3])}
    metric_collections = get_metric_collections(
        config,
        number_of_classes,
        code_system2code_indices=code_system2code_indices,
        split2code_indices=split2code_indices,
        splits_with_multiple_code_systems=splits_with_multiple_targets,
    )
    assert metric_collections["train"]["all"].metrics[0].name == "precision_micro"
    assert metric_collections["train"]["all"].metrics[1].name == "f1_macro"
    assert metric_collections["train"]["all"].metrics[0].number_of_classes == 2
    assert metric_collections["val"]["all"].metrics[0].name == "precision_micro"
    assert metric_collections["val"]["all"].metrics[1].name == "f1_macro"
    assert metric_collections["val"]["all"].metrics[0].number_of_classes == 1
    assert metric_collections["test"]["all"].metrics[0].name == "precision_micro"
    assert metric_collections["test"]["all"].metrics[1].name == "f1_macro"
    assert (
        metric_collections["test"]["all"].metrics[0].number_of_classes
        == number_of_classes
    )

    assert metric_collections["val"]["icd9_proc"].metrics[0].name == "precision_micro"
    assert metric_collections["val"]["icd9_proc"].metrics[1].name == "f1_macro"
    assert metric_collections["val"]["icd9_proc"].metrics[0].number_of_classes == 0
    assert metric_collections["val"]["icd9_diag"].metrics[0].name == "precision_micro"
    assert metric_collections["val"]["icd9_diag"].metrics[1].name == "f1_macro"
    assert metric_collections["val"]["icd9_diag"].metrics[0].number_of_classes == 1

    assert metric_collections["test"]["icd9_proc"].metrics[0].name == "precision_micro"
    assert metric_collections["test"]["icd9_proc"].metrics[1].name == "f1_macro"
    assert metric_collections["test"]["icd9_proc"].metrics[0].number_of_classes == 2
    assert metric_collections["test"]["icd9_diag"].metrics[0].name == "precision_micro"
    assert metric_collections["test"]["icd9_diag"].metrics[1].name == "f1_macro"
    assert metric_collections["test"]["icd9_diag"].metrics[0].number_of_classes == 2
