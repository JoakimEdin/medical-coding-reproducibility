from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from rich.progress import track

import src.metrics as metrics
from src.settings import (
    HUE_ORDER,
    MODEL_NAMES,
    PALETTE,
    PROJECT,
    TARGET_COLUMN,
    EXPERIMENT_DIR,
)
from reports.wandb_utils import get_best_runs

SWEEP_ID = "83jehgfy"


def one_hot(
    targets: list[list[str]], number_of_classes: int, target2index: dict[str, int]
) -> torch.Tensor:
    output_tensor = torch.zeros((len(targets), number_of_classes))
    for idx, case in enumerate(targets):
        for target in case:
            if target in target2index:
                output_tensor[idx, target2index[target]] = 1
    return output_tensor.long()


def load_predictions(run_id: str) -> tuple[dict[str, torch.Tensor], str, int]:
    predictions = pd.read_feather(EXPERIMENT_DIR / run_id / "predictions_test.feather")

    predictions[TARGET_COLUMN] = predictions[TARGET_COLUMN].apply(
        lambda x: x.tolist()
    )  # convert from numpy array to list
    targets = predictions[[TARGET_COLUMN]]
    unique_targets = list(set.union(*targets[TARGET_COLUMN].apply(set)))
    logits = predictions[unique_targets]
    target2index = {target: idx for idx, target in enumerate(unique_targets)}
    number_of_classes = len(target2index)

    # Mapping from target to index and vice versa
    targets_torch = one_hot(
        targets[TARGET_COLUMN].to_list(), number_of_classes, target2index
    )  # one_hot encoding of targets
    logits_torch = torch.tensor(logits.values)
    cases = {"logits": logits_torch, "targets": targets_torch}
    return cases, number_of_classes


runs = get_best_runs(
    PROJECT, "f1_micro", {"Sweep": {"$regex": f"{SWEEP_ID}"}, "State": "finished"}
)
# runs = [run for run in runs]
# runs = [get_run(PROJECT, "nae4hhly"), get_run(PROJECT, "3ld34ds4") ]
thresholds = np.linspace(0, 1, 100)
results = []

for model_name, run in runs.items():
    run_results = pd.DataFrame()

    if model_name in MODEL_NAMES:
        model_name = MODEL_NAMES[model_name]

    cases, number_of_classes = load_predictions(run.id)

    run_results["threshold"] = thresholds
    run_results["f1_score"] = np.zeros(len(thresholds))
    run_results["Model"] = model_name
    for idx, threshold in enumerate(track(thresholds)):
        metric = metrics.F1Score(
            number_of_classes=number_of_classes, average="micro", threshold=threshold
        )
        metric.update(cases)
        run_results.loc[idx, "f1_score"] = metric.compute().numpy()
    print(run_results["f1_score"].max())
    results.append(run_results)
results = pd.concat(results)

sns.set_theme("paper", style="whitegrid")

ax = sns.lineplot(
    data=results,
    x="threshold",
    y="f1_score",
    hue="Model",
    palette=PALETTE,
    hue_order=HUE_ORDER,
    linewidth=1.5,
)
ax.set(xlabel="Decision Boundary", ylabel=f"F1 Micro", ylim=(0.30, 0.65))
ax.grid(visible=True, which="minor", color="black", linewidth=0.075)
sns.move_legend(ax, "upper right")
plt.plot([0.5, 0.5], [0, 1], color="grey", linestyle="--")
ax.get_figure().savefig(
    f"files/images/mimiciii/threshold_tuning_micro_full.png",
    bbox_inches="tight",
    dpi=400,
)
plt.clf()
results = []

for model_name, run in runs.items():
    run_results = pd.DataFrame()

    if model_name in MODEL_NAMES:
        model_name = MODEL_NAMES[model_name]

    cases, number_of_classes = load_predictions(run.id)

    run_results["threshold"] = thresholds
    run_results["f1_score"] = np.zeros(len(thresholds))
    run_results["Model"] = model_name
    for idx, threshold in enumerate(track(thresholds)):
        metric = metrics.F1Score(
            number_of_classes=number_of_classes, average="macro", threshold=threshold
        )
        metric.update(cases)
        run_results.loc[idx, "f1_score"] = metric.compute().numpy()
    print(run_results["f1_score"].max())
    results.append(run_results)
results = pd.concat(results)


ax = sns.lineplot(
    data=results,
    x="threshold",
    y="f1_score",
    hue="Model",
    palette=PALETTE,
    hue_order=HUE_ORDER,
    linewidth=1.5,
)
ax.set(xlabel="Decision Boundary", ylabel=f"F1 Macro", ylim=(0, 0.35))
ax.grid(visible=True, which="minor", color="black", linewidth=0.075)
sns.move_legend(ax, "upper right")
plt.plot([0.5, 0.5], [0, 1], color="grey", linestyle="--")
ax.get_figure().savefig(
    f"files/images/mimiciii/threshold_tuning_macro_full.png",
    bbox_inches="tight",
    dpi=400,
)
