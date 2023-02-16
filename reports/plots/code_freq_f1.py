from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

import src.metrics as metrics
import wandb
from src.settings import (
    ID_COLUMN,
    MODEL_NAMES,
    PROJECT,
    TARGET_COLUMN,
    EXPERIMENT_DIR,
)
from reports.utils import get_db
from reports.wandb_utils import get_best_runs

sns.set_theme("paper", style="whitegrid", palette="colorblind", font_scale=1.5)


def one_hot(
    targets: list[list[str]], number_of_classes: int, target2index: dict[str, int]
) -> torch.Tensor:
    output_tensor = torch.zeros((len(targets), number_of_classes))
    for idx, case in enumerate(targets):
        for target in case:
            if target in target2index:
                output_tensor[idx, target2index[target]] = 1
    return output_tensor.long()


def load_predictions(
    run_id: str, split: str = "val"
) -> tuple[dict[str, torch.Tensor], str, int]:
    predictions = pd.read_feather(
        EXPERIMENT_DIR / run_id / f"predictions_{split}.feather"
    )

    predictions[TARGET_COLUMN] = predictions[TARGET_COLUMN].apply(
        lambda x: x.tolist()
    )  # convert from numpy array to list
    targets = predictions[[TARGET_COLUMN]]
    ids = predictions[ID_COLUMN].to_list()
    unique_targets = list(set.union(*targets[TARGET_COLUMN].apply(set)))
    logits = predictions[unique_targets]
    target2index = {target: idx for idx, target in enumerate(unique_targets)}
    index2target = {idx: target for idx, target in enumerate(unique_targets)}
    number_of_classes = len(target2index)

    # Mapping from target to index and vice versa
    targets_torch = one_hot(
        targets[TARGET_COLUMN].to_list(), number_of_classes, target2index
    )  # one_hot encoding of targets
    logits_torch = torch.tensor(logits.values)
    code_names = logits.columns.tolist()

    return logits_torch, targets_torch, ids, number_of_classes, code_names


def get_target_counts(targets: list[list[str]]) -> dict[str, int]:
    counts = {}
    for case in targets:
        for target in case:
            if target in counts:
                counts[target] += 1
            else:
                counts[target] = 1
    return counts


def plot_metric_vs_target_counts(
    metric_name: str,
    run: wandb.apis.public.Run,
    target_counts: dict[str, int],
) -> None:
    model_name = run.config["model"]["name"]

    if model_name in MODEL_NAMES:
        model_name = MODEL_NAMES[model_name]

    db = get_db(run.id)

    logits, targets, ids, number_of_classes, code_names = load_predictions(
        run.id, "test"
    )
    metric_class = getattr(metrics, metric_name)
    metric = metric_class(
        number_of_classes=number_of_classes, average=None, threshold=db
    )
    cases = {"logits": logits, "targets": targets}
    metric.update(cases)
    metric_name = metric_name.lower()

    results_df = pd.DataFrame({metric_name: metric.compute().numpy()}, index=code_names)
    results_df["code"] = code_names
    results_df["counts"] = results_df.index.map(lambda x: target_counts.get(x, 0))
    results_df = results_df[results_df["counts"] > 0]
    results_df["counts_log"] = np.log10(results_df["counts"])
    results_df["model"] = model_name
    print(f"{model_name}: ALL")
    print(
        f"Pearson Correlation: {results_df[['counts_log', metric_name]].corr()[metric_name]['counts_log']}"
    )
    print(
        f"Spearman Correlation: {results_df[['counts_log', metric_name]].corr(method='spearman')[metric_name]['counts_log']}"
    )

    return results_df


mimiciv_icd9 = pd.read_feather("path/to/mimiciv_icd9.feather")
mimiciv_icd9_splits = pd.read_feather("path/to/mimiciv_icd9_split.feather")
mimiciv_icd9 = mimiciv_icd9.merge(
    mimiciv_icd9_splits[[ID_COLUMN, "split"]], on=ID_COLUMN, how="left"
)

mimiciv_icd10 = pd.read_feather("path/to/mimiciv_icd10.feather")
mimiciv_icd10_splits = pd.read_feather("path/to/mimiciv_icd10_split.feather")
mimiciv_icd10 = mimiciv_icd10.merge(
    mimiciv_icd10_splits[[ID_COLUMN, "split"]], on=ID_COLUMN, how="left"
)

mimic_clean = pd.read_feather("path/to/mimiciii_clean.feather")
mimic_clean_splits = pd.read_feather("path/to/mimiciii_clean_split.feather")
mimic_clean = mimic_clean.merge(
    mimic_clean_splits[[ID_COLUMN, "split"]], on=ID_COLUMN, how="left"
)

train_mimiciv_icd9 = mimiciv_icd9[mimiciv_icd9["split"] == "train"]
train_mimiciv_icd10 = mimiciv_icd10[mimiciv_icd10["split"] == "train"]
train_mimic_clean = mimic_clean[mimic_clean["split"] == "train"]


target_counts_mimiciv_icd9 = get_target_counts(
    train_mimiciv_icd9[TARGET_COLUMN].tolist()
)
target_counts_mimiciv_icd10 = get_target_counts(
    train_mimiciv_icd10[TARGET_COLUMN].tolist()
)
target_counts_mimic_clean = get_target_counts(train_mimic_clean[TARGET_COLUMN].tolist())

sweep_id = "5ykjry46"
run_mimiciv_icd9 = get_best_runs(
    PROJECT,
    "f1_micro",
    {
        "Sweep": {"$regex": f"{sweep_id}"},
        "State": "finished",
        "config.model.name": "PLMICD",
    },
)["PLMICD"]

sweep_id = "46ah61d0"
run_mimiciv_icd10 = get_best_runs(
    PROJECT,
    "f1_micro",
    {
        "Sweep": {"$regex": f"{sweep_id}"},
        "State": "finished",
        "config.model.name": "PLMICD",
    },
)["PLMICD"]

sweep_id = "z89rf0ls"
run_mimic_clean = get_best_runs(
    PROJECT,
    "f1_micro",
    {
        "Sweep": {"$regex": f"{sweep_id}"},
        "State": "finished",
        "config.model.name": "PLMICD",
    },
)["PLMICD"]

results = []
df = plot_metric_vs_target_counts(
    "F1Score", run_mimiciv_icd9, target_counts_mimiciv_icd9
)
df["Dataset"] = "MIMIC-IV ICD-9"
results.append(df)

df = plot_metric_vs_target_counts(
    "F1Score", run_mimiciv_icd10, target_counts_mimiciv_icd10
)
df["Dataset"] = "MIMIC-IV ICD-10"
results.append(df)

# df = plot_metric_vs_target_counts("F1Score", run_mimic_clean, target_counts_mimic_clean)
# df["Dataset"] = "MIMIC-III clean"
# results.append(df)


# bin so each bin has same number of samples
result = results[0].copy().sort_values("counts")
# remove models in NOT_INCLUDED
bins = []
num_bins = 100
for idx in range(num_bins):
    bins.append(
        result.iloc[
            int(idx / num_bins * len(result)) : int((idx + 1) / num_bins * len(result))
        ]["counts"].mean()
    )

bins.append(result["counts"].min())
bins.append(5.1)
# #drop duplicated bins
# bins = sorted(list(set(bins)))
# log bins
# bins = np.logspace(0, 4, 50, base=10, dtype=int)
bins = sorted(list(set(bins)))
print(bins)

results_df = pd.concat(results)
results_df = results_df.reset_index(drop=True)
out = pd.cut(results_df["counts"], bins=bins)
results_df["counts_binned"] = out.apply(lambda x: float(x.mid))

g = sns.JointGrid(
    x="counts_binned",
    y="f1score",
    data=results_df,
    hue="Dataset",
)
g.plot_joint(
    sns.lineplot,
    errorbar="sd",
    # err_style="bars",
    linewidth=1.5,
    alpha=0.8,
)

sns.kdeplot(
    x="counts",
    # weights="counts",
    data=results_df,
    ax=g.ax_marg_x,
    log_scale=True,
    hue="Dataset",
    common_norm=False,
    bw_adjust=0.8,
    legend=False,
)
# sns.kdeplot(y="f1_micro", data=best_run_results, ax=g.ax_marg_y, hue="model", legend=False)
g.set_axis_labels("Code Occurences", "F1 Macro")
sns.move_legend(g.ax_joint, "lower right")
# remove y axis marginal plots
# g.ax_joint.set_ylim(0, 0.9)
g.ax_marg_y.remove()

g.ax_joint.set_xscale("log")
g.ax_marg_x.tick_params(labelleft=True)
g.ax_marg_x.grid(True, axis="y", ls=":")
g.ax_marg_x.yaxis.label.set_visible(True)
# g.ax_marg_x.set_yticks([0, 0.25, 0.5, 0.75, 1])
g.ax_joint.set_xlim(
    results_df["counts_binned"].min(), results_df["counts_binned"].max()
)

g.savefig(
    "files/images/code_count_vs_f1_all_datasets.png",
    format="png",
    dpi=400,
    bbox_inches="tight",
)

# bin so each bin has same number of samples
result = results[0][results[0]["f1score"] > 0].copy().sort_values("counts")
# remove models in NOT_INCLUDED
bins = []
num_bins = 100
for idx in range(num_bins):
    bins.append(
        result.iloc[
            int(idx / num_bins * len(result)) : int((idx + 1) / num_bins * len(result))
        ]["counts"].mean()
    )

bins.append(result["counts"].min())
bins.append(5.1)
# #drop duplicated bins
# bins = sorted(list(set(bins)))
# log bins
# bins = np.logspace(0, 4, 50, base=10, dtype=int)
bins = sorted(list(set(bins)))
print(bins)

results_df = results_df[results_df["f1score"] > 0]
results_df = results_df.reset_index(drop=True)
out = pd.cut(results_df["counts"], bins=bins)
results_df["counts_binned"] = out.apply(lambda x: float(x.mid))

g = sns.JointGrid(
    x="counts_binned",
    y="f1score",
    data=results_df,
    hue="Dataset",
)
g.plot_joint(
    sns.lineplot,
    errorbar="sd",
    # err_style="bars",
    linewidth=1.5,
    alpha=0.8,
)

sns.kdeplot(
    x="counts",
    # weights="counts",
    data=results_df,
    ax=g.ax_marg_x,
    log_scale=True,
    hue="Dataset",
    common_norm=False,
    bw_adjust=0.8,
    legend=False,
)
# sns.kdeplot(y="f1_micro", data=best_run_results, ax=g.ax_marg_y, hue="model", legend=False)
g.set_axis_labels("Code Occurences", "F1 Macro")
sns.move_legend(g.ax_joint, "lower right")
# remove y axis marginal plots
# g.ax_joint.set_ylim(0, 0.9)
g.ax_marg_y.remove()

g.ax_joint.set_xscale("log")
g.ax_marg_x.tick_params(labelleft=True)
g.ax_marg_x.grid(True, axis="y", ls=":")
g.ax_marg_x.yaxis.label.set_visible(True)
# g.ax_marg_x.set_yticks([0, 0.25, 0.5, 0.75, 1])
g.ax_joint.set_xlim(
    results_df["counts_binned"].min(), results_df["counts_binned"].max()
)

g.savefig(
    "files/images/code_count_vs_f1_all_datasets_filtered.png",
    format="png",
    dpi=400,
    bbox_inches="tight",
)
