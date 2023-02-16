from pathlib import Path

import pandas as pd
import seaborn as sns
import torch

import src.metrics as metrics
from reports.utils import get_db
from reports.wandb_utils import get_best_runs
from src.settings import (
    DATA_DIRECTORY_MIMICIV_ICD9,
    EXPERIMENT_DIR,
    HUE_ORDER,
    ID_COLUMN,
    MODEL_NAMES,
    PALETTE,
    PROJECT,
    TARGET_COLUMN,
)

sns.set_theme("paper", style="whitegrid", palette="muted")

SWEEP_ID = "5ykjry46"
df = pd.read_feather(Path(DATA_DIRECTORY_MIMICIV_ICD9) / "mimiciv_icd9.feather")


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

    return logits_torch, targets_torch, ids, number_of_classes, index2target


def get_predicted_codes(logits, index2target):
    predicted_codes = []
    for logit in logits:
        predicted_codes.append(
            [index2target[idx] for idx in torch.where(logit == 1)[0].tolist()]
        )
    return predicted_codes


def get_document_results(run) -> pd.DataFrame:
    logits, targets, ids, number_of_classes, index2target = load_predictions(
        run.id, "test"
    )
    db = get_db(run.id)
    metric_collection = metrics.MetricCollection(
        [
            metrics.MeanAveragePrecision(),
            metrics.PrecisionAtRecall(),
            metrics.F1Score(
                number_of_classes=number_of_classes, average="micro", threshold=db
            ),
            metrics.Recall(
                number_of_classes=number_of_classes, average="micro", threshold=db
            ),
            metrics.Precision(
                number_of_classes=number_of_classes, average="micro", threshold=db
            ),
        ]
    )
    results = {}
    for logit, target, id in zip(logits, targets, ids):
        logit = logit.unsqueeze(0)
        target = target.unsqueeze(0)
        batch = {"logits": logit, "targets": target}
        metric_collection.update(batch)
        results[id] = {
            key: value.item() for key, value in metric_collection.compute().items()
        }
        metric_collection.reset()

    results_df = (
        pd.DataFrame.from_dict(results, orient="index")
        .reset_index()
        .rename(columns={"index": ID_COLUMN})
    )
    results_df["predicted_codes"] = get_predicted_codes(logits > db, index2target)
    results_df = results_df.merge(df, on=ID_COLUMN, how="inner")
    results_df["num_words"] = results_df["text"].apply(lambda x: len(x.split()))
    # results_df[results_df["num_words"]>4000]["num_words"]=4000
    results_df["num_codes"] = results_df[TARGET_COLUMN].apply(lambda x: len(list(x)))
    return results_df


best_runs = runs = get_best_runs(
    PROJECT, "f1_micro", {"Sweep": {"$regex": f"{SWEEP_ID}"}, "State": "finished"}
)
best_run_results = []
for model_name, run in best_runs.items():
    if model_name in MODEL_NAMES:
        model_name = MODEL_NAMES[model_name]

    results_df = get_document_results(run)
    results_df["model"] = model_name
    best_run_results.append(results_df)

best_run_results = pd.concat(best_run_results)

best_run_results = best_run_results.copy().sort_values("num_words")
bins = []
for idx in range(50):
    bins.append(
        best_run_results.iloc[
            int(idx / 50 * len(best_run_results)) : int(
                (idx + 1) / 50 * len(best_run_results)
            )
        ]["num_words"].min()
        - 1
    )
bins.append(
    best_run_results.iloc[
        int(idx / 50 * len(best_run_results)) : int(
            (idx + 1) / 50 * len(best_run_results)
        )
    ]["num_words"].max()
    + 1
)
bins.append(best_run_results["num_words"].max() + 1)
bins = sorted(list(set(bins)))
print(bins)


out = pd.cut(best_run_results["num_words"], bins=bins)
best_run_results["num_words_binned"] = out.apply(lambda x: x.mid)

g = sns.JointGrid(
    x="num_words_binned",
    y="f1_micro",
    data=best_run_results,
    hue="model",
    palette=PALETTE,
    hue_order=HUE_ORDER,
)
g.plot_joint(sns.lineplot, errorbar=None, linewidth=1.5)
sns.histplot(
    x="num_words",
    data=best_run_results[best_run_results["model"] == "PLM-ICD"],
    ax=g.ax_marg_x,
)
# sns.kdeplot(y="f1_micro", data=best_run_results, ax=g.ax_marg_y, hue="model", legend=False)
g.refline(x=4000)
g.set_axis_labels("Text Length (# words)", "F1 Micro")
sns.move_legend(g.ax_joint, "upper left")
# remove y axis marginal plots
g.ax_joint.set_ylim(0.3, 0.9)
g.ax_marg_x.tick_params(labelleft=True)
g.ax_marg_x.grid(True, axis="y", ls=":")
g.ax_marg_x.yaxis.label.set_visible(True)
g.ax_joint.set_xlim(0, 6000)
g.ax_marg_y.remove()
g.savefig(
    "files/images/mimiciv_icd9/f1_micro_vs_num_words.png",
    format="png",
    dpi=400,
    bbox_inches="tight",
)

metric_name = "f1_micro"
for model_name, df in best_run_results.groupby("model"):
    print(f"{model_name}")
    original_length = len(df)
    df = df[df["num_words"] < 4000]
    df = df[df["num_words"] > 1000]
    print(f"Removed {len(df)/original_length} documents")
    print(
        f"Pearson Correlation: {df[['num_words', metric_name]].corr()[metric_name]['num_words']}"
    )
    print(
        f"Spearman Correlation: {df[['num_words', metric_name]].corr(method='spearman')[metric_name]['num_words']}"
    )
