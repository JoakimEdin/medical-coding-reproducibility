from collections import defaultdict

import pandas as pd
import seaborn as sns

from src.settings import HUE_ORDER, MODEL_NAMES, PALETTE, PROJECT
from reports.wandb_utils import get_best_runs, get_runs

sns.set_theme("paper", style="whitegrid", palette="muted", font_scale=1.2)

metric = "f1_macro"
sweep_id = "rcy0wu5x"
sizes = [25, 50, 75, 100, 125]

runs = get_runs(
    PROJECT,
    {
        "Sweep": sweep_id,
        "State": "finished",
    },
)
runs_dict = defaultdict(dict)
for run in runs:
    subset = run.config["data.split_filename"]
    size = subset.split("_")[-1].split(".")[0].replace("k", "")
    if size == "10":
        continue
    model_name = run.config["model"]["name"]
    print(size, model_name)
    runs_dict[int(size)][model_name] = run


SWEEP_ID = "5ykjry46"
runs_dict[150] = get_best_runs(
    PROJECT, "f1_micro", {"Sweep": {"$regex": f"{SWEEP_ID}"}, "State": "finished"}
)

tuple_list = []
for size, runs in runs_dict.items():
    print(f"Size: {size}")
    for model_name, run in runs.items():
        if model_name in MODEL_NAMES:
            model_name = MODEL_NAMES[model_name]
        tuple_list.append(
            (
                int(run.config["data_info"]["num_train_examples"]),
                model_name,
                run.summary["test"]["all"][metric],
            )
        )
        print(
            (
                int(run.config["data_info"]["num_train_examples"]),
                model_name,
                run.summary["test"]["all"][metric],
            )
        )


ax = sns.lineplot(
    x="size",
    y=metric,
    hue="Model",
    data=pd.DataFrame(tuple_list, columns=["size", "Model", metric]),
    palette=PALETTE,
    hue_order=HUE_ORDER,
    linewidth=1.5,
)
ax.set_xlabel(
    "Training Examples",
)
if metric == "f1_micro":
    ax.set_ylabel("F1 Micro")
    ax.set_ylim(0.4, 0.8)
elif metric == "f1_macro":
    ax.set_ylabel("F1 Macro")
    ax.set_ylim(0.05, 0.45)

sns.move_legend(ax, "upper left")
ax.get_figure().savefig(
    f"files/images/mimiciv_icd9/train_size_{metric}.png",
    format="png",
    dpi=400,
    bbox_inches="tight",
)
