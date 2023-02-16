import pandas as pd
import seaborn as sns

from src.settings import HUE_ORDER, MODEL_NAMES, PALETTE, PROJECT
from reports.wandb_utils import get_best_runs

sns.set_theme("paper", style="whitegrid", palette="muted")

metric = "f1_micro"
sweep_id = "sxoivxgr"
sizes = [0.2, 0.4, 0.6, 0.8]

runs_dict = {}
for size in sizes:
    dataset_name = f"mimiciii_clean_subsplit_{size}.feather"
    runs_dict[size] = get_best_runs(
        PROJECT,
        "f1_micro",
        {
            "Sweep": sweep_id,
            "config.data.split_filename": dataset_name,
            "State": "finished",
        },
    )

SWEEP_ID = "83jehgfy"
runs_dict[1.0] = get_best_runs(
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
    y="f1_micro",
    hue="model",
    data=pd.DataFrame(tuple_list, columns=["size", "model", "f1_micro"]),
    palette=PALETTE,
    hue_order=HUE_ORDER,
    linewidth=1.5,
)
ax.set_xlabel(
    "Training Examples",
)
ax.set_ylabel("F1 Micro")


sns.move_legend(ax, "lower right")
ax.get_figure().savefig(
    "files/images/train_size_f1.png",
    format="png",
    dpi=400,
    bbox_inches="tight",
)
