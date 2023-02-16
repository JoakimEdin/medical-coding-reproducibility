import numpy as np
import pandas as pd

from src.settings import HUE_ORDER, MODEL_NAMES, PROJECT
from reports.wandb_utils import get_best_runs, get_run, get_runs

sweep_ids = {
    "Best": "xxlsqdh3",
    "Max Length = 2500": "2at45wg7",
    "Decision Boundary Tuning = False": "y1h4j74r",
    "Text Preprocessing = Mullenbach": "62kfhbem",
    "Hyperparameters = Original": "epfp6efe",
}

runs_dict = {}
for name, sweep_id in sweep_ids.items():
    if name == "Best":
        runs_dict[name] = list(
            get_best_runs(PROJECT, "f1_micro", {"Sweep": sweep_id}).values()
        )
        continue

    runs_dict[name] = get_runs(PROJECT, {"Sweep": sweep_id})
    if name == "Hyperparameters = Original":
        runs_dict[name] += [get_run(PROJECT, "tkqdmmqd")]

table = pd.DataFrame(np.zeros((5, 7)), columns=[""] + HUE_ORDER)

for row_idx, (name, runs) in enumerate(runs_dict.items()):
    table.iloc[row_idx, 0] = name
    for run in runs:
        model_name = run.config["model"]["name"]
        if model_name in MODEL_NAMES:
            model_name = MODEL_NAMES[model_name]
        f1 = run.summary["test"]["all"]["f1_micro"]
        table.loc[row_idx, model_name] = f"{f1*100:.1f}"

table = table.set_index("", drop=True)

print(
    table.style.to_latex(
        hrules=True,
        position_float="centering",
        environment="table*",
        column_format="l|cccccc",
        caption="Ablation study on MIMIC-III clean. The numbers are the F1 scores on the test set.",
        label="tab:ablation study",
        convert_css=True,
    )
)
