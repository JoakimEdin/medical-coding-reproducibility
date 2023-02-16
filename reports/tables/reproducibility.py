from collections import defaultdict

import numpy as np
import pandas as pd

import wandb
from reports.wandb_utils import get_runs
from src.settings import PROJECT

api = wandb.Api()

SPLIT = "test"
TARGET = "all"
METRICS_50 = ["auc_micro", "auc_macro", "f1_micro", "f1_macro", "precision@5"]
METRICS_FULL = [
    "auc_micro",
    "auc_macro",
    "f1_micro_mullenbach",
    "f1_macro_mullenbach",
    "f1_macro",
    "precision@8_mullenbach",
    "precision@15_mullenbach",
]
MODEL_DICT = {
    "CAML": "CAML \cite{mullenbachExplainablePredictionMedical2018a}",
    "VanillaConv": "CNN \cite{mullenbachExplainablePredictionMedical2018a}",
    "VanillaRNN": "Bi-GRU \cite{mullenbachExplainablePredictionMedical2018a}",
    "LAAT": "LAAT \cite{vuLabelAttentionModel2020}",
    "MultiResCNN": "MultiResCNN \cite{liICDCodingClinical2020}",
    "PLMICD": "PLM-ICD \cite{huangPLMICDAutomaticICD2022}",
}
columns = []

for metric in METRICS_FULL:
    if "_mullenbach" in metric:
        metric = metric.replace("_mullenbach", "")
    elif "f1_macro" in metric:
        metric += "*"
    if "@" in metric:
        level0, level1 = metric.split("@")
        level0 = level0[0].upper()
        level0 += "@k"

    elif "_" in metric:
        level0, level1 = metric.split("_")
        level0 = level0.upper()
        level1 = level1.title()
    else:
        level0 = metric
        level1 = ""
        level0 = level0.upper()
    columns.append(("MIMIC-III full", level0, level1))

for metric in METRICS_50:
    if "@" in metric:
        level0, level1 = metric.split("@")
        level0 = level0[0].upper()
        level0 += "@k"

    elif "_" in metric:
        level0, level1 = metric.split("_")
        level0 = level0.upper()
        level1 = level1.title()
    else:
        level0 = metric
        level1 = ""
        level0 = level0.upper()
    columns.append(("MIMIC-III 50", level0, level1))
columns = pd.MultiIndex.from_tuples(columns)

df = pd.DataFrame(
    np.zeros((len(MODEL_DICT), len(METRICS_FULL) + len(METRICS_50))), columns=columns
)
df.index = list(MODEL_DICT.values())
# print(df.to_latex(escape=False, float_format="%.3f", column_format="l" + "c" * 7, caption="Results on the test set for the three baselines. The best results are highlighted in bold.", label="tab:baselines"))

sweep_id = "mgv5uphg"
dataset = "mimiciii_50"
mimic_50_runs = get_runs(
    PROJECT, {"Sweep": sweep_id, "config.data.data_filename": {"$regex": f"{dataset}*"}}
)
mimic_50_runs += [api.run(f"{PROJECT}/1ooqo0me")]

sweep_id = "mgv5uphg|dvovucvj"
dataset = "mimiciii_full"
mimic_full_runs = get_runs(
    PROJECT,
    {
        "Sweep": {"$regex": sweep_id},
        "config.data.data_filename": {"$regex": f"{dataset}*"},
    },
)
for run in mimic_full_runs:
    model_class_name = run.config["model"]["name"]

    if model_class_name not in MODEL_DICT:
        continue

    model_name = MODEL_DICT[model_class_name]

    results = []
    for idx, metric in enumerate(METRICS_FULL):
        results.append(run.summary[SPLIT][TARGET][metric] * 100)
    df.loc[model_name, "MIMIC-III full"] = results

for run in mimic_50_runs:
    model_class_name = run.config["model"]["name"]

    if model_class_name not in MODEL_DICT:
        continue

    model_name = MODEL_DICT[model_class_name]

    results = []
    for idx, metric in enumerate(METRICS_50):
        results.append(run.summary[SPLIT][TARGET][metric] * 100)
    df.loc[model_name, "MIMIC-III 50"] = results

df.sort_values(by=("MIMIC-III full", "F1", "Micro"), ascending=True, inplace=True)
styler = df.style.format("{:.1f}")

# styler.highlight_max(axis=0, props="font-weight:bold;")

print(
    styler.to_latex(
        hrules=True,
        position_float="centering",
        environment="table*",
        multicol_align="c",
        column_format="l|" + "c" * len(METRICS_FULL) + "|" + "c" * len(METRICS_50),
        caption="Reproduced results on MIMIC-III v1.4 using the Mullenbach et al. preprocessing pipeline and splits \cite{mullenbachExplainablePredictionMedical2018a}. Each model was reproduced using the hyperparameters presented in their respective papers. The shown results are on the Mullenbach test set.",
        label="tab:reproduced_results",
        convert_css=True,
    )
)
