from collections import defaultdict

import numpy as np
import pandas as pd

import wandb
from reports.wandb_utils import get_runs
from src.settings import PROJECT

api = wandb.Api()


SPLIT = "test"
TARGET = "all"
METRICS = [
    "auc_micro",
    "auc_macro",
    "f1_micro",
    "f1_macro",
    "exact_match_ratio",
    "precision@8",
    "precision@15",
    "precision@recall",
    "map",
]
MODEL_DICT = {
    "CAML": "CAML \cite{mullenbachExplainablePredictionMedical2018a}",
    "VanillaConv": "CNN \cite{mullenbachExplainablePredictionMedical2018a}",
    "VanillaRNN": "Bi-GRU \cite{mullenbachExplainablePredictionMedical2018a}",
    "LAAT": "LAAT \cite{vuLabelAttentionModel2020}",
    "MultiResCNN": "MultiResCNN \cite{liICDCodingClinical2020}",
    # "EffectiveCAN": "EffectiveCAN \cite{liuEffectiveConvolutionalAttention2021}",
    "PLMICD": "PLM-ICD \cite{huangPLMICDAutomaticICD2022}",
}
columns = []

for metric in METRICS:
    metric = metric.replace("exact_match_ratio", "EMR")
    if "precision@recall" in metric:
        level0 = "R-precision"
        level1 = ""
    elif "@" in metric:
        level0, level1 = metric.split("@")
        level0 = level0.title()
        level0 += "@k"

    elif "_" in metric:
        level0, level1 = metric.split("_")
        if level0 in {"recall", "precision"}:
            level0 = level0.title()
        else:
            level0 = level0.upper()
        level1 = level1.title()
    else:
        level0 = metric
        level1 = ""
        level0 = level0.upper()
    columns.append((level0, level1))

columns = pd.MultiIndex.from_tuples(columns)

df = pd.DataFrame(np.zeros((len(MODEL_DICT), len(METRICS))), columns=columns)
df.index = list(MODEL_DICT.values())
# print(df.to_latex(escape=False, float_format="%.3f", column_format="l" + "c" * 7, caption="Results on the test set for the three baselines. The best results are highlighted in bold.", label="tab:baselines"))

sweep_id = "ub6ydxq3"
runs = get_runs(PROJECT, {"Sweep": sweep_id, "State": "finished"})

results = defaultdict(lambda: defaultdict(list))
for run in runs:
    model_class_name = run.config["model"]["name"]
    if model_class_name not in MODEL_DICT:
        continue
    model_name = MODEL_DICT[model_class_name]

    for idx, metric in enumerate(METRICS):
        results[model_name][metric].append(run.summary[SPLIT][TARGET][metric] * 100)

for model_name, results_dict in results.items():
    model_metrics = []
    for _, results in results_dict.items():
        model_metrics.append(f"{np.mean(results):.1f}$\pm${np.std(results):.1f}")
    df.loc[model_name] = model_metrics


df.sort_values(
    by=("F1", "Micro"),
    ascending=True,
    inplace=True,
    key=lambda x: x.str.split("$").str[0].astype(float),
)
styler = df.style


def highlight_max(s):
    f = s.str.split("$").str[0].astype(float)
    is_max = f == f.max()
    return ["font-weight:bold" if v else "" for v in is_max]


styler.apply(highlight_max)

print(
    styler.to_latex(
        hrules=True,
        position_float="centering",
        environment="table*",
        multicol_align="c",
        column_format="l|" + "c" * len(METRICS),
        caption="MIMIC-IV ICD-10 results. We performed a McNemar's test with Bonferroni correction and found them all to be significantly different ($p < 0.001$)",
        label="tab:mimiciv_icd10_results",
        convert_css=True,
    )
)
