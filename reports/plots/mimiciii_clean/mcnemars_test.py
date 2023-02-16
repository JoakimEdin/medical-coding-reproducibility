import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from matplotlib.colors import ListedColormap
from statsmodels.stats.contingency_tables import mcnemar

from reports.utils import get_db, get_results, get_target_counts, load_split_data
from reports.wandb_utils import get_best_runs
from src.settings import MODEL_NAMES, PROJECT, TARGET_COLUMN

SWEEP_ID = "xxlsqdh3"
DATASET = "mimiciii_clean"

sns.set_theme("paper", style="whitegrid", palette="muted")


def calculate_contingency_table(
    errors1: torch.Tensor, errors2: torch.Tensor
) -> np.array:
    contingency_table = torch.zeros((2, 2))
    contingency_table[0, 0] = torch.sum(
        torch.logical_not(torch.logical_and(errors1, errors2))
    )  # both are correct
    contingency_table[0, 1] = torch.sum(
        torch.logical_and(torch.logical_not(errors1), errors2)
    )  # model 1 is correct, model 2 is incorrect
    contingency_table[1, 0] = torch.sum(
        torch.logical_and(errors1, torch.logical_not(errors2))
    )  # model 1 is incorrect, model 2 is correct
    contingency_table[1, 1] = torch.sum(
        torch.logical_and(errors1, errors2)
    )  # both are incorrect
    return contingency_table.numpy().astype(int)


def calculate_mcnemar(errors1: torch.Tensor, errors2: torch.Tensor) -> float:
    contingency_table = calculate_contingency_table(errors1, errors2)
    test = mcnemar(contingency_table, exact=False, correction=True)
    return test


best_runs = get_best_runs(PROJECT, "f1_micro", {"Sweep": SWEEP_ID})
errors = {}
for model_name, run in best_runs.items():
    db = get_db(run.id)
    logits, targets, ids, target2index = get_results(run.id, "test")
    predictions = (logits > db).long()
    fp = predictions * (1 - targets)
    fn = (1 - predictions) * targets
    errors[model_name] = torch.logical_or(fp, fn)

model_names = list(errors.keys())
num_models = len(model_names)
mcnemar_matrix = np.zeros((num_models, num_models))

for i in range(num_models):
    for j in range(i + 1, num_models):
        errors1 = errors[model_names[i]]
        errors2 = errors[model_names[j]]
        mcnemar_matrix[j, i] = calculate_mcnemar(errors1, errors2).pvalue

formatted_model_names = [
    MODEL_NAMES.get(model_name, model_name) for model_name in model_names
]
mcnemar_matrix = pd.DataFrame(
    mcnemar_matrix, columns=formatted_model_names, index=formatted_model_names
)
mask = np.triu(np.ones_like(mcnemar_matrix, dtype=bool))


g = sns.heatmap(
    mcnemar_matrix,
    annot=True,
    mask=mask,
    cbar=False,
    cmap=ListedColormap(["grey"]),
    linewidths=0.5,
)
g.get_figure().savefig(
    "files/images/mimiciii/mcnemar_all.png",
    dpi=1000,
    bbox_inches="tight",
)
plt.clf()

data = load_split_data(dataset="mimiciii_clean")
train = data[data["split"] == "train"]
target_counts_train = get_target_counts(train[TARGET_COLUMN].tolist())
rare_codes = [target for target, count in target_counts_train.items() if count < 100]
rare_code_indices = [target2index[target] for target in rare_codes]
common_codes = [
    target for target, count in target_counts_train.items() if count >= 1000
]
common_codes_indices = [target2index[target] for target in common_codes]

# rare codes

model_names = list(errors.keys())
num_models = len(model_names)
mcnemar_matrix = np.zeros((num_models, num_models))

for i in range(num_models):
    for j in range(i + 1, num_models):
        errors1 = errors[model_names[i]]
        errors2 = errors[model_names[j]]
        errors1 = torch.index_select(errors1, 1, torch.tensor(rare_code_indices))
        errors2 = torch.index_select(errors2, 1, torch.tensor(rare_code_indices))

        mcnemar_matrix[j, i] = calculate_mcnemar(errors1, errors2).pvalue

mcnemar_matrix = pd.DataFrame(
    mcnemar_matrix, columns=formatted_model_names, index=formatted_model_names
)
mask = np.triu(np.ones_like(mcnemar_matrix, dtype=bool))


g = sns.heatmap(
    mcnemar_matrix,
    annot=True,
    mask=mask,
    cbar=False,
    cmap=ListedColormap(["grey"]),
    linewidths=0.5,
)
g.get_figure().savefig(
    "files/images/mimiciii/mcnemar_rare.png",
    dpi=1000,
    bbox_inches="tight",
)
plt.clf()

# Common codes

model_names = list(errors.keys())
num_models = len(model_names)
mcnemar_matrix = np.zeros((num_models, num_models))

for i in range(num_models):
    for j in range(i + 1, num_models):
        errors1 = errors[model_names[i]]
        errors2 = errors[model_names[j]]
        errors1 = torch.index_select(errors1, 1, torch.tensor(common_codes_indices))
        errors2 = torch.index_select(errors2, 1, torch.tensor(common_codes_indices))

        mcnemar_matrix[j, i] = calculate_mcnemar(errors1, errors2).pvalue

mcnemar_matrix = pd.DataFrame(
    mcnemar_matrix, columns=formatted_model_names, index=formatted_model_names
)
mask = np.triu(np.ones_like(mcnemar_matrix, dtype=bool))


g = sns.heatmap(
    mcnemar_matrix,
    annot=True,
    mask=mask,
    cbar=False,
    cmap=ListedColormap(["grey"]),
    linewidths=0.5,
)
g.get_figure().savefig(
    "files/images/mimiciii/mcnemar_common.png",
    dpi=1000,
    bbox_inches="tight",
)
