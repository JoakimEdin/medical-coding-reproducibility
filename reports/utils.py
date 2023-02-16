import os
from enum import Enum
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.settings import (
    DATA_DIRECTORY_MIMICIII_CLEAN,
    DATA_DIRECTORY_MIMICIV_ICD9,
    DATA_DIRECTORY_MIMICIV_ICD10,
    DATA_DIRECTORY_MIMICIII_50,
    DATA_DIRECTORY_MIMICIII_FULL,
    DOWNLOAD_DIRECTORY_MIMICIII,
    EXPERIMENT_DIR,
    ID_COLUMN,
    TARGET_COLUMN,
)

ADMISSION_PATH = Path(DOWNLOAD_DIRECTORY_MIMICIII) / "ADMISSIONS.csv.gz"
PATIENTS_PATH = Path(DOWNLOAD_DIRECTORY_MIMICIII) / "PATIENTS.csv.gz"

DATASETS = {
    "mimiciii_clean": (
        os.path.join(DATA_DIRECTORY_MIMICIII_CLEAN, "mimiciii_clean.feather"),
        os.path.join(DATA_DIRECTORY_MIMICIII_CLEAN, "mimiciii_clean_splits.feather"),
    ),
    "mimiciii_full": (
        os.path.join(DATA_DIRECTORY_MIMICIII_FULL, "mimiciii_full.feather"),
        os.path.join(DATA_DIRECTORY_MIMICIII_FULL, "mimiciii_full_splits.feather"),
    ),
    "mimiciii_50": (
        os.path.join(DATA_DIRECTORY_MIMICIII_50, "mimiciii_full.feather"),
        os.path.join(DATA_DIRECTORY_MIMICIII_50, "mimiciii_50_splits.feather"),
    ),
    "mimiciv_icd9": (
        os.path.join(DATA_DIRECTORY_MIMICIV_ICD9, "mimiciv_icd9.feather"),
        os.path.join(DATA_DIRECTORY_MIMICIV_ICD9, "mimiciv_icd9_split.feather"),
    ),
    "mimiciv_icd10": (
        os.path.join(DATA_DIRECTORY_MIMICIV_ICD10, "mimiciv_icd10.feather"),
        os.path.join(DATA_DIRECTORY_MIMICIV_ICD10, "mimiciv_icd10_split.feather"),
    ),
}


def one_hot(targets: list[list[str]], target2index: dict[str, int]) -> torch.Tensor:
    number_of_classes = len(target2index)
    output_tensor = torch.zeros((len(targets), number_of_classes))
    for idx, case in enumerate(targets):
        for target in case:
            if target in target2index:
                output_tensor[idx, target2index[target]] = 1
    return output_tensor.long()


def load_results(
    run_id: str, split: str = "val", experiment_dir: Path = EXPERIMENT_DIR
) -> tuple[np.array, np.array, np.array]:
    results = pd.read_feather(experiment_dir / run_id / f"predictions_{split}.feather")
    targets = results[[TARGET_COLUMN]].values.squeeze()
    ids = results[ID_COLUMN].values
    logits_columns = results.drop(columns=[ID_COLUMN, TARGET_COLUMN])
    logits = logits_columns.values
    unique_targets = list(logits_columns.columns.unique())
    return logits, targets, ids, unique_targets


def parse_results(
    logits, targets, ids, unique_targets
) -> tuple[dict[str, torch.Tensor], str, int]:
    target2index = {target: idx for idx, target in enumerate(unique_targets)}
    # Mapping from target to index and vice versa
    targets = one_hot(targets, target2index)  # one_hot encoding of targets
    logits = torch.tensor(logits)
    return logits, targets, ids, target2index


def get_results(
    run_id: str, split: str = "val", experiment_dir: Path = EXPERIMENT_DIR
) -> tuple[torch.Tensor, torch.Tensor, np.array]:
    logits, targets, ids, unique_targets = load_results(run_id, split, experiment_dir)
    logits, targets, ids, target2index = parse_results(
        logits, targets, ids, unique_targets
    )
    return logits, targets, ids, target2index


def get_target_counts(targets: list[list[str]]) -> dict[str, int]:
    counts = {}
    for case in targets:
        for target in case:
            if target in counts:
                counts[target] += 1
            else:
                counts[target] = 1
    return counts


def f1_score_db_tuning(logits, targets, average="micro", type="single"):
    if average not in ["micro", "macro"]:
        raise ValueError("Average must be either 'micro' or 'macro'")
    dbs = torch.linspace(0, 1, 100)
    tp = torch.zeros((len(dbs), targets.shape[1]))
    fp = torch.zeros((len(dbs), targets.shape[1]))
    fn = torch.zeros((len(dbs), targets.shape[1]))
    for idx, db in enumerate(dbs):
        predictions = (logits > db).long()
        tp[idx] = torch.sum((predictions) * (targets), dim=0)
        fp[idx] = torch.sum(predictions * (1 - targets), dim=0)
        fn[idx] = torch.sum((1 - predictions) * targets, dim=0)
    if average == "micro":
        f1_scores = tp.sum(1) / (tp.sum(1) + 0.5 * (fp.sum(1) + fn.sum(1)) + 1e-10)
    else:
        f1_scores = torch.mean(tp / (tp + 0.5 * (fp + fn) + 1e-10), dim=1)
    if type == "single":
        best_f1 = f1_scores.max()
        best_db = dbs[f1_scores.argmax()]
        print(f"Best F1: {best_f1:.4f} at DB: {best_db:.4f}")
        return best_f1, best_db
    if type == "per_class":
        best_f1 = f1_scores.max(1)
        best_db = dbs[f1_scores.argmax(0)]
        print(f"Best F1: {best_f1} at DB: {best_db}")
        return best_f1, best_db


def fpr_tuning(logits, targets, average="micro"):
    dbs = torch.linspace(1, 0, 100)
    tp = torch.zeros((len(dbs), targets.shape[1]))
    fp = torch.zeros((len(dbs), targets.shape[1]))
    fn = torch.zeros((len(dbs), targets.shape[1]))
    tn = torch.zeros((len(dbs), targets.shape[1]))
    for idx, db in enumerate(dbs):
        predictions = (logits > db).long()
        tp[idx] = torch.sum((predictions) * (targets), dim=0)
        fp[idx] = torch.sum(predictions * (1 - targets), dim=0)
        fn[idx] = torch.sum((1 - predictions) * targets, dim=0)
        tn[idx] = torch.sum((1 - predictions) * (1 - targets), dim=0)
    fprs = fp / (fp + tn + 1e-10)
    tprs = tp / (tp + fn + 1e-10)
    tprs[fprs > 0.0015] = -1
    fprs[fprs > 0.0015] = -1
    best_tpr, indices = torch.max(tprs, 0)
    best_db = dbs[indices]


def precision_tuning(logits, targets, average="micro"):
    dbs = torch.linspace(1, 0, 100)
    tp = torch.zeros((len(dbs), targets.shape[1]))
    fp = torch.zeros((len(dbs), targets.shape[1]))
    fn = torch.zeros((len(dbs), targets.shape[1]))
    tn = torch.zeros((len(dbs), targets.shape[1]))
    for idx, db in enumerate(dbs):
        predictions = (logits > db).long()
        tp[idx] = torch.sum((predictions) * (targets), dim=0)
        fp[idx] = torch.sum(predictions * (1 - targets), dim=0)
        fn[idx] = torch.sum((1 - predictions) * targets, dim=0)
        tn[idx] = torch.sum((1 - predictions) * (1 - targets), dim=0)
    precision = tp / (fp + tp + 1e-10)
    f1 = tp / (tp + 0.5 * (fp + fn) + 1e-10)
    f1[precision < 0.2] = -1
    best_f1, indices = torch.max(f1, 0)
    best_db = dbs[indices]
    best_db[best_db == 1] = 0.5
    print(best_f1, best_db)
    return best_db


def micro_f1(pred: torch.Tensor, targets: torch.Tensor) -> float:
    tp = torch.sum((pred) * (targets), dim=0)
    fp = torch.sum(pred * (1 - targets), dim=0)
    fn = torch.sum((1 - pred) * targets, dim=0)
    f1 = tp / (tp + 0.5 * (fp + fn) + 1e-10)
    return torch.mean(f1)


def get_db(run_id: str, experiment_dir: Path = EXPERIMENT_DIR) -> torch.Tensor:
    logits, targets, _, _ = get_results(run_id, "val", experiment_dir)
    _, db = f1_score_db_tuning(logits, targets, average="micro")
    return db


def load_data(
    data_dir: Path = DATA_DIRECTORY_MIMICIII_CLEAN,
    data_file_name: str = "mimiciii_clean.feather",
) -> pd.DataFrame:
    return pd.read_feather(os.path.join(data_dir, data_file_name))


def load_splits(
    data_dir: Path = DATA_DIRECTORY_MIMICIII_CLEAN,
    split_file_name: str = "mimiciii_clean_splits.feather",
) -> pd.DataFrame:
    return pd.read_feather(os.path.join(data_dir, split_file_name))


def load_split_data(
    data_dir: Path = DATA_DIRECTORY_MIMICIII_CLEAN,
    dataset: str = "mimiciii_clean",
) -> pd.DataFrame:
    data_file_name, split_file_name = DATASETS[dataset]
    data = load_data(data_dir, data_file_name)
    splits = load_splits(data_dir, split_file_name)
    return data.merge(splits[[ID_COLUMN, "split"]], on=ID_COLUMN, how="left")


def create_id2age_dict() -> dict[int, int]:
    admissions = pd.read_csv(ADMISSION_PATH, compression="gzip")
    patients = pd.read_csv(PATIENTS_PATH, compression="gzip")
    admissions = admissions.merge(
        patients[["SUBJECT_ID", "DOB"]], on="SUBJECT_ID", how="inner"
    )
    admissions = admissions.drop_duplicates(subset="SUBJECT_ID")
    admission_dates = (
        admissions["ADMITTIME"]
        .str.split(" ", expand=True)[0]
        .str.split("-", expand=True)
    )
    birth_dates = (
        admissions["DOB"].str.split(" ", expand=True)[0].str.split("-", expand=True)
    )
    admissions["age"] = (
        admission_dates[0].astype(int)
        - birth_dates[0].astype(int)
        + (admission_dates[1].astype(int) - birth_dates[1].astype(int)) / 12
        + (admission_dates[2].astype(int) - birth_dates[2].astype(int)) / 365
    )
    # All ages above 89 were shifted to hide the identities
    admissions.loc[admissions["age"] > 90, "age"] = 90.0
    return dict(zip(admissions["SUBJECT_ID"], admissions["age"]))
