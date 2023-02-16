import logging
import random
from pathlib import Path

import pandas as pd

from src.settings import (
    DATA_DIRECTORY_MIMICIII_CLEAN,
    ID_COLUMN,
    SUBJECT_ID_COLUMN,
    TARGET_COLUMN,
)
from src.utils.stratify_function import (
    iterative_stratification,
    kl_divergence,
    labels_not_in_split,
)

TEST_SIZE = 0.15  # Test split ratios
VAL_SIZE = 0.1  # Val split ratio
STEP_SIZE = 0.2  # Step size for the iterative stratification

random.seed(10)

logging.basicConfig(level=logging.INFO)
output_dir = Path(DATA_DIRECTORY_MIMICIII_CLEAN)

mimic_clean = pd.read_feather(output_dir / "mimiciii_clean.feather")
mimic_clean[TARGET_COLUMN] = mimic_clean[TARGET_COLUMN].apply(lambda x: list(x))

# Generate splits

splits = mimic_clean[[SUBJECT_ID_COLUMN, ID_COLUMN]]
subject_series = mimic_clean.groupby(SUBJECT_ID_COLUMN)[TARGET_COLUMN].sum()
subject_ids = subject_series.index.to_list()
codes = subject_series.to_list()
subject_ids_train, subject_ids_test = iterative_stratification(
    subject_ids, codes, [1 - TEST_SIZE, TEST_SIZE]
)
codes_train = [codes[subject_ids.index(subject_id)] for subject_id in subject_ids_train]
val_size = VAL_SIZE / (1 - TEST_SIZE)
subject_ids_train, subject_ids_val = iterative_stratification(
    subject_ids_train, codes_train, [1 - val_size, val_size]
)

codes_train = [codes[subject_ids.index(subject_id)] for subject_id in subject_ids_train]
codes_val = [codes[subject_ids.index(subject_id)] for subject_id in subject_ids_val]
codes_test = [codes[subject_ids.index(subject_id)] for subject_id in subject_ids_test]

splits.loc[splits[SUBJECT_ID_COLUMN].isin(subject_ids_train), "split"] = "train"
splits.loc[splits[SUBJECT_ID_COLUMN].isin(subject_ids_val), "split"] = "val"
splits.loc[splits[SUBJECT_ID_COLUMN].isin(subject_ids_test), "split"] = "test"

logging.info("------------- Splits Statistics -------------")
logging.info(
    f"Labels missing in the test set: {labels_not_in_split(codes, codes_test)}"
)
logging.info(
    f"Labels missing in the val set: {labels_not_in_split(codes, codes_val)} %"
)
logging.info(
    f"Labels missing in the train set: {labels_not_in_split(codes, codes_train)} %"
)
logging.info(f"Test: KL divergence: {kl_divergence(codes, codes_test)}")
logging.info(f"Val: KL divergence: {kl_divergence(codes, codes_val)}")
logging.info(f"Train: KL divergence: {kl_divergence(codes, codes_train)}")
logging.info(f"Test Size: {len(codes_test) / len(codes)}")
logging.info(f"Val Size: {len(codes_val) / len(codes)}")
logging.info(f"Train Size: {len(codes_train) / len(codes)}")

splits = splits[[ID_COLUMN, "split"]].reset_index(drop=True)
splits.to_feather(output_dir / "mimiciii_clean_splits.feather")
logging.info(
    "Splits generated and saved. Now making subsplits used to analyse the performance of the models when trained on less data."
)


logging.info("\n------------- Subsplits Statistics -------------")
mimic_clean = mimic_clean.merge(splits, on=ID_COLUMN)
train_split = mimic_clean[mimic_clean["split"] == "train"]
val_split = mimic_clean[mimic_clean["split"] == "val"]
test_split = mimic_clean[mimic_clean["split"] == "test"]

train_split = train_split.reset_index(drop=True)
train_indices = train_split.index.to_list()
train_codes = train_split[TARGET_COLUMN].to_list()

splits = []
ratio = STEP_SIZE
train_indices_remaining = train_indices.copy()
train_codes_remaining = train_codes.copy()


while ratio < 1:
    train_indices_remaining, train_indices_split = iterative_stratification(
        train_indices_remaining,
        train_codes_remaining,
        [
            1 - len(train_indices) * STEP_SIZE / len(train_indices_remaining),
            len(train_indices) * STEP_SIZE / len(train_indices_remaining),
        ],
    )
    train_codes_remaining = [train_codes[index] for index in train_indices_remaining]
    splits += train_indices_split
    logging.info(
        f"Labels missing in the subsplit: {labels_not_in_split(train_codes, train_codes_remaining)} %"
    )
    train_subsplit = train_split.iloc[splits]
    subsplit = pd.concat([train_subsplit, val_split, test_split])
    subsplit = subsplit[[ID_COLUMN, "split"]].reset_index(drop=True)
    subsplit.to_feather(output_dir / f"mimiciii_clean_subsplit_{ratio:.1f}.feather")
    logging.info(f"Subsplit {ratio:.1f} saved with training size {len(train_subsplit)}")
    ratio += STEP_SIZE


logging.info("\nMimic III splits succesfully created!")
