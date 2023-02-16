import logging
import os
import random
import shutil
from collections import Counter
from functools import partial
from pathlib import Path

import pandas as pd

from prepare_data.utils import (
    TextPreprocessor,
    format_code_dataframe,
    get_mimiciii_notes,
    merge_code_dataframes,
    merge_reports_addendum,
    preprocess_documents,
    reformat_icd9,
    remove_duplicated_codes,
    replace_nans_with_empty_lists,
)
from src.settings import (
    DATA_DIRECTORY_MIMICIII_CLEAN,
    DOWNLOAD_DIRECTORY_MIMICIII,
    ID_COLUMN,
    SUBJECT_ID_COLUMN,
    TARGET_COLUMN,
    TEXT_COLUMN,
)

CODE_SYSTEMS = [
    ("ICD9-DIAG", "DIAGNOSES_ICD.csv.gz", "ICD9_CODE", "icd9_diag"),
    ("ICD9-PROC", "PROCEDURES_ICD.csv.gz", "ICD9_CODE", "icd9_proc"),
]
MIN_TARGET_COUNT = 10  # Minimum number of times a code must appear to be included
preprocessor = TextPreprocessor(
    lower=True,
    remove_special_characters_mullenbach=True,
    remove_special_characters=False,
    remove_digits=True,
    remove_accents=False,
    remove_brackets=False,
    convert_danish_characters=False,
)

random.seed(10)


# The dataset requires a Licence in physionet. Once it is obtained, download the dataset with the following command in the terminal:
# wget -r -N -c -np --user <your_physionet_user_name> --ask-password https://physionet.org/files/mimiciii/1.4/
# Change the path of DOWNLOAD_DIRECTORY to the path where you downloaded mimiciii

logging.basicConfig(level=logging.INFO)

download_dir = Path(DOWNLOAD_DIRECTORY_MIMICIII)
output_dir = Path(DATA_DIRECTORY_MIMICIII_CLEAN)
output_dir.mkdir(parents=True, exist_ok=True)


def get_duplicated_icd9_proc_codes() -> set:
    """Get the duplicated ICD9-PROC codes. The codes are duplicated because they are saved as integers,
    removing any zeros at the beginning of the codes. These codes will not be included in the dataset.

    Returns:
        set: The duplicated ICD9-PROC codes
    """
    icd9_proc_codes = pd.read_csv(
        download_dir / "D_ICD_PROCEDURES.csv.gz",
        compression="gzip",
        dtype={"ICD9_CODE": str},
    )
    return set(
        icd9_proc_codes[icd9_proc_codes["ICD9_CODE"].astype(str).duplicated()][
            "ICD9_CODE"
        ]
    )


def prepare_discharge_summaries(mimic_notes: pd.DataFrame) -> pd.DataFrame:
    """Format the notes dataframe into the discharge summaries dataframe

    Args:
        mimic_notes (pd.DataFrame): The notes dataframe

    Returns:
        pd.DataFrame: Formatted discharge summaries dataframe
    """
    mimic_notes = mimic_notes.rename(
        columns={
            "HADM_ID": ID_COLUMN,
            "SUBJECT_ID": SUBJECT_ID_COLUMN,
            "TEXT": TEXT_COLUMN,
        }
    )
    logging.info(f"{mimic_notes[ID_COLUMN].nunique()} number of admissions")
    discharge_summaries = merge_reports_addendum(mimic_notes)
    discharge_summaries = discharge_summaries.sort_values(
        [SUBJECT_ID_COLUMN, ID_COLUMN]
    )

    discharge_summaries = discharge_summaries.reset_index(drop=True)
    logging.info(
        f"{discharge_summaries[SUBJECT_ID_COLUMN].nunique()} subjects, {discharge_summaries[ID_COLUMN].nunique()} admissions"
    )
    return discharge_summaries


def filter_codes(df: pd.DataFrame, columns: list[str], min_count: int) -> pd.DataFrame:
    """Filter the codes dataframe to only include codes that appear at least min_count times

    Args:
        df (pd.DataFrame): The codes dataframe
        col (str): The column name of the codes
        min_count (int): The minimum number of times a code must appear

    Returns:
        pd.DataFrame: The filtered codes dataframe
    """
    for col in columns:
        code_counts = Counter([code for codes in df[col] for code in codes])
        codes_to_keep = set(
            code for code, count in code_counts.items() if count >= min_count
        )
        df[col] = df[col].apply(lambda x: [code for code in x if code in codes_to_keep])
    return df


def download_and_preprocess_code_systems(code_systems: list[tuple]) -> pd.DataFrame:
    """Download and preprocess the code systems dataframe

    Args:
        code_systems (List[tuple]): The code systems to download and preprocess

    Returns:
        pd.DataFrame: The preprocessed code systems dataframe"""
    code_dfs = []
    for name, fname, col_in, col_out in code_systems:
        logging.info(f"Loading {name} codes...")
        df = pd.read_csv(
            download_dir / fname, compression="gzip", dtype={"ICD9_CODE": str}
        )
        df = format_code_dataframe(df, col_in, col_out)
        df = remove_duplicated_codes(df, [col_out])
        code_dfs.append(df)

    merged_codes = merge_code_dataframes(code_dfs)
    merged_codes = replace_nans_with_empty_lists(merged_codes)
    merged_codes["icd9_diag"] = merged_codes["icd9_diag"].apply(
        lambda codes: list(map(partial(reformat_icd9, is_diag=True), codes))
    )
    merged_codes["icd9_proc"] = merged_codes["icd9_proc"].apply(
        lambda codes: list(map(partial(reformat_icd9, is_diag=False), codes))
    )
    merged_codes[TARGET_COLUMN] = merged_codes["icd9_proc"] + merged_codes["icd9_diag"]
    return merged_codes


get_duplicated_icd9_proc_codes()
# MIMIC-III full
mimic_notes = get_mimiciii_notes(download_dir)
discharge_summaries = prepare_discharge_summaries(mimic_notes)
merged_codes = download_and_preprocess_code_systems(CODE_SYSTEMS)

full_dataset = discharge_summaries.merge(
    merged_codes, on=[SUBJECT_ID_COLUMN, ID_COLUMN], how="inner"
)
full_dataset = replace_nans_with_empty_lists(full_dataset)
# Remove codes that appear less than 10 times
full_dataset = filter_codes(
    full_dataset, [TARGET_COLUMN, "icd9_proc", "icd9_diag"], min_count=MIN_TARGET_COUNT
)
# Remove admissions with no codes
full_dataset = full_dataset[full_dataset[TARGET_COLUMN].apply(len) > 0]

full_dataset = preprocess_documents(df=full_dataset, preprocessor=preprocessor)

logging.info(f"{full_dataset[ID_COLUMN].nunique()} number of admissions")
full_dataset = full_dataset.reset_index(drop=True)
full_dataset.to_feather(output_dir / "mimiciii_clean.feather")
logging.info(f"Saved full dataset to {output_dir / 'mimiciii_clean.feather'}")
