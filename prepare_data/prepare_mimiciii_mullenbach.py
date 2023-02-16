import logging
from functools import partial
from pathlib import Path

import pandas as pd

from prepare_data.utils import (
    TextPreprocessor,
    download_mullenbach_splits,
    filter_codes,
    filter_mullenbach_splits,
    format_code_dataframe,
    get_icd9_descriptions,
    get_mimiciii_notes,
    get_mullenbach_splits,
    merge_code_dataframes,
    merge_reports_addendum,
    preprocess_documents,
    reformat_icd9,
    remove_duplicated_codes,
    replace_nans_with_empty_lists,
    save_mullenbach_splits,
)
from src.settings import (
    DATA_DIRECTORY_MIMICIII_50,
    DATA_DIRECTORY_MIMICIII_FULL,
    DOWNLOAD_DIRECTORY_MIMICIII,
    ID_COLUMN,
    SUBJECT_ID_COLUMN,
    TARGET_COLUMN,
    TEXT_COLUMN,
)

SUBSETS = ["train_50", "train_full", "dev_50", "dev_full", "test_50", "test_full"]
CODE_SYSTEMS = [
    ("CPT", "CPTEVENTS.csv.gz", "CPT_CD", "cpt_id"),
    ("ICD9-DIAG", "DIAGNOSES_ICD.csv.gz", "ICD9_CODE", "icd9_diag"),
    ("ICD9-PROC", "PROCEDURES_ICD.csv.gz", "ICD9_CODE", "icd9_proc"),
]

""" When calculating the top 50 most frequent codes, we didn't get the same results as the original paper. 
We suspect this is due to the fact that the original paper used the MIMIC-III v1.2 dataset, while we used v1.4. 
Also, they didn't remove duplicated codes. We decided to use the top 50 codes from the original paper, which are listed below."""
TOP_50_MULLENBACH_CODES = {
    "401.9",
    "38.93",
    "428.0",
    "427.31",
    "414.01",
    "96.04",
    "96.6",
    "584.9",
    "250.00",
    "96.71",
    "272.4",
    "518.81",
    "99.04",
    "39.61",
    "599.0",
    "530.81",
    "96.72",
    "272.0",
    "285.9",
    "88.56",
    "244.9",
    "486",
    "38.91",
    "285.1",
    "36.15",
    "276.2",
    "496",
    "99.15",
    "995.92",
    "V58.61",
    "507.0",
    "038.9",
    "88.72",
    "585.9",
    "403.90",
    "311",
    "305.1",
    "37.22",
    "412",
    "33.24",
    "39.95",
    "287.5",
    "410.71",
    "276.1",
    "V45.81",
    "424.0",
    "45.13",
    "V15.82",
    "511.9",
    "37.23",
}
preprocessor = TextPreprocessor(
    lower=True,
    remove_special_characters_mullenbach=True,
    remove_special_characters=False,
    remove_digits=True,
    remove_accents=False,
    remove_brackets=False,
    convert_danish_characters=False,
)
# The dataset requires a Licence in physionet. Once it is obtained, download the dataset with the following command in the terminal:
# wget -r -N -c -np --user <your_physionet_user_name> --ask-password https://physionet.org/files/mimiciii/1.4/
# Change the path of DOWNLOAD_DIRECTORY to the path where you downloaded mimiciii

logging.basicConfig(level=logging.INFO)

download_dir = Path(DOWNLOAD_DIRECTORY_MIMICIII)
output_dir_50 = Path(DATA_DIRECTORY_MIMICIII_50)
output_dir_50.mkdir(parents=True, exist_ok=True)

output_dir_full = Path(DATA_DIRECTORY_MIMICIII_FULL)
output_dir_full.mkdir(parents=True, exist_ok=True)


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


# MIMIC-III full
mimic_notes = get_mimiciii_notes(download_dir)
discharge_summaries = prepare_discharge_summaries(mimic_notes)
merged_codes = download_and_preprocess_code_systems(CODE_SYSTEMS)

full_dataset = discharge_summaries.merge(
    merged_codes, on=[SUBJECT_ID_COLUMN, ID_COLUMN], how="inner"
)
full_dataset = replace_nans_with_empty_lists(full_dataset)
# Remove admissions with no codes
full_dataset = full_dataset[full_dataset[TARGET_COLUMN].apply(len) > 0]
full_dataset = preprocess_documents(df=full_dataset, preprocessor=preprocessor)
logging.info(f"{full_dataset[ID_COLUMN].nunique()} number of admissions")
full_dataset = full_dataset.reset_index(drop=True)
full_dataset.to_feather(output_dir_full / "mimiciii_full.feather")
logging.info(f"Saved full dataset to {output_dir_full / 'mimiciii_full.feather'}")

# MIMIC-III 50
# Caluclate the top 50 most occuring codes
all_codes = full_dataset["icd9_diag"] + full_dataset["icd9_proc"]
top_50_codes = all_codes.explode().value_counts().index.tolist()[:50]

# print all codes that doesn't overlap between top_50_codes and TOP_50_CODES_MULLENBACH
print(set(top_50_codes) ^ set(TOP_50_MULLENBACH_CODES))


top_50_dataset = filter_codes(
    full_dataset,
    ["icd9_diag", "icd9_proc", TARGET_COLUMN],
    codes_to_keep=TOP_50_MULLENBACH_CODES,
)
top_50_dataset = top_50_dataset[top_50_dataset[TARGET_COLUMN].apply(len) > 0]
top_50_dataset = top_50_dataset.reset_index(drop=True)
top_50_dataset.to_feather(output_dir_50 / "mimiciii_50.feather")
logging.info(f"Saved top 50 dataset to {output_dir_50 / 'mimiciii_50.feather'}")

# Download, process and save the Mullenbach splits
download_mullenbach_splits(SUBSETS, download_dir)
mullenbach_splits = get_mullenbach_splits(SUBSETS, download_dir)
filtered_mullenbach_splits = filter_mullenbach_splits(mullenbach_splits, full_dataset)
save_mullenbach_splits(filtered_mullenbach_splits, output_dir_50, output_dir_full)

logging.info("\nMimic III dataset succesfully processed!")
