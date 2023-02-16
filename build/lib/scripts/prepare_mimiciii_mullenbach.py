import re
from functools import partial
import logging
from collections import Counter
from pathlib import Path
import pandas as pd
from nltk.tokenize import RegexpTokenizer

from scripts import (
    download_mullenbach_splits,
    download_mullenbach_icd9_description,
    get_icd9_descriptions,
    get_mullenbach_splits,
    filter_mullenbach_splits,
    save_mullenbach_splits,
    merge_code_dataframes,
    replace_nans_with_empty_lists,
    reformat_icd,
    merge_reports_addendum,
    make_version_dir,
    get_mimiciii_notes,
    format_code_dataframe,
    remove_duplicated_codes,
    filter_codes,
)

OUTPUT_DIRECTORY = "/data/je/mimiciii/pre-processed/mullenbach"  # Where the pre-processed data will be stored
DOWNLOAD_DIRECTORY = "/data/je/mimiciii/1.4"
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

# The dataset requires a Licence in physionet. Once it is obtained, download the dataset with the following command in the terminal:
# wget -r -N -c -np --user <your_physionet_user_name> --ask-password https://physionet.org/files/mimiciii/1.4/
# Change the path of DOWNLOAD_DIRECTORY to the path where you downloaded mimiciii

logging.basicConfig(level=logging.INFO)

download_dir = Path(DOWNLOAD_DIRECTORY)
output_dir = Path(OUTPUT_DIRECTORY)

# retain only alphanumeric
tokenizer = RegexpTokenizer(r"\w+")


def preprocess_text(text: str) -> str:
    tokens = [t.lower() for t in tokenizer.tokenize(text) if not t.isnumeric()]
    # Don't know why they add quotations, but it's in the original code
    text = '"' + " ".join(tokens) + '"'
    return text


def preprocess_icd9_description(text: str) -> str:
    tokens = [t.lower() for t in tokenizer.tokenize(text) if not t.isnumeric()]
    return " ".join(tokens)


def preprocess_notes(mimic_notes: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the notes dataframe

    Args:
        mimic_notes (pd.DataFrame): The notes dataframe

    Returns:
        pd.DataFrame: The preprocessed notes dataframe
    """

    logging.info("Preprocessing discharge summaries...")
    mimic_notes["TEXT"] = mimic_notes["TEXT"].apply(preprocess_text)
    logging.info(f"{mimic_notes.HADM_ID.nunique()} number of admissions")
    discharge_summaries = merge_reports_addendum(mimic_notes)
    discharge_summaries = discharge_summaries.sort_values(["SUBJECT_ID", "HADM_ID"])
    discharge_summaries = discharge_summaries.rename(
        columns={"HADM_ID": "hadm_id", "SUBJECT_ID": "subject_id", "TEXT": "text"}
    )
    discharge_summaries = discharge_summaries.reset_index(drop=True)
    logging.info(
        f"{discharge_summaries.subject_id.nunique()} subjects, {discharge_summaries.hadm_id.nunique()} admissions"
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
        df = pd.read_csv(download_dir / fname, compression="gzip")
        df = format_code_dataframe(df, col_in, col_out)
        df = remove_duplicated_codes(df, [col_out])
        code_dfs.append(df)

    merged_codes = merge_code_dataframes(code_dfs)
    merged_codes = replace_nans_with_empty_lists(merged_codes)
    merged_codes["icd9_diag"] = merged_codes["icd9_diag"].apply(
        lambda codes: list(map(partial(reformat_icd, is_diag=True), codes))
    )
    merged_codes["icd9_proc"] = merged_codes["icd9_proc"].apply(
        lambda codes: list(map(partial(reformat_icd, is_diag=False), codes))
    )
    return merged_codes


# MIMIC-III full
output_dir = make_version_dir(output_dir)
mimic_notes = get_mimiciii_notes(download_dir)
discharge_summaries = preprocess_notes(mimic_notes)
merged_codes = download_and_preprocess_code_systems(CODE_SYSTEMS)

full_dataset = discharge_summaries.merge(
    merged_codes, on=["subject_id", "hadm_id"], how="inner"
)
full_dataset = replace_nans_with_empty_lists(full_dataset)
logging.info(f"{discharge_summaries.hadm_id.nunique()} number of admissions")
logging.info(f"{full_dataset.hadm_id.nunique()} number of admissions")
full_dataset = full_dataset.reset_index(drop=True)
full_dataset.to_feather(output_dir / "mimiciii_full.feather")
logging.info(f"Saved full dataset to {output_dir / 'mimiciii_full.feather'}")

# MIMIC-III 50
top_50_codes = filter_codes(
    full_dataset, ["icd9_diag", "icd9_proc"], codes_to_keep=TOP_50_MULLENBACH_CODES
)
num_codes = top_50_codes["icd9_diag"].apply(len) + top_50_codes["icd9_proc"].apply(len)
top_50_codes = top_50_codes[num_codes > 0]
top_50_codes = top_50_codes.reset_index(drop=True)
top_50_codes.to_feather(output_dir / "mimiciii_50.feather")
logging.info(f"Saved top 50 dataset to {output_dir / 'mimiciii_50.feather'}")

# Download, process and save the Mullenbach splits
download_mullenbach_splits(SUBSETS, download_dir)
mullenbach_splits = get_mullenbach_splits(SUBSETS, download_dir)
filtered_mullenbach_splits = filter_mullenbach_splits(mullenbach_splits, full_dataset)
save_mullenbach_splits(filtered_mullenbach_splits, output_dir)

# ICD9 descriptions
icd_9_descriptions = get_icd9_descriptions(download_dir=download_dir)
icd_9_descriptions["icd9_description"] = icd_9_descriptions["icd9_description"].apply(
    preprocess_icd9_description
)
icd_9_descriptions = icd_9_descriptions.reset_index(drop=True)
icd_9_descriptions.to_feather(output_dir / "icd9_descriptions.feather")
logging.info(f"Saved ICD9 descriptions to {output_dir / 'icd9_descriptions.feather'}")

logging.info("\nMimic III dataset succesfully processed!")
