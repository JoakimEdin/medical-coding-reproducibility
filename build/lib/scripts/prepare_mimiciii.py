import re
from functools import partial
import logging

from pathlib import Path
import pandas as pd
from nltk.tokenize import RegexpTokenizer

from scripts import (
    download_mullenbach_splits,
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
)

OUTPUT_DIRECTORY = "/data/je/mimiciii/pre-processed/sotiris"  # Where the pre-processed data will be stored
DOWNLOAD_DIRECTORY = "/data/je/mimiciii/1.4"
SUBSETS = ["train_50", "train_full", "dev_50", "dev_full", "test_50", "test_full"]

# The dataset requires a Licence in physionet. Once it is obtained, download the dataset with the following command in the terminal:
# wget -r -N -c -np --user <your_physionet_user_name> --ask-password https://physionet.org/files/mimiciii/1.4/
# Change the path of DOWNLOAD_DIRECTORY to the path where you downloaded mimiciii

logging.basicConfig(level=logging.INFO)

download_dir = Path(DOWNLOAD_DIRECTORY)
output_dir = Path(OUTPUT_DIRECTORY)


def preprocess_text(text):
    # Remove all anonymized stuff
    text = re.sub(r"\[.*?\]", r"", text)
    # Remove leading / trailing spaces
    text = re.sub(r"(^ +| +$)", r"", text, flags=re.M)
    # Concatenate lines with just one newline
    text = re.sub(r"([^\s])(\n)([^\s])", r"\1 \3", text)
    # Remove empty lines (whitespace)
    text = re.sub(r"^[^\w]*$", r"", text, flags=re.M)
    # Replace multiple spaces with one
    text = re.sub(r" {2,}", r" ", text)
    # Replace multiple newlines with one
    text = re.sub(r"\n{2,}", r"\n", text)
    return text


def preprocess_notes(mimic_notes: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the notes dataframe

    Args:
        mimic_notes (pd.DataFrame): The notes dataframe

    Returns:
        pd.DataFrame: The preprocessed notes dataframe
    """
    # Remove NaNs
    mimic_notes = mimic_notes[mimic_notes.ISERROR.isna()].drop("ISERROR", axis=1)

    discharge_summaries = merge_reports_addendum(mimic_notes)

    logging.info("Preprocessing discharge summaries...")
    discharge_summaries = discharge_summaries.TEXT.apply(preprocess_text)

    discharge_summaries = discharge_summaries[
        discharge_summaries.DESCRIPTION.str.contains("Report")
    ]
    discharge_summaries = discharge_summaries.sort_values(["SUBJECT_ID", "HADM_ID"])
    discharge_summaries = discharge_summaries.rename(
        columns={"HADM_ID": "hadm_id", "SUBJECT_ID": "subject_id", "TEXT": "text"}
    )
    discharge_summaries["num_words"] = discharge_summaries.text.apply(
        lambda s: len(s.split())
    )
    discharge_summaries["num_char"] = discharge_summaries.text.apply(lambda s: len(s))
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


output_dir = make_version_dir(output_dir)
mimic_notes = get_mimiciii_notes(download_dir)
discharge_summaries = preprocess_notes(mimic_notes)
code_systems = [
    ("CPT", "CPTEVENTS.csv.gz", "CPT_CD", "cpt_id"),
    ("ICD9-DIAG", "DIAGNOSES_ICD.csv.gz", "ICD9_CODE", "icd9_diag"),
    ("ICD9-PROC", "PROCEDURES_ICD.csv.gz", "ICD9_CODE", "icd9_proc"),
]
merged_codes = download_and_preprocess_code_systems(code_systems)
# Filter codes that do not have clinical notes
full_dataset = discharge_summaries.merge(
    merged_codes, on=["subject_id", "hadm_id"], how="left"
)
full_dataset = full_dataset.reset_index(drop=True)
full_dataset.to_feather(output_dir / "full_dataset.feather")
logging.info(f"Saved full dataset to {output_dir / 'full_dataset.feather'}")

# Download, process and save the Mullenbach splits
download_mullenbach_splits(SUBSETS, download_dir)
mullenbach_splits = get_mullenbach_splits(SUBSETS, download_dir)
filtered_mullenbach_splits = filter_mullenbach_splits(mullenbach_splits, full_dataset)
save_mullenbach_splits(filtered_mullenbach_splits, output_dir)

logging.info("\nMimic III dataset succesfully processed!")
