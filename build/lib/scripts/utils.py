import wget
import logging
from functools import partial
from collections import Counter

import pandas as pd
from pathlib import Path


def make_version_dir(output_dir: Path) -> Path:
    """Creates a new directory for the current version of the dataset."""
    if not output_dir.is_dir():
        output_dir = output_dir / "1"
        logging.info(f"Creating directory {output_dir}")
        output_dir.mkdir(parents=True)
    else:
        logging.info(f"Directory {output_dir} already exists")
        latest_version = max(output_dir.iterdir(), key=lambda x: int(x.name))
        new_version = str(int(latest_version.name) + 1)
        logging.info(
            f"Latest version is {latest_version.name}, bumping to {new_version}"
        )
        output_dir = output_dir / new_version
        logging.info(f"Creating directory {output_dir}")
        output_dir.mkdir()
    return output_dir


def get_mimiciii_notes(download_dir: Path) -> pd.DataFrame:
    """Reads the notes from the mimiciii dataset into a dataframe."""
    # Reading .feather files is faster than reading .csv and .csv.gz files
    if (download_dir / "NOTEEVENTS.feather").is_file():
        logging.info(
            "NOTEEVENTS.feather already exists, loading data from NOTEEVENTS.feather into a pandas dataframe."
        )
        mimic_notes = pd.read_feather(download_dir / "NOTEEVENTS.feather")
    else:
        logging.info(
            "Loading data from NOTEEVENTS.csv.gz into a pandas dataframe. This may take a while..."
        )
        mimic_notes = pd.read_csv(
            download_dir / "NOTEEVENTS.csv.gz", compression="gzip"
        )
        mimic_notes.to_feather(download_dir / "NOTEEVENTS.feather")

    return mimic_notes


def download_mullenbach_icd9_description() -> pd.DataFrame:
    """Download the icd9 description file from the mullenbach github repo

    Returns:
        pd.DataFrame: ICD9 description dataframe
    """
    logging.info("Downloading ICD9 description file...")
    url = "https://raw.githubusercontent.com/jamesmullenbach/caml-mimic/master/mimicdata/ICD9_descriptions"
    df = pd.read_csv(url, sep="\t", header=None)
    df.columns = ["icd9_code", "icd9_description"]
    return df


def get_icd9_descriptions(download_dir: Path) -> pd.DataFrame:
    """Gets the IC  D9 descriptions"""
    icd9_proc_desc = pd.read_csv(
        download_dir / "D_ICD_PROCEDURES.csv.gz", compression="gzip"
    )
    icd9_proc_desc = clean_icd9_desc_df(icd9_proc_desc, is_diag=False)
    icd9_diag_desc = pd.read_csv(
        download_dir / "D_ICD_DIAGNOSES.csv.gz", compression="gzip"
    )
    icd9_diag_desc = clean_icd9_desc_df(icd9_diag_desc, is_diag=True)
    icd9_mullenbach_desc = download_mullenbach_icd9_description()
    icd9_desc = pd.concat([icd9_proc_desc, icd9_diag_desc, icd9_mullenbach_desc])
    return icd9_desc.drop_duplicates(subset=["icd9_code"])


def clean_icd9_desc_df(icd_desc: pd.DataFrame, is_diag: bool) -> pd.DataFrame:
    """
    Cleans the ICD9 description dataframe.
    Args:
        icd_desc (pd.DataFrame): ICD9 description dataframe to clean

    Returns:
        pd.DataFrame: Clean ICD9 description dataframe
    """
    icd_desc = icd_desc.rename(
        columns={"ICD9_CODE": "icd9_code", "LONG_TITLE": "icd9_description"}
    )
    icd_desc["icd9_code"] = icd_desc["icd9_code"].astype(str)
    icd_desc["icd9_code"] = icd_desc["icd9_code"].apply(
        lambda code: reformat_icd(code, is_diag)
    )
    return icd_desc[["icd9_code", "icd9_description"]]


def download_mullenbach_splits(splits: list[str], download_directory: Path) -> None:
    """Downloads the Mullenbach splits from the github repository."""
    for split in splits:
        download_url = f"https://raw.githubusercontent.com/jamesmullenbach/caml-mimic/master/mimicdata/mimic3/{split}_hadm_ids.csv"
        logging.info(f"\nDownloading - {split}:")
        wget.download(download_url, str(download_directory))


def get_mullenbach_splits(
    splits: list[str], download_directory: Path
) -> dict[str, pd.DataFrame]:
    """Gets the downloaded Mullenbach splits."""
    splits_dict = {}
    for split in splits:
        logging.info(f"\nLoading - {split}:")
        splits_dict[split] = pd.read_csv(
            download_directory / f"{split}_hadm_ids.csv", header=None
        )
        splits_dict[split].columns = ["hadm_id"]
    return splits_dict


def filter_mullenbach_splits(
    splits: dict[str, pd.DataFrame], dataset: pd.DataFrame
) -> pd.DataFrame:
    """Filters the Mullenbach splits to only include hadm_ids that are available in the full dataset."""
    filtered_splits = {}
    for split, df in splits.items():
        logging.info(f"\nFiltering - {split}:")
        len_df = len(df)
        filtered_splits[split] = df[df["hadm_id"].isin(dataset["hadm_id"])]
        filtered_splits[split] = filtered_splits[split].reset_index(drop=True)
        logging.info(f"\t{len_df} -> {len(filtered_splits[split])}")
    return filtered_splits


def save_mullenbach_splits(
    splits: dict[str, pd.DataFrame], ouput_directory: Path
) -> None:
    """Saves the filtered Mullenbach splits to feather files."""
    for split, df in splits.items():
        logging.info(f"\nSaving - {split}:")
        df.to_feather(ouput_directory / f"{split}_hadm_ids.feather")


def merge_code_dataframes(code_dfs: list[pd.DataFrame]) -> pd.DataFrame:
    """Merges all code dataframes into a single dataframe.

    Args:
        code_dfs (list[pd.DataFrame]): List of code dataframes.

    Returns:
        pd.DataFrame: Merged code dataframe.
    """
    merged_codes = code_dfs[0]
    for code_df in code_dfs[1:]:
        merged_codes = merged_codes.merge(
            code_df, how="outer", on=["subject_id", "hadm_id"]
        )
    return merged_codes


def replace_nans_with_empty_lists(
    df: pd.DataFrame, columns: list[str] = ["cpt_id", "icd9_diag", "icd9_proc"]
) -> pd.DataFrame:
    """Replaces nans in the columns with empty lists."""
    for column in columns:
        df.loc[df[column].isnull(), [column]] = df.loc[
            df[column].isnull(), column
        ].apply(lambda x: [])
    return df


def reformat_icd(code: str, is_diag: bool) -> str:
    """
    Put a period in the right place because the MIMIC-3 data files exclude them.
    Generally, procedure codes have dots after the first two digits,
    while diagnosis codes have dots after the first three digits.
    """
    code = "".join(code.split("."))
    if is_diag:
        if code.startswith("E"):
            if len(code) > 4:
                return code[:4] + "." + code[4:]
        else:
            if len(code) > 3:
                return code[:3] + "." + code[3:]
    else:
        if len(code) > 2:
            return code[:2] + "." + code[2:]
    return code


def reformat_code_dataframe(row: pd.DataFrame, col: str) -> pd.Series:
    """Takes a dataframe and a column name and returns a series with the column name and a list of codes.

    Example:
        Input:

            SUBJECT_ID  HADM_ID icd9_diag
        608           2   163353     V3001
        609           2   163353      V053
        610           2   163353      V290

        Output:

        icd9_diag    [V053, V290, V3001]

    Args:
        row (pd.DataFrame): Dataframe with a column of codes.
        col (str): column name of the codes.

    Returns:
        pd.Series: Series with the column name and a list of codes.
    """
    return pd.Series({col: row[col].sort_values().tolist()})


def merge_report_addendum_helper_function(row: pd.DataFrame) -> pd.Series:
    """Merges the report and addendum text."""
    dout = dict()
    if len(row) == 1:
        dout["DESCRIPTION"] = row.iloc[0].DESCRIPTION
        dout["TEXT"] = row.iloc[0].TEXT
    else:
        row = row.sort_values(["DESCRIPTION", "CHARTDATE"], ascending=[False, True])
        dout["DESCRIPTION"] = "+".join(row.DESCRIPTION)
        dout["TEXT"] = " ".join(row.TEXT)
    return pd.Series(dout)


def format_code_dataframe(df: pd.DataFrame, col_in: str, col_out: str) -> pd.DataFrame:
    """Formats the code dataframe.

    Args:
        df (pd.DataFrame): The dataframe containing the codes.
        col_in (str): The name of the column containing the codes.
        col_out (str): The new name of the column containing the codes

    Returns:
        pd.DataFrame: The formatted dataframe.
    """
    df = df.rename(
        columns={"HADM_ID": "hadm_id", "SUBJECT_ID": "subject_id", "TEXT": "text"}
    )
    df = df.sort_values(["subject_id", "hadm_id"])
    df[col_in] = df[col_in].astype(str).str.strip()
    df = df[["subject_id", "hadm_id", col_in]].rename({col_in: col_out}, axis=1)
    # remove codes that are nan
    df = df[df[col_out] != "nan"]
    return (
        df.groupby(["subject_id", "hadm_id"])
        .apply(partial(reformat_code_dataframe, col=col_out))
        .reset_index()
    )


def merge_reports_addendum(mimic_notes: pd.DataFrame) -> pd.DataFrame:
    """Merges the reports and addendum into one dataframe.

    Args:
        mimic_notes (pd.DataFrame): The dataframe containing the notes from the mimiciii dataset.

    Returns:
        pd.DataFrame: The dataframe containing the discharge summaries consisting of reports and addendum.
    """
    discharge_summaries = mimic_notes[mimic_notes["CATEGORY"] == "Discharge summary"]
    discharge_summaries.HADM_ID = discharge_summaries.HADM_ID.astype(int)
    return (
        discharge_summaries.groupby(["SUBJECT_ID", "HADM_ID"])
        .apply(merge_report_addendum_helper_function)
        .reset_index()
    )


def top_k_codes(df: pd.DataFrame, column_names: list[str], k: int) -> set[str]:
    """Get the top k most frequent codes from a dataframe"""
    df = df.copy()
    counter = Counter()
    for col in column_names:
        list(map(counter.update, df[col]))
    top_k = counter.most_common(k)
    return set(map(lambda x: x[0], top_k))


def filter_codes(
    df: pd.DataFrame, column_names: list[str], codes_to_keep: set[str]
) -> pd.DataFrame:
    """Filter the codes in the dataframe to only keep the desired codes"""
    df = df.copy()
    for col in column_names:
        df[col] = df[col].apply(
            lambda codes: list(filter(lambda x: x in codes_to_keep, codes))
        )
    return df


def remove_duplicated_codes(df: pd.DataFrame, column_names: list[str]) -> pd.DataFrame:
    """Remove duplicated codes in the dataframe"""
    df = df.copy()
    for col in column_names:
        df[col] = df[col].apply(lambda codes: list(set(codes)))
    return df
