from pathlib import Path

import pandas as pd


def join_codes(df: pd.DataFrame, column_names: list[str]) -> pd.DataFrame:
    """Join the codes in the column names to a label column.

    Args:
        df (pd.DataFrame): The dataset to join the codes.
        column_names (list[str]): The column names to join.

    Returns:
        pd.DataFrame: The dataset with the joined codes.
    """
    df["label"] = [[] for _ in range(len(df))]
    for column_name in column_names:
        df["label"] += df[column_name].apply(lambda x: x.tolist())
    return df


def get_splits(split_config: dict[list[str]], data_path: Path) -> dict[str, list[int]]:
    """Get the splits from the split config.

    Args:
        split_config (dict[list[str]]): The split config.
        data_path (Path): The path to the data.

    Returns:
        dict[str, list[int]]: The splits.
    """
    splits = {}
    for split_name, split_files in split_config.items():
        split = []
        for split_file in split_files:
            split += pd.read_feather(data_path / split_file)["hadm_id"].tolist()
        splits[split_name] = split
    return splits


def get_split_df(
    split_ids: list[int], df: pd.DataFrame
) -> list[tuple[str, set[str], dict]]:
    """Get the split from the split ids.

    Args:
        split_ids (list[int]): The split ids.
        df (pd.DataFrame): The dataset.

    Returns:
        list[tuple[str,set(str),int]]: The split. The first element is the data, the second element is the labels and the third element is the index.
    """
    df = df.copy()
    df = df[df["hadm_id"].isin(split_ids)]
    split = list(df[["text", "label", "hadm_id"]].itertuples(index=False, name=None))
    # convert labels to sets
    return [(data, set(label), {"hadm_id": idx}) for data, label, idx in split]


def get_icd9_description_dict(path: str) -> dict[str, str]:
    df = pd.read_feather(path)
    return dict(zip(df["icd9_code"], df["icd9_description"]))
