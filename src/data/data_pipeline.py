from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.feather
import vaex
from omegaconf import OmegaConf

from src.data.datatypes import Data
from src.settings import ID_COLUMN, TARGET_COLUMN, TEXT_COLUMN


def get_code_system2code_counts(
    df: vaex.dataframe.DataFrame, code_systems: list[str]
) -> dict[str, dict[str, int]]:
    """

    Args:
        df (vaex.dataframe.DataFrame): The dataset in vaex dataframe format
        code_systems (list[str]): list of code systems to get counts for
    Returns:
        dict[str, dict[str, int]]: A dictionary with code systems as keys and a dictionary of code counts as values
    """
    code_system2code_counts = defaultdict(dict)
    for col in code_systems:
        codes = df[col].values.flatten().value_counts().to_pylist()
        code_system2code_counts[col] = {
            code["values"]: code["counts"] for code in codes
        }
    return code_system2code_counts


def data_pipeline(config: OmegaConf) -> Data:
    """The data pipeline.

    Args:
        config (OmegaConf): The config.

    Returns:
        Data: The data.
    """
    dir = Path(config.dir)
    with vaex.cache.memory_infinite():  # type: ignore
        df = vaex.from_arrow_table(
            pyarrow.feather.read_table(
                dir / config.data_filename,
                columns=[
                    ID_COLUMN,
                    TEXT_COLUMN,
                    TARGET_COLUMN,
                    "num_words",
                    "num_targets",
                ]
                + config.code_column_names,
            )
        )
        splits = vaex.from_arrow_table(
            pyarrow.feather.read_table(
                dir / config.split_filename,
            )
        )
        df = df.join(splits, on=ID_COLUMN, how="inner")
        code_system2code_counts = get_code_system2code_counts(
            df, config.code_column_names
        )
        schema = pa.schema(
            [
                pa.field(ID_COLUMN, pa.int64()),
                pa.field(TEXT_COLUMN, pa.large_utf8()),
                pa.field(TARGET_COLUMN, pa.list_(pa.large_string())),
                pa.field("split", pa.large_string()),
                pa.field("num_words", pa.int64()),
                pa.field("num_targets", pa.int64()),
            ]
        )

        return Data(
            df[
                [
                    ID_COLUMN,
                    TEXT_COLUMN,
                    TARGET_COLUMN,
                    "split",
                    "num_words",
                    "num_targets",
                ]
            ]
            .to_arrow_table()
            .cast(schema),
            code_system2code_counts,
        )
