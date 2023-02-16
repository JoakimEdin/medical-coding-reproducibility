import pandas as pd
import torch
from omegaconf import OmegaConf

from src.data.datatypes import Data, Lookups
from src.data.transform import Transform


def load_lookups(
    config: OmegaConf,
    data: Data,
    label_transform: Transform,
    text_transform: Transform,
) -> Lookups:
    """Load the lookups.

    Args:
        config (OmegaConf): The config.

    Returns:
        Lookups: The lookups.
    """
    data_info = get_data_info(
        data, text_transform.vocab_size, label_transform.pad_index
    )
    code_system2code_indices = get_code_system2code_indices(data, label_transform)
    split2code_indices = get_split2code_indices(data, label_transform)

    return Lookups(
        data_info=data_info,
        code_system2code_indices=code_system2code_indices,
        split2code_indices=split2code_indices,
    )


def get_code_system2code_indices(
    data: Data, label_transform: Transform
) -> dict[str, torch.Tensor]:
    code_system2code_indices = {}
    for codesystem, codes in data.code_system2code_counts.items():
        code_system2code_indices[codesystem] = label_transform.get_indices(
            set(codes.keys())
        )
    return code_system2code_indices


def get_split2code_indices(
    data: Data, label_transform: Transform
) -> dict[str, torch.Tensor]:
    split2code_indices = {}
    split2code_indices["train"] = label_transform.get_indices(
        data.split_targets("train")
    )
    split2code_indices["train_val"] = label_transform.get_indices(
        data.split_targets("train")
    )
    split2code_indices["val"] = label_transform.get_indices(data.split_targets("val"))
    split2code_indices["test"] = label_transform.get_indices(data.split_targets("test"))
    return split2code_indices


def get_data_info(data: Data, vocab_size: int, pad_index: int) -> dict:
    data_info = data.info
    data_info["vocab_size"] = vocab_size
    data_info["pad_index"] = pad_index
    return data_info


# def get_code_description_lookup(
#     preprocessor: BasePreprocessor, config: OmegaConf
# ) -> dict[str, str]:
#     path = config.code_desc_path
#     if path is None:
#         return None

#     df = pd.read_feather(path)
#     df[config.desc_column] = preprocessor.preprocess_text(
#         [config.desc_column].to_list()
#     )
#     return dict(zip(df[config.code_column], df[config.desc_column]))
