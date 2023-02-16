from prepare_data.utils import (
    TextPreprocessor,
)
import pandas as pd
import vaex

df_test = pd.DataFrame(
    {
        "text": [
            "This is a tést",
            "This åø is 2 nd TEst? [**what 123**] he's [**MD Number(1) 1605**]",
            "This is 54 356 354 63.5 a 3rd tæst****",
        ],
        "label": [[], [1, 2, 3], [4, 5, 6]],
    }
)


def test_text_preprocessor():
    df_test_vaex = vaex.from_pandas(df_test, copy_index=False)
    preprocessor = TextPreprocessor(
        lower=True,
        remove_special_characters_mullenbach=False,
        remove_digits=False,
        convert_danish_characters=False,
    )
    df = preprocessor(df_test_vaex)
    assert df["text"].tolist() == [
        "this is a tést",
        "this åø is 2 nd test? [**what 123**] he's [**md number(1) 1605**]",
        "this is 54 356 354 63.5 a 3rd tæst****",
    ]

    df_test_vaex = vaex.from_pandas(df_test, copy_index=False)
    preprocessor = TextPreprocessor(
        lower=False,
        remove_special_characters_mullenbach=True,
        remove_digits=False,
        convert_danish_characters=False,
    )
    df = preprocessor(df_test_vaex)
    assert df["text"].tolist() == [
        "This is a t st",
        "This is 2 nd TEst what 123 he s MD Number 1 1605",
        "This is 54 356 354 63 5 a 3rd t st",
    ]

    df_test_vaex = vaex.from_pandas(df_test, copy_index=False)
    preprocessor = TextPreprocessor(
        lower=False,
        remove_special_characters_mullenbach=False,
        remove_digits=True,
        convert_danish_characters=False,
    )
    df = preprocessor(df_test_vaex)
    assert df["text"].tolist() == [
        "This is a tést",
        "This åø is nd TEst? [**what 123**] he's [**MD Number(1) 1605**]",
        "This is 63.5 a 3rd tæst****",
    ]

    df_test_vaex = vaex.from_pandas(df_test, copy_index=False)
    preprocessor = TextPreprocessor(
        lower=True,
        remove_special_characters_mullenbach=True,
        remove_digits=True,
        convert_danish_characters=True,
    )
    df = preprocessor(df_test_vaex)
    assert df["text"].tolist() == [
        "this is a t st",
        "this aaoe is nd test what he s md number",
        "this is a 3rd taest",
    ]

    df_test_vaex = vaex.from_pandas(df_test, copy_index=False)
    preprocessor = TextPreprocessor(
        lower=True,
        remove_special_characters_mullenbach=True,
        remove_accents=True,
        remove_digits=True,
        convert_danish_characters=True,
        remove_brackets=True,
    )
    df_test_vaex = vaex.from_pandas(df_test, copy_index=False)
    df = preprocessor(df_test_vaex)
    assert df["text"].tolist() == [
        "this is a test",
        "this aaoe is nd test he s",
        "this is a 3rd taest",
    ]
    df_test_vaex = vaex.from_pandas(df_test, copy_index=False)
    preprocessor = TextPreprocessor(
        lower=True,
        remove_special_characters=True,
        remove_accents=True,
        remove_digits=False,
        convert_danish_characters=True,
        remove_brackets=True,
    )
    df = preprocessor(df_test_vaex)
    assert df["text"].tolist() == [
        "this is a test",
        "this aaoe is 2 nd test hes",
        "this is 54 356 354 635 a 3rd taest",
    ]
