import re
from typing import List, Tuple

import pandas as pd


def load_data(path: str) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], List[str]]:
    """
    Load data from a CSV file and perform initial preprocessing.

    Args:
        path (str): path to the dataset

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, List[str], List[str]]: tuple of
           preprocessed data, targets, text features and numerical features.
    """

    df = pd.read_csv(path)
    df['DA'] = df['DA'].astype(int).apply(lambda x: 0 if x == -1 else x)
    targets = df[['DS', 'DA']]
    df = df.drop(columns=['Unnamed: 0', 'DS', 'DA', 'DS, predicted', 'DA, predicted', 'Ссылка',
                        'Аналитические комбинации слов', 'Аналитические комбинации слов в работе'])

    cols = list(df.columns)
    cols = list(filter(lambda x: not x.isdigit(), cols))
    df = df[cols]

    text_features = list(df[cols].dtypes[df[cols].dtypes == 'object'].index)
    num_features = list(df[cols].dtypes[df[cols].dtypes != 'object'].index)

    return df, targets, text_features, num_features


def clean_text(text_series):
    """Converts text features to lowercase and removes abundant punctuation and whitespaces."""
    text_series = text_series.fillna('').astype(str)
    cleaned = text_series.apply(lambda x: 
        x.lower()
        .strip())
    return cleaned


def clean_dataset(df, text_features):
    """Applies `clean_text` to all text features."""
    for tf in text_features:
        df[tf] = clean_text(df[tf])
    return df