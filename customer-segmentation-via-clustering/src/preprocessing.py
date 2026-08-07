import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


DROP_COLUMNS = ["Channel", "Region"]


def clean_data(data):
    data = data.copy()

    # Remove duplicate rows
    data = data.drop_duplicates()

    # Drop columns not used for behavioral clustering
    data = data.drop(columns=DROP_COLUMNS, errors="ignore")

    return data


def transform_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Apply log1p transformation to reduce skewness.
    """

    return pd.DataFrame(
        np.log1p(data),
        columns=data.columns,
        index=data.index
    )


def scale_data(data_log: pd.DataFrame):
    """
    Standardize transformed features.
    """

    scaler = StandardScaler()

    data_scaled = scaler.fit_transform(data_log)

    return data_scaled, scaler