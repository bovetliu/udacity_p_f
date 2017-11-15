# pandas fillna function
import pandas as pd


def fill_missing_values(df_data: pd.DataFrame) -> pd.DataFrame:
    """Fill missing values in data frame, in place, method appeared in quiz of lesson 6, Incomplete Data"""
    df_data.fillna(method='ffill', inplace=True)
    df_data.fillna(method='bfill', inplace=True)
    return df_data
