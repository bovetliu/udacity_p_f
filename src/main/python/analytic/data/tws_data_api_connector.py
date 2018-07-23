import datetime
import os
from enum import Enum
from typing import List

import numpy as np
import pandas as pd
import requests
from pandas import DataFrame

from analytic import time_utils
from analytic.utility import RAW_DATA_PATH, MAIN_RESOURCES_PATH

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

__dtype_dict = {
    "m_time": np.int32,
    "m_volume": np.int64,
    "m_count": np.int32,
    "m_wap": np.float64,
    "m_high": np.float64,
    "m_low": np.float64,
    "m_open": np.float64,
    "m_close": np.float64,
    "m_time_iso": np.str_,
}


class BarSize(Enum):
    """
    BarSize can only be of _1_MIN and _1_DAY
    """
    MIN_1 = "_1_min"
    DAY_1 = "_1_day"

    def __str__(self):
        return self.value


def get_historical_data_prices(symbol: str, end_local_date_str: str, num_of_bars: int, bar_size: BarSize,
                               inside_rth: bool) -> DataFrame:
    """
    get historical data from stockData app.
    :param symbol: stock symbol
    :param end_local_date_str: end date string, having a format of "%Y%m%d %H:%M:%S"
    :param num_of_bars: number of bars
    :param bar_size: bar size
    :param inside_rth: inside Regular Trading Hours ?
    :return: a pandas.DataFrame indexed by m_time_iso
    """
    if not symbol:
        raise ValueError("symbol cannot be None or empty")
    if not end_local_date_str:
        raise ValueError("end local date str cannot be None or emtpy")
    # api - endpoint
    url = "http://localhost:8080/stockData/reqHistoricalPrices"

    # defining a params dict for the parameters to be sent to the API
    params = {
        "symbol": symbol,
        "endLocalDateStr": end_local_date_str,
        "numOfBars": num_of_bars,
        "barSize": bar_size.value,
        "insideRTH": inside_rth
    }

    # sending get request and saving the response as response object
    r = requests.get(url=url, params=params, timeout=20)
    # extracting data in json format
    # data = r.json()
    df = pd.read_json(r.content, orient="records", typ="frame", dtype=__dtype_dict, date_unit="s")
    df["m_time_iso"] = pd.to_datetime(df["m_time_iso"], yearfirst=True)
    df.set_index("m_time_iso", inplace=True)
    df.drop("m_time", axis=1, inplace=True)
    return df


def sync_sombol_to_local(symbol: str, bar_size: BarSize = BarSize.DAY_1, file_path: str = None):
    """
    fetch 272 trading-day-worth daily data using method get_historical_data_prices(...) and merge with local data of
    the same symbol. If there is no local symbol file, create it.

    :param bar_size:
    :param symbol: stock symbol
    :param file_path: target file path to hold data
    :return: the pandas.DateFrame updated.
    """
    if bar_size != BarSize.DAY_1:
        raise ValueError("bar_size only supports DAY_1")
    if type(symbol) is not str or not symbol:
        raise ValueError("symbol cannot be None or emtpy")
    symbol = symbol.upper()
    # noinspection PyUnresolvedReferences
    current_timestamp = pd.Timestamp(ts_input=datetime.datetime.now())
    if file_path is None:
        file_path = RAW_DATA_PATH + "/" + symbol + "-TWS-DATA-{}.csv".format(
            "DAILY" if bar_size == BarSize.DAY_1 else "MINUTELY")
    num_of_days_needed = 272
    orignal_df = None
    try:
        orignal_df = pd.read_csv(file_path, parse_dates=True, index_col="m_time_iso", dtype=__dtype_dict)
        time_delta = current_timestamp - orignal_df.index[-1]
        print("time_delta.days: {}".format(time_delta.days))
        num_of_days_needed = time_delta.days
    except FileNotFoundError:
        print("no filepath : {}".format(file_path))

    end_local_date_str = current_timestamp.strftime(time_utils.IB_LOCAL_DATETIME_FORMAT)
    print("end_local_date_str: {}".format(end_local_date_str))
    df = get_historical_data_prices(symbol, end_local_date_str, num_of_days_needed, bar_size, True)
    if orignal_df is None:
        df.to_csv(file_path)
        return df
    for ele in df.index:
        orignal_df.loc[ele] = df.loc[ele]
    orignal_df.to_csv(file_path)
    return orignal_df


def query_symbol_list(index_name: str, query=None, queried_column: str = None, return_df: bool = False) -> List[str]:
    """
    find symbols matching query

    :param index_name: index name like "sp500"
    :param query: the query, can be one single str or a list of strings, such as "lmt" or ["Advanced", "apple"]
    :param queried_column: specify which column should be used to match
    :param return_df: indicate whether to return DataFrame or a list of symbols
    :return: list of symbols matching query.
    """
    if isinstance(queried_column, str) and queried_column.lower() == "symbol":
        if isinstance(query, str):
            query = query.upper()
    if query is not None and (not isinstance(query, str) and not isinstance(query, list)):
        raise TypeError("query can only be a str or a list of strings")
    index_symbols_dir = os.path.join(MAIN_RESOURCES_PATH, "index_symbols")
    target_file = os.path.join(index_symbols_dir, index_name + ".csv")
    symbol_df = pd.read_csv(target_file, index_col="symbol", dtype={
        "symbol": np.str,
        "name": np.str,
        "sector": np.str
    })
    if queried_column == "sector":
        if isinstance(query, str):
            symbol_df = symbol_df.loc[symbol_df['sector'] == query]
        elif isinstance(query, list):
            symbol_df = symbol_df.loc[symbol_df['sector'].isin(query)]
    if (queried_column is None and query is not None) or queried_column == "name":
        def filter_func(name):
            if name is None or not isinstance(name, str):
                return False
            name = name.lower()
            if isinstance(query, str):
                return query.lower() in name
            else:
                return any(q.lower() in name for q in query)
        symbol_df = symbol_df.loc[symbol_df["name"].apply(filter_func)]

    return symbol_df if return_df else symbol_df.index.tolist()


def get_local_data(symbol: str, bar_size: BarSize = BarSize.DAY_1, file_path: str = None) -> DataFrame:
    if type(symbol) is not str or not symbol:
        raise ValueError("symbol cannot be None or emtpy")
    symbol = symbol.upper()
    if bar_size != BarSize.DAY_1:
        raise ValueError("bar_size only supports DAY_1")
    if file_path is None:
        file_path = RAW_DATA_PATH + "/" + symbol + "-TWS-DATA-{}.csv".format(
            "DAILY" if bar_size == BarSize.DAY_1 else "MINUTELY")
    return pd.read_csv(file_path, parse_dates=True, index_col="m_time_iso", dtype=__dtype_dict)
