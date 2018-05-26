from typing import List
from enum import Enum
import requests
import json
import pandas as pd
from pandas import DataFrame
import numpy as np
import datetime
from analytic import time_utils
from analytic.utility import RAW_DATA_PATH

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


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
    if not symbol:
        raise ValueError("symbol cannot be None or empty")
    if not end_local_date_str:
        raise ValueError("end local date str cannot be None or emtpy")
    # api - endpoint
    url = "http://localhost:8080/stockData/reqHistoricalPrices"

    # location given here
    location = "delhi technological university"

    # defining a params dict for the parameters to be sent to the API
    params = {
        "symbol": symbol,
        "endLocalDateStr": end_local_date_str,
        "numOfBars": num_of_bars,
        "barSize": bar_size.value,
        "insideRTH": inside_rth
    }

    # sending get request and saving the response as response object
    r = requests.get(url=url, params=params, timeout=60)
    # extracting data in json format
    # data = r.json()

    dtype_dict = {
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
    df = pd.read_json(r.content, orient="records", typ="frame", dtype=dtype_dict, date_unit="s")
    df["m_time_iso"] = pd.to_datetime(df["m_time_iso"], yearfirst=True)
    df.set_index("m_time_iso", inplace=True)
    df.drop("m_time", axis=1, inplace=True)

    print(df)
    return df


def sync_sombol_to_local(symbol: str, bar_size: BarSize=BarSize.DAY_1, file_path: str=None):
    """
    by default store 1 year worth daily data to target file_path
    :param bar_size:
    :param symbol: stock symbol
    :param file_path: target file path to hold data
    :return: true if new data is downloaded, false means file_path is already updated.
    """
    if not symbol:
        raise ValueError("symbol cannot be None or emtpy")
    symbol = symbol.capitalize()

    if not file_path:
        file_path = RAW_DATA_PATH + "/" + symbol + "-TWS-DATA.csv"
    end_local_date_str = datetime.datetime.now().strftime(time_utils.IB_LOCAL_DATETIME_FORMAT)
    print(end_local_date_str)
    df = get_historical_data_prices(symbol, end_local_date_str, 272, bar_size, True)
    df.to_csv(file_path)

