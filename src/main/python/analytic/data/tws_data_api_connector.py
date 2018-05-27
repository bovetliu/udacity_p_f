import datetime
from enum import Enum

import numpy as np
import pandas as pd
import requests
from pandas import DataFrame

from analytic import time_utils
from analytic.utility import RAW_DATA_PATH

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

    :param symbol:
    :param end_local_date_str:
    :param num_of_bars:
    :param bar_size:
    :param inside_rth:
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

    # print(df.head(5))
    return df


def sync_sombol_to_local(symbol: str, bar_size: BarSize = BarSize.DAY_1, file_path: str = None):
    """
    by default store 1 year worth daily data to target file_path
    :param bar_size:
    :param symbol: stock symbol
    :param file_path: target file path to hold data
    :return: true if new data is downloaded, false means file_path is already updated.
    """
    if bar_size != BarSize.DAY_1:
        raise ValueError("bar_size only supports DAY_1")
    if type(symbol) is not str or not symbol:
        raise ValueError("symbol cannot be None or emtpy")
    symbol = symbol.upper()
    current_timestamp = pd.Timestamp(ts_input=datetime.datetime.now())
    if not file_path:
        file_path = RAW_DATA_PATH + "/" + symbol + "-TWS-DATA.csv"
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
