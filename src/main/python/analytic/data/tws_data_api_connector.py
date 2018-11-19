import datetime
import os
from enum import Enum
from typing import List

import numpy as np
import pandas as pd
import requests
from pandas import DataFrame
from pandas.tseries.offsets import BDay

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


# noinspection PyUnresolvedReferences
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
    num_of_bars_left = num_of_bars
    tbr_df = None
    # sending get request and saving the response as response object
    exhausted = False
    while num_of_bars_left > 0 and (not exhausted):
        params = {
            "symbol": symbol,
            "endLocalDateStr": end_local_date_str,
            "numOfBars": num_of_bars_left if num_of_bars_left < 300 else 300,
            "barSize": bar_size.value,
            "insideRTH": inside_rth
        }
        print(str(params))

        new_df = None
        cnt = 0
        len_of_df_list = []
        while cnt < 3:
            r = requests.get(url=url, params=params, timeout=20)
            r.raise_for_status()
            new_df = pd.read_json(r.content, orient="records", typ="frame", dtype=__dtype_dict, date_unit="s")
            len_of_df_list.append(len(new_df))
            if len(new_df) == params["numOfBars"]:
                break
            else:
                print("requested {} but got {} records, going to retry".format(params["numOfBars"], len(new_df)))
                cnt += 1
        if new_df is None:
            raise Exception("could not fetch required DataFrame")
        if len(new_df) != params["numOfBars"]:
            if (len_of_df_list[0] != len_of_df_list[1]) or (len_of_df_list[1] != len_of_df_list[2]):
                raise Exception("have retried three times, but could not make len(new_df) == params[\"numOfBars\"]")
            else:
                exhausted = True
        new_df["m_time_iso"] = pd.to_datetime(new_df["m_time_iso"], yearfirst=True)
        new_df.set_index("m_time_iso", inplace=True)
        new_df.drop("m_time", axis=1, inplace=True)
        if tbr_df is None:
            tbr_df = new_df
        else:
            new_df = new_df.loc[new_df.index < tbr_df.index[0]]
            tbr_df = new_df.append(tbr_df, verify_integrity=True)
        num_of_bars_left = num_of_bars - len(tbr_df)
        time_delta = pd.Timedelta("1 days") if BarSize.DAY_1 else pd.Timedelta("1 minutes")
        end_local_date_str = (tbr_df.index[0] - time_delta).strftime(time_utils.IB_LOCAL_DATETIME_FORMAT)
    return tbr_df


# noinspection PyUnresolvedReferences
def get_local_synced(symbol: str, num_of_days_needed: int = 272, bar_size: BarSize = BarSize.DAY_1,
                     file_path: str = None):
    """
    fetch 272 trading-day-worth daily data using method get_historical_data_prices(...) and merge with local data of
    the same symbol. If there is no local symbol file, create it.

    :param symbol: stock symbol
    :param num_of_days_needed number of days needed
    :param bar_size: bar size
    :param file_path: target file path to hold data
    :return: the pandas.DateFrame updated.
    """
    if bar_size != BarSize.DAY_1:
        raise ValueError("bar_size only supports DAY_1")
    if type(symbol) is not str or not symbol:
        raise ValueError("symbol cannot be None or emtpy")
    symbol = symbol.upper()
    # noinspection PyUnresolvedReferences
    # current_timestamp = pd.Timestamp(2018, 7, 20, 10, 44)
    current_timestamp = pd.Timestamp(ts_input=datetime.datetime.now())
    minus_one_bday = current_timestamp - BDay()
    temp = minus_one_bday + BDay()
    if temp != current_timestamp:
        recent_bday = pd.Timestamp(minus_one_bday.year, minus_one_bday.month, minus_one_bday.day)
    else:
        recent_bday = pd.Timestamp(current_timestamp.year, current_timestamp.month, current_timestamp.day)
    if file_path is None:
        file_path = RAW_DATA_PATH + "/" + symbol + "-TWS-DATA-{}.csv".format(
            "DAILY" if bar_size == BarSize.DAY_1 else "MINUTELY")
    try:
        orignal_df = pd.read_csv(file_path, parse_dates=True, index_col="m_time_iso", dtype=__dtype_dict)
        last_timestamp = orignal_df.index[-1]
        last_day = pd.Timestamp(last_timestamp.strftime('%Y-%m-%d'))
        first_timestamp = orignal_df.index[0]
        first_day_in_df = pd.Timestamp(first_timestamp.strftime('%Y-%m-%d'))
        time_delta = recent_bday - last_day
        num_of_days_needed_at_right = time_delta.days
        if num_of_days_needed_at_right < 0:
            raise ValueError("num_of_days_needed: {} smaller than 0, this is illegal state.".format(num_of_days_needed))
        end_local_date_str = current_timestamp.strftime(time_utils.IB_LOCAL_DATETIME_FORMAT)
        right_df = get_historical_data_prices(symbol, end_local_date_str, num_of_days_needed_at_right, bar_size, True)
        right_df = right_df.loc[right_df.index > last_timestamp]
        orignal_df = orignal_df.append(right_df, verify_integrity=True)
        num_of_days_needed_at_left = num_of_days_needed - len(orignal_df)
        if num_of_days_needed_at_left > 0:
            end_local_date_str = (first_day_in_df - pd.Timedelta('1 minutes')).strftime(
                time_utils.IB_LOCAL_DATETIME_FORMAT)
            left_df = get_historical_data_prices(symbol, end_local_date_str, num_of_days_needed_at_left, bar_size, True)
            orignal_df = left_df.append(orignal_df, verify_integrity=True)
        orignal_df.to_csv(file_path)
        return orignal_df
    except FileNotFoundError:
        print("no filepath : {}".format(file_path))
    end_local_date_str = current_timestamp.strftime(time_utils.IB_LOCAL_DATETIME_FORMAT)
    df = get_historical_data_prices(symbol, end_local_date_str, num_of_days_needed, bar_size, True)
    df.to_csv(file_path)
    return df


def query_symbol_list(index_name: str, query=None, queried_column: str = None, return_df: bool = False) -> List[str]:
    """
    find symbols matching query

    :param index_name: index name like "sp500"
    :param query: the query, can be one single str or a list of strings, such as "lmt" or ["Advanced", "apple"]
    :param queried_column: specify which column should be used to match
    :param return_df: indicate whether to return DataFrame or a list of symbols
    :return: list of symbols matching query.
    """
    if index_name != "sp500":
        raise ValueError("For now only sp500 is supported.")
    if isinstance(queried_column, str) and queried_column.lower() == "symbol":
        if isinstance(query, str):
            query = query.upper()
    if query is not None and (not isinstance(query, str) and not isinstance(query, list)):
        raise TypeError("query can only be a str or a list of strings")
    index_symbols_dir = os.path.join(MAIN_RESOURCES_PATH, "index_symbols")
    target_file = os.path.join(index_symbols_dir, index_name + ".csv")
    symbol_df = pd.read_csv(target_file, index_col="#", dtype={
        "#": np.int32,
        "Company": np.str,
        "Symbol": np.str,
        "Weight": np.float32,
        "Price": np.float32,
        "Change": np.float32
    })

    if (queried_column is None and query is not None) or queried_column == "Company":
        def filter_func(name):
            if name is None or not isinstance(name, str):
                return False
            name = name.lower()
            if isinstance(query, str):
                return query.lower() in name
            else:
                return any(q.lower() in name for q in query)
        symbol_df = symbol_df.loc[symbol_df["Company"].apply(filter_func)]

    symbol_df["Symbol"] = symbol_df["Symbol"].apply(lambda val_in_ser: val_in_ser.replace(".", " "))
    return symbol_df if return_df else symbol_df["Symbol"].tolist()


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
