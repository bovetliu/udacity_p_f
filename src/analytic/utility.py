from typing import List

import matplotlib.pyplot as plt
import pandas as pd


def get_bars_of_stock(file_name: str, dates: pd.DatetimeIndex = None, base_dir="../rawdata") -> pd.DataFrame:
    """
    utility method of get bars of one stock
    :param file_name: file name
    :param dates: a DatetimeIndex of minutes
    :param base_dir: base directory holding data file
    :return: a data frame indexed by date time at minute granularity, columns are OHLCV
    """
    df = None
    if dates is not None:
        df = pd.DataFrame(index=dates)
    df_temp = pd.read_table(filepath_or_buffer="{}/{}".format(base_dir, file_name), sep='\t',
                            index_col='time',
                            parse_dates=[0],
                            date_parser=lambda date: pd.datetime.strptime(date, '%Y%m%d %H:%M:%S'),
                            usecols=['time', 'open', 'high', 'low', 'close', 'volume'])

    if df is None:
        df = df_temp
    else:
        df = df.join(df_temp, how="inner")
    return df


def get_cols_from_csv_names(file_names: List[str],
                            dates: pd.DatetimeIndex = None,
                            interested_col: List[str] = ('Date', 'Adj Close', 'Volume'),
                            join_spy_for_data_integrity=True,
                            keep_spy_if_not_having_spy=True,
                            base_dir="../rawdata") -> pd.DataFrame:
    """

    :param file_names: list of csv file names, without suffix
    :param dates: pandas.DatetimeIndex
    :param interested_col: interested columns
    :param join_spy_for_data_integrity will join spy for data integrity
    :param keep_spy_if_not_having_spy join spy into data frame or not
    :param base_dir: base dir path
    :return: pandas.DateFrame
    """
    df = None
    if dates is not None:
        df = pd.DataFrame(index=dates)

    originally_has_spy = False
    for file_name in file_names:
        if "SPY" in file_name.upper():
            originally_has_spy = True
            break
    # print("get_data_from_file_names: originally_has_spy {}".format(originally_has_spy))
    if not originally_has_spy:
        if join_spy_for_data_integrity:
            file_names.insert(0, "SPY_20100104-20171013")

    for file_name in file_names:
        temp_symbol = file_name.split('_', 1)[0]
        col_rename_map = {
            'Adj Close': "{}_ADJ_CLOSE".format(temp_symbol),
            'close': "{}_CLOSE".format(temp_symbol),
            'Close': "{}_CLOSE".format(temp_symbol),
            'open': "{}_OPEN".format(temp_symbol),
            'Open': "{}_OPEN".format(temp_symbol),
            'high': "{}_HIGH".format(temp_symbol),
            'High': "{}_HIGH".format(temp_symbol),
            'low': "{}_LOW".format(temp_symbol),
            'Low': "{}_LOW".format(temp_symbol),
            'time': 'Date',
            'volume': "{}_VOLUME".format(temp_symbol),
            'Volume': "{}_VOLUME".format(temp_symbol),
        }
        df_temp = pd.read_csv("{}/{}.csv".format(base_dir, file_name),
                              index_col='Date' if 'Date' in interested_col else 'time',
                              parse_dates=True,
                              usecols=interested_col,
                              na_values=['NaN']).rename(columns=col_rename_map)
        if df is None:
            df = df_temp
        else:
            df = df.join(df_temp, how="inner")
    if not originally_has_spy and not keep_spy_if_not_having_spy:
        for column_name in df:
            if 'SPY' in column_name:
                df.drop([column_name], axis=1, inplace=True)
    return df


def plot_data(df, title="Stock prices", ylabel="Price") -> None:
    """
    Plot stock prices
    :param df: data frame, not Null, not necessarily data frame, pandas Series is also okay
    :param title: title of the plot
    :param ylabel: label of y axis
    :return: None
    """
    axes_sub_plot = df.plot(title=title)
    axes_sub_plot.set_xlabel("Date")
    axes_sub_plot.set_ylabel(ylabel)

    plt.show(block=True)


def rescale(in_ser: pd.Series, shift_so_zero_mean: bool=False, name: str=None):
    """
    rescale a pandas series so 2 * its standard deviation = 0.9545. its mean equals to 0
    :param in_ser:
    :param shift_so_zero_mean: should vertically shift series so it have a 0 mean?
    :param name: new name of returned series
    :return:
    """
    if not isinstance(in_ser, pd.Series):
        raise TypeError("in_ser should be pandas Series")
    std = in_ser.std()
    x = 0.9545 / 2 / std

    stretched = in_ser.mul(x)

    tbr = stretched - stretched.mean() if shift_so_zero_mean else stretched
    tbr.rename(name if name else "RESCALED_{}".format(in_ser.name), inplace=True)
    return tbr


def get_appropriate_file(symbol):
    map = {
        "AMD": "AMD_20151224-20171222",
        "AAPL": "AAPL_20121114_20171114",
        "EWA": "EWA_20060103_20171228",
        "EWC": "EWC_20060103_20171228",
        "QQQ": "QQQ_2003-01-06_2017-11-28",
        "SPY": "SPY_20090102-20171017",
        "XOM": "XOM_20100104-20171013",
        "IBM": "IBM_20100104-20171013",
        "UVXY": "UVXY_20111223-20171222"
    }

    semiconductor_selected = ["MU", "TSM", "AMAT", "ASML",
                              "KLAC", "LRCX", "INTC", "NVDA",
                              "TXN"]
    for symbol in semiconductor_selected:
        map[symbol] = "{}_2017-05-26-2018-01-05_1_min".format(symbol)
    return map[symbol]
