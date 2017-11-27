from typing import List

import pandas as pd
import matplotlib.pyplot as plt


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
                            keep_spy_if_not_having_spy=True,
                            base_dir="../rawdata") -> pd.DataFrame:
    """

    :param file_names: list of csv file names, without suffix
    :param dates: pandas.DatetimeIndex
    :param interested_col: interested columns
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
        file_names.insert(0, "SPY_20100104-20171013")

    for file_name in file_names:
        temp_symbol = file_name.split('_', 1)[0]
        col_rename_map = {
            'Adj Close': "{}_ADJ_CLOSE".format(temp_symbol),
            'Close': "{}_CLOSE".format(temp_symbol),
            'Open': "{}_OPEN".format(temp_symbol),
            'High': "{}_HIGH".format(temp_symbol),
            'Low': "{}_LOW".format(temp_symbol),
            'Volume': "{}_VOLUME".format(temp_symbol),
        }
        df_temp = pd.read_csv("{}/{}.csv".format(base_dir, file_name),
                              index_col='Date',
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
    :param df: data frame, not Null
    :param title: title of the plot
    :param ylabel: label of y axis
    :return: None
    """
    axes_sub_plot = df.plot(title=title)
    axes_sub_plot.set_xlabel("Date")
    axes_sub_plot.set_ylabel(ylabel)
    # axes_sub_plot.set_xlabel("Date")
    # axes_sub_plot.set_ylable("Price") no longer works

    plt.show(block=True)


def get_relative_net_worth(data_frame: pd.DataFrame, symbol: str) -> pd.Series:
    """
    get relative net worth pandas series
    :param data_frame: data frame
    :param symbol: stock symbol
    :return: pandas series recording net worth change along bars and signals
    """

    if not isinstance(data_frame, pd.DataFrame):
        raise TypeError("data_frame can only be pd.DataFrame")
    if not symbol or not isinstance(symbol, str):
        raise TypeError("symbol not valid.")
    net_worth = 1.0
    current_position = 0  # 0 means emtpy, 1 means long, -1 means short
    prev_price = 0

    pending_action = 0
    tbr = pd.Series(data=[], index=[])
    for index, row in data_frame.iterrows():
        cur_price = row['{}_OPEN'.format(symbol)]
        # following is handling pening action at open of each bar
        if pending_action is not None and pending_action != current_position:
            # close position first
            if current_position != 0:
                ratio = (cur_price / prev_price) if prev_price != 0 else 1
                net_worth = ratio * net_worth if current_position > 0 else net_worth / ratio
            # simulate order_target_percent
            if pending_action in (-1, 0, 1):
                current_position = pending_action
            else:
                raise ValueError("Unrecognized sig value {}".format(pending_action))
            # prev_price always record a check point
            prev_price = cur_price

        # end of market opening, now it is end of market
        cur_sig = row['{}_SIGNAL'.format(symbol)]
        cur_price = row['{}_CLOSE'.format(symbol)]
        if cur_sig != current_position:
            pending_action = cur_sig
        else:
            pending_action = None
        # update net worth at end of each trading day
        if current_position != 0:
            ratio = (cur_price / prev_price) if prev_price != 0 else 1
            net_worth = ratio * net_worth if current_position > 0 else net_worth / ratio

        # https://github.com/pandas-dev/pandas/issues/2801  the contributor closed pandas inplace appending series.
        tbr = tbr.append(pd.Series(data=[net_worth], index=[index]))
        # after market close
        prev_price = cur_price
    return tbr





def rescale(in_ser: pd.Series):
    """
    rescale a pandas series so 2 * its standard deviation = 0.9545. its mean equals to 0
    :param in_ser:
    :return:
    """
    if not isinstance(in_ser, pd.Series):
        raise TypeError("in_ser should be pandas Series")
    std = in_ser.std()
    x = 0.9545 / 2 / std

    stretched = in_ser.mul(x)
    return stretched - stretched.mean()


if __name__ == "__main__":
    pass
