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


def get_adjclose_from_csv_names(file_names: List[str], dates: pd.DatetimeIndex = None,
                                base_dir="../rawdata") -> pd.DataFrame:
    """

    :param file_names: list of csv file names, without suffix
    :param dates: pandas.DatetimeIndex
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
        df_temp = pd.read_csv("{}/{}.csv".format(base_dir, file_name),
                              index_col='Date',
                              parse_dates=True,
                              usecols=['Date', 'Adj Close'],
                              na_values=['NaN']).rename(columns={'Adj Close': file_name.split('_', 1)[0]})
        if df is None:
            df = df_temp
        else:
            df = df.join(df_temp, how="inner")

    if not originally_has_spy:
        df = df.drop("SPY", axis=1)

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


if __name__ == "__main__":
    csv_files = ['AAPL_20100104-20171013']
    aapl_df = get_adjclose_from_csv_names(csv_files)

    print(aapl_df)
