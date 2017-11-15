import pandas
import matplotlib.pyplot as plt
from typing import List

RAW_DATA_PATH = "../rawdata"


def get_max_close(file_name):
    """
    return maximum close price of stock historical data indicated by file name
    :param file_name: file name of csv
    :return: max close price of this csv
    """
    df = pandas.read_csv("../rawdata/{}.csv".format(file_name))
    # df['Volume'].mean()  # calculate mean of column
    return df['Close'].max()


def test_run():
    data_frame = pandas.read_csv("../rawdata/AAPL_20100104-20171013.csv")
    print("data_frame.tail(5)")
    print(data_frame.tail(5))
    print("\ndata_frame.head(5)")
    print(data_frame.head(5))

    print("\ndata_frame[10:15]")
    print(data_frame[10:15])

    print("\nprint max price of AAPL and IBM")
    for file_name in ['AAPL_20100104-20171013', 'IBM_20100104-20171013']:
        print("Max close of {}".format(file_name))
        print(get_max_close(file_name))

    print("\nplot close and adj close")
    data_frame['Adj Close'].plot()
    data_frame[['Close', 'Adj Close']].plot()
    plt.show()


def get_data_from_file_names(file_names: List[str], dates: pandas.DatetimeIndex=None,
                             base_dir="../rawdata") -> pandas.DataFrame:
    """

    :param file_names: list of file names
    :param dates: pandas.DatetimeIndex
    :param base_dir: base dir path
    :return: pandas.DateFrame
    """
    df = None
    if dates is not None:
        df = pandas.DataFrame(index=dates)

    originally_has_spy = False
    for file_name in file_names:
        if "SPY" in file_name.upper():
            originally_has_spy = True
            break
    # print("get_data_from_file_names: originally_has_spy {}".format(originally_has_spy))
    if not originally_has_spy:
        file_names.insert(0, "SPY_20100104-20171013")

    for file_name in file_names:
        df_temp = pandas.read_csv("{}/{}.csv".format(base_dir, file_name), index_col='Date',
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


def plot_data(df: pandas.DataFrame, title="Stock prices", ylabel="Price") -> None:
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


def normalize_data(df: pandas.DataFrame) -> pandas.DataFrame:
    return df / df.iloc[0, :]   # 0th row, every column


def lesson_03_slicing_plotting():
    dates = pandas.date_range('2010-01-04', '2010-12-31')
    # dates = None
    file_names = [
        'SPY_20100104-20171013',
        'GOOG_20100104-20171013',
        'IBM_20100104-20171013',
        'GLD_20100104-20171013'
    ]
    df = get_data_from_file_names(file_names, dates=dates)

    # Slice by row range (dates) using DateFrame.ix[] selector
    # print(df.loc['2010-01-01' : '2010-01-31'])
    #
    # # Slice by column (columns)
    # print(df['GOOG'])
    # print(df[['IBM', 'GLD']])

    # slice by row and column
    print(df.loc['2010-03-10': '2010-03-15', ['SPY', 'IBM']])

    # same footing
    # df_samefoot = df / df[0]
    plot_data(normalize_data(df), "normalized_data")


if __name__ == "__main__":
    lesson_03_slicing_plotting()
