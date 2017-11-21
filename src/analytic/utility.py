import pandas as pd


def get_bars_of_stock(file_name: str, dates: pd.DatetimeIndex=None, base_dir="../rawdata") -> pd.DataFrame:
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


if __name__ == "__main__":
    file = 'amzn_to_2017_11_20.out'
    date_range = pd.date_range(start='20170524 09:30:00', end='20170524 15:59:00', freq="1min")
    my_df = get_bars_of_stock(file_name=file)

    # avg_rt_15 means average return in next 15 days, std_avg_rt_15 means standard deviation of avg_rt_15
    # open high low close volume, normalized_macd_5_15, normalized_macd_signal, normalized ema5, normalized ema15,
    # normalized_volume_ema15
    print(my_df)
