from typing import List

import math
import pandas as pd

from analytic import utility
from analytic import ta_indicators

def practice_01():
    csv_files = ['AAPL_20100104-20171013']
    aapl_df = utility.get_adjclose_from_csv_names(csv_files)
    aapl_sr = aapl_df['AAPL']
    # appl ema5 pandas series
    aapl_ema05 = ta_indicators.get_ema(aapl_sr, 5)
    # aapl ema15 pandas series
    aapl_ema15 = ta_indicators.get_ema(aapl_sr, 15)
    aapl_ema0515_diff = aapl_ema05 - aapl_ema15
    print(aapl_ema0515_diff)
    assert (not math.isnan(aapl_ema0515_diff.loc['2010-01-04']))
    aapl_ema0515_norm = ta_indicators.get_normalized(aapl_ema0515_diff, 15, 'aapl_0515_norm')
    utility.plot_data(aapl_ema0515_norm, title=aapl_ema0515_norm.name, ylabel=aapl_ema0515_norm.name)


def demo_future_return_calc():
    series = pd.Series([15, 15, 15, 16, 17, 18, 17, 16, 15, 15.5, 19, 21, 4])
    # moment_series = ta_indicators.get_momentum(series, 2)
    # print(moment_series)

    daily_return = (series - series.shift(1)) / series.shift(1)
    print(daily_return)
    daily_return_reversed = pd.Series(daily_return).reindex(daily_return.index[::-1])
    daily_return_reversed_sum = daily_return_reversed.rolling(window=3).sum()
    print(daily_return_reversed_sum)
    future_window_daily_return_sum = daily_return_reversed_sum.reindex(daily_return_reversed_sum.index[::-1])
    print(future_window_daily_return_sum.shift(-1))

if __name__ == "__main__":
    demo_future_return_calc()

