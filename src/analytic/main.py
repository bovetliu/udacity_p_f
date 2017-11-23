from typing import List

import math
import pandas as pd

from analytic import utility
from analytic import ta_indicators

if __name__ == "__main__":
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
