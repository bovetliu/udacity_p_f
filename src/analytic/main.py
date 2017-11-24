import math
import matplotlib.pyplot as plt
import pandas as pd

from analytic import utility
from analytic import ta_indicators


def practice_01():
    window_size = 20
    csv_files = ['AAPL_20100104-20171013']
    aapl_df = utility.get_adjclose_from_csv_names(csv_files)
    # utility.plot_data(aapl_df)
    aapl_series = aapl_df['AAPL']
    aapl_simple_mean = ta_indicators.get_rolling_mean(aapl_series, window_size)
    aapl_std = ta_indicators.get_rolling_std(aapl_series, window_size)

    rel_pos_bb = ta_indicators.get_window_normalized(aapl_series, window_size, 2)
    rescaled_volatility = utility.rescale(aapl_std / aapl_simple_mean)
    rescaled_volatility.name = 'AAPL_VOLATILITY'
    aapl_simple_mean_drtn_rescaled = utility.rescale(ta_indicators.get_daily_return(aapl_simple_mean))
    vol_window_normlzd = ta_indicators.get_window_normalized(aapl_df['AAPL_VOL'], window_size)

    momentum_rescaled = utility.rescale(ta_indicators.get_momentum(aapl_series, 20))

    # following are labels, without rescaling
    future_returns = ta_indicators.get_frws(aapl_series)
    # print(rel_pos_bb)
    combined_df = pd.concat([rel_pos_bb,
                             rescaled_volatility,
                             aapl_simple_mean_drtn_rescaled,
                             vol_window_normlzd,
                             momentum_rescaled,
                             future_returns], axis=1)
    combined_df.plot()
    plt.show(block=True)


if __name__ == "__main__":
    practice_01()

