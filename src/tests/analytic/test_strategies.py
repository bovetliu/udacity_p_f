import unittest

from analytic import utility, ta_indicators
from analytic.strategies.AvoidSlump import AvoidSlump

import pandas as pd

import matplotlib.transforms as mtransforms
from matplotlib import pyplot as plt


class TestStrategies(unittest.TestCase):

    def test_avoid_slump(self):
        requested_col = ['time', 'high', 'low', 'open', 'close', 'volume']
        data_frame = utility.get_cols_from_csv_names(file_names=['AMAT_to_2018-01-05'],
                                                     interested_col=requested_col,
                                                     join_spy_for_data_integrity=False,
                                                     keep_spy_if_not_having_spy=False,
                                                     base_dir="../../rawdata")

        selected_df = data_frame.loc['2017-08-18']
        avoid_slump_run = AvoidSlump("AMAT", selected_df, starting_cash=15000)
        avoid_slump_run.start()
        # print(avoid_slump_run.positions.head(100))

        closes = selected_df['AMAT_CLOSE']
        ax = closes.plot(title="avoid slump strategy",
                         legend=True, figsize=(12, 7),
                         ylim=(closes.min() - 0.5, closes.max() + 0.5))
        ma = ta_indicators.get_rolling_mean(closes, avoid_slump_run.sma_window)
        ma.plot(ax=ax, legend=True)
        zhishun_line_pdser = pd.Series(avoid_slump_run.zhishun_line, selected_df.index)
        zhishun_line_pdser.name = "zhishun_line"
        zhishun_line_pdser.plot(ax=ax, legend=True)
        trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
        ax.fill_between(closes.index, 0, 1, where=(avoid_slump_run.positions <= 0).values,
                        facecolors="red",
                        alpha=0.2,
                        transform=trans)
        plt.show()
