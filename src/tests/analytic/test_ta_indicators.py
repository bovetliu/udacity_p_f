import unittest

import pandas as pd
import matplotlib.pyplot as plt

from analytic import ta_indicators, utility


class TestTAIndicators(unittest.TestCase):

    def test_get_frws(self):
        series = pd.Series([15, 15, 15, 16, 17, 18, 17, 16, 15, 15.5, 19, 21, 4], name='DEMO')
        future_rtn = ta_indicators.get_frws(series)

        self.assertAlmostEqual(0.0000, future_rtn[0], 4)
        self.assertAlmostEqual(0.0000, future_rtn[1], 4)
        self.assertAlmostEqual(0.0667, future_rtn[2], 4)
        self.assertAlmostEqual(0.0625, future_rtn[3], 4)
        self.assertAlmostEqual(0.0588, future_rtn[4], 4)

    def test_remove_shunhao(self):
        symbol = "UVXY"
        requested_col = ['time', 'high', 'low', 'open', 'close', 'volume']
        data_frame = utility.get_cols_from_csv_names(file_names=[utility.get_appropriate_file(symbol)],
                                                     interested_col=requested_col,
                                                     join_spy_for_data_integrity=False,
                                                     keep_spy_if_not_having_spy=False,
                                                     base_dir="../../rawdata")
        closes = data_frame['{}_CLOSE'.format(symbol)]
        closes_corrected = ta_indicators.remove_shunhao(closes)
        closes_corrected = closes_corrected.groupby(pd.Grouper(level=0, freq="1B")).last()
        closes_corrected = closes_corrected.dropna()
        # closes_corrected.plot(title="UVXY removed shunhao")
        # plt.show()
        print(closes_corrected.tail())

        # at this time vix
        data_frame = utility.get_cols_from_csv_names(file_names=[utility.get_appropriate_file("VIX")],
                                                     interested_col=requested_col,
                                                     join_spy_for_data_integrity=False,
                                                     keep_spy_if_not_having_spy=False,
                                                     base_dir="../../rawdata")

        opens = data_frame["VIX_OPEN"]
        opens_sma = ta_indicators.get_rolling_mean(opens, window_size=100)
        ax = opens.plot(title="corrected uvxy")
        opens_sma.plot(ax=ax)
        plt.show()

        # uvxy_daily_open = data_frame["UVXY_OPEN"]
        # uvxy_daily_open_sma = ta_indicators.get_rolling_mean(uvxy_daily_open, window_size=20)
        # ax = uvxy_daily_open.plot(title="uvxy daily open")
        # uvxy_daily_open_sma.plot(ax=ax)
        # plt.show()
