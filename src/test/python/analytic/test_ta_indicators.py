import unittest

import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from analytic import ta_indicators, utility
from analytic.data import tws_data_api_connector


class TestTAIndicators(unittest.TestCase):

    def test_calculate_ema(self):
        closes = np.ndarray(shape=(100, 3), dtype=float)
        period = 30
        k = 2.0 / (period + 1.0)
        print(k)

        print(closes.shape)
        self.assertEqual((100, 3), closes.shape)
        print(closes)

        # delete last row

        deleted_earliest = closes[2:]
        print(deleted_earliest)
        self.assertEqual((len(closes) - 2, 3), deleted_earliest.shape)
        # numpy.ma.average(a, axis=None, weights=None, returned=False)[source]
        weights = [k * ((1 - k) ** i) for i in reversed(range(0, len(deleted_earliest)))]
        print(weights)
        print(len(weights))
        ema = np.ma.average(deleted_earliest, axis=0, weights=weights, returned=False)
        ema = ema + closes[0] * (1 - k) ** len(closes)
        print(ema)

        for i in reversed(range(2, 10)):
            print(i)

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

    def test_get_3_ema_fenli_plot(self):
        symbol = "NVDA"
        nvda_df = tws_data_api_connector.get_local_data(symbol)
        nvda_closes = nvda_df["m_close"]
        window_sizes = [20, 35, 50]
        fenli = ta_indicators.get_3_ema_fenli(nvda_closes, window_sizes)

        fig, ax1 = plt.subplots()
        nvda_closes.plot(title="nvda price vs fenli", ax=ax1, style='b-')
        ax1.set_xlabel('date')
        ax1.set_ylabel('price')

        ax2 = ax1.twinx()
        fenli.plot(ax=ax2, style='c-')
        ax2.set_ylabel('fenli')
        fig.tight_layout()
        plt.show()

    def test_get_3_ema_fenli(self):
        symbol = "NVDA"
        nvda_df = tws_data_api_connector.get_local_data(symbol)
        nvda_closes = nvda_df["m_close"]
        window_sizes = [20, 35, 50]
        fenli = ta_indicators.get_3_ema_fenli(nvda_closes, window_sizes)
        print(fenli)

    def test_get_draw_down(self):
        symbol = "LMT"
        tws_data_api_connector.get_local_synced(symbol)
        nvda_df = tws_data_api_connector.get_local_data(symbol)
        nvda_df_closes = nvda_df["m_close"]
        nvda_df_closes = nvda_df_closes.iloc[-272:]
        nvda_df_closes.name = "{}_close".format(symbol)

        plt.figure(1, figsize=(8, 16))
        plt.subplot(4, 1, 1)
        plt.plot(nvda_df_closes.index, nvda_df_closes.values, 'g-')
        plt.xlabel('date')
        plt.ylabel('price')

        plt.subplot(4, 1, 2)
        window = None
        drawdown = ta_indicators.get_draw_down(nvda_df_closes, window=window)
        reversed_drawdown = ta_indicators.get_reversed_draw_down(nvda_df_closes, window=window)

        line_drawdown = plt.plot(drawdown.index, drawdown.values, 'g-')
        line_reversed_drawdown = plt.plot(reversed_drawdown.index, reversed_drawdown.values, 'r-')
        plt.setp(line_reversed_drawdown, alpha=.5)
        plt.xlabel('date')
        plt.ylabel('ratio')
        plt.legend((drawdown.name, reversed_drawdown.name), loc='lower left')

        plt.subplot(4, 1, 3)
        window = 100
        drawdown = ta_indicators.get_draw_down(nvda_df_closes, window=window)
        reversed_drawdown = ta_indicators.get_reversed_draw_down(nvda_df_closes, window=window)
        line_drawdown = plt.plot(drawdown.index, drawdown.values, 'g-')
        line_reversed_drawdown = plt.plot(reversed_drawdown.index, reversed_drawdown.values, 'r-')
        plt.setp(line_reversed_drawdown, alpha=.5)
        plt.xlabel('date')
        plt.ylabel('ratio')
        plt.legend((drawdown.name, reversed_drawdown.name), loc='lower left')

        plt.subplot(4, 1, 4)
        window = 50
        drawdown = ta_indicators.get_draw_down(nvda_df_closes, window=window)
        reversed_drawdown = ta_indicators.get_reversed_draw_down(nvda_df_closes, window=window)
        line_drawdown = plt.plot(drawdown.index, drawdown.values, 'g-')
        line_reversed_drawdown = plt.plot(reversed_drawdown.index, reversed_drawdown.values, 'r-')
        plt.setp(line_reversed_drawdown, alpha=.5)
        plt.xlabel('date')
        plt.ylabel('ratio')
        plt.legend((drawdown.name, reversed_drawdown.name), loc='lower left')
        plt.savefig("/home/boweiliu/workrepo/udacity_p_f_pics/drawdown/drawdown_{}_{}.png".format
                    (symbol, datetime.datetime.now().date().isoformat()),
                    bbox_inches='tight')
        # plt.show()
        # print(reversed_drawdown)
