import unittest

from analytic import utility, ta_indicators
from analytic.strategies.AvoidSlump import AvoidSlump

import pandas as pd

import matplotlib.transforms as mtransforms
from matplotlib import pyplot as plt


class TestStrategies(unittest.TestCase):

    def test_one_day(self):
        see_pic = True
        symbol_name = "AMAT"
        requested_col = ['time', 'high', 'low', 'open', 'close', 'volume']
        data_frame = utility.get_cols_from_csv_names(file_names=['AMAT_to_2018-01-05'],
                                                     interested_col=requested_col,
                                                     join_spy_for_data_integrity=False,
                                                     keep_spy_if_not_having_spy=False,
                                                     base_dir="../../rawdata")
        selected_df = data_frame['2017-09-27']
        avoid_slump_run = AvoidSlump(symbol_name, selected_df, starting_cash=15000)
        avoid_slump_run.start()
        if see_pic:
            closes = selected_df['{}_CLOSE'.format(symbol_name)]
            ax = closes.plot(title="avoid slump strategy",
                             legend=True, figsize=(12, 7),
                             ylim=(closes.min() - 0.5, closes.max() + 0.5))
            ma = ta_indicators.get_rolling_mean(closes, avoid_slump_run.sma_window)
            ma.plot(ax=ax, legend=True)
            zhishun_line_pdser = pd.Series(avoid_slump_run.zhishun_line_befei, selected_df.index)
            zhishun_line_pdser.name = "zhishun_line"
            zhishun_line_pdser.plot(ax=ax, legend=True)

            dail_loss_control = pd.Series(avoid_slump_run.daily_loss_control_beifen,
                                          selected_df.index,
                                          name="daily_loss_control")
            dail_loss_control.plot(ax=ax, legend=True)
            trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
            ax.fill_between(closes.index, 0, 1, where=(avoid_slump_run.positions <= 0).values,
                            facecolors="red",
                            alpha=0.2,
                            transform=trans)
            plt.show()

        back_test_res_df = avoid_slump_run.generate_report()
        intraday_effect = back_test_res_df['INTRADAY_RTN'] - back_test_res_df['INTRADAY_{}_RTN'.format(symbol_name)]
        back_test_res_df = back_test_res_df.assign(intraday_effect=intraday_effect)
        print(back_test_res_df['intraday_effect'])

    def test_avoid_slump(self):
        symbol_name = "AMAT"
        requested_col = ['time', 'high', 'low', 'open', 'close', 'volume']
        data_frame = utility.get_cols_from_csv_names(file_names=['AMAT_to_2018-01-05'],
                                                     interested_col=requested_col,
                                                     join_spy_for_data_integrity=False,
                                                     keep_spy_if_not_having_spy=False,
                                                     base_dir="../../rawdata")

        do_actual_comparison = False
        see_pic = False

        # selected_df = data_frame.loc['2017-08-18': '2017-08-31']
        selected_df = data_frame
        avoid_slump_run = AvoidSlump(symbol_name, selected_df, starting_cash=15000)
        avoid_slump_run.start()
        # print(avoid_slump_run.positions.head(100))

        if see_pic:
            closes = selected_df['{}_CLOSE'.format(symbol_name)]
            ax = closes.plot(title="avoid slump strategy",
                             legend=True, figsize=(12, 7),
                             ylim=(closes.min() - 0.5, closes.max() + 0.5))
            ma = ta_indicators.get_rolling_mean(closes, avoid_slump_run.sma_window)
            ma.plot(ax=ax, legend=True)
            zhishun_line_pdser = pd.Series(avoid_slump_run.zhishun_line_befei, selected_df.index)
            zhishun_line_pdser.name = "zhishun_line"
            zhishun_line_pdser.plot(ax=ax, legend=True)
            trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
            ax.fill_between(closes.index, 0, 1, where=(avoid_slump_run.positions <= 0).values,
                            facecolors="red",
                            alpha=0.2,
                            transform=trans)
            plt.show()
        back_test_res_df = avoid_slump_run.generate_report()

        if do_actual_comparison:
            self.assertAlmostEqual(back_test_res_df['INTRADAY_{}_RTN'.format(symbol_name)].iloc[0], -0.010498, 6)
            self.assertAlmostEqual(back_test_res_df['INTRADAY_{}_RTN'.format(symbol_name)].iloc[1], -0.018808, 6)
        intraday_effect = back_test_res_df['INTRADAY_RTN'] - back_test_res_df['INTRADAY_{}_RTN'.format(symbol_name)]
        back_test_res_df = back_test_res_df.assign(intraday_effect=intraday_effect)
        #  did back come fist
        col_names = ["time", "intraday_effect", "INTRADAY_RTN", "INTRADAY_{}_RTN".format(symbol_name)]
        back_test_res_df.sort_values(by="intraday_effect", inplace=True)
        print("{}                 {}   {}   {}".format(*col_names))

        for i in range(len(back_test_res_df)):

            print("{0}  {1:9.6f}%  {2:9.6f}%  {3:9.6f}%"
                  .format(back_test_res_df.index[i], back_test_res_df[col_names[1]].iloc[i] * 100,
                          back_test_res_df[col_names[2]].iloc[i] * 100,
                          back_test_res_df[col_names[3]].iloc[i] * 100))

        print("\nintraday_effect.mean(): {0:9.6f}, intraday_effect.std(): {1:9.6f}".format(
            back_test_res_df[col_names[1]].mean(), back_test_res_df[col_names[1]].std()))


    def test_calc_rocp_of_sma(self):
        requested_col = ['time', 'high', 'low', 'open', 'close', 'volume']
        data_frame = utility.get_cols_from_csv_names(file_names=['AMAT_to_2018-01-05'],
                                                     interested_col=requested_col,
                                                     join_spy_for_data_integrity=False,
                                                     keep_spy_if_not_having_spy=False,
                                                     base_dir="../../rawdata")

        selected_df = data_frame.loc['2017-08-18']
        closes = selected_df['AMAT_CLOSE']
        ma_expected = closes.rolling(window=7, min_periods=1).mean()
        rocp_ma_expected = ta_indicators.get_rocp(ma_expected, window_size='60s')
        temp_list = []
        for i in range(len(closes)):
            temp_list.append(closes.iloc[i])
            rocp = AvoidSlump.calc_rocp_of_sma(temp_list, window=7)
            self.assertAlmostEqual(rocp_ma_expected.iloc[i], rocp, 6)

