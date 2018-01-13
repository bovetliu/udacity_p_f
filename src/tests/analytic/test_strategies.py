import unittest

import matplotlib.transforms as mtransforms
import pandas as pd
from matplotlib import pyplot as plt

from analytic import utility, ta_indicators
from analytic.strategies.AvoidSlump import AvoidSlump


class TestStrategies(unittest.TestCase):

    @staticmethod
    def trend_pic(avoid_slump_run: AvoidSlump, date):

        closes = avoid_slump_run.hist_data[date][avoid_slump_run.col_dict['close']]
        ax = closes.plot(title="avoid slump strategy {} {}".format(avoid_slump_run.symbol_name, date),
                         legend=True, figsize=(12, 7),
                         ylim=(closes.min() - 0.5, max(closes.min() * 1.03, closes.max())), style="c.-")
        ma = ta_indicators.get_rolling_mean(closes, avoid_slump_run.sma_window)
        ma.plot(ax=ax, legend=True)
        zhishun_line_pdser = pd.Series(avoid_slump_run.zhishun_line_befei, closes.index)
        zhishun_line_pdser.name = "zhishun_line"
        zhishun_line_pdser.plot(ax=ax, legend=True)
        trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
        ax.fill_between(closes.index, 0, 1, where=(avoid_slump_run.positions <= 0).values,
                        facecolors="red",
                        alpha=0.2,
                        transform=trans)
        plot_name = "{}_TREND_{}.png".format(avoid_slump_run.symbol_name, date)
        plt.savefig("../../quantopian_algs_backup/{}".format(plot_name))
        plt.close()
        # plt.show()

    @staticmethod
    def rtn_compare_pic(avoid_slump_run: AvoidSlump, date):
        closes = avoid_slump_run.hist_data[date][avoid_slump_run.col_dict['close']]
        closes_rtn = ta_indicators.get_rocp(closes, 2).add(1).cumprod()
        closes_rtn.name = 'closes_rtn'
        totals = avoid_slump_run.totals
        totals_rtn = ta_indicators.get_rocp(totals, 2).add(1).cumprod()
        totals_rtn.name = "totals_rtn"
        ylim = (min(0.97, min(closes_rtn.min(), totals_rtn.min())), max(1.03, closes_rtn.max()))
        ax = closes_rtn.plot(legend=True, figsize=(12, 7),
                             title="B-N-H VS. AVOID SLUMP {} {}".format(avoid_slump_run.symbol_name, date),
                             ylim=ylim)
        totals_rtn.plot(ax=ax, legend=True)

        plot_name = "{}_RTN_COMPARE_{}.png".format(avoid_slump_run.symbol_name, date)
        plt.savefig("../../quantopian_algs_backup/{}".format(plot_name))
        plt.close()
        # plt.show()

    def test_save_figs(self):
        see_pic = True
        see_rtn_compare = True
        symbol_name = "AMAT"
        requested_col = ['time', 'high', 'low', 'open', 'close', 'volume']
        data_frame = utility.get_cols_from_csv_names(file_names=['AMAT_to_2018-01-05'],
                                                     interested_col=requested_col,
                                                     join_spy_for_data_integrity=False,
                                                     keep_spy_if_not_having_spy=False,
                                                     base_dir="../../rawdata")
        selected_dates = ["2017-06-27", "2017-11-02", "2017-07-07", "2017-07-05", "2017-11-20",
                          "2017-10-26", "2017-08-11", "2017-10-03", "2017-09-18", "2017-08-08",
                          "2017-12-18", "2017-09-27", "2017-06-28", "2017-07-03", "2017-07-11",
                          "2018-01-03", "2017-11-08", "2017-11-27", "2017-12-15", "2017-07-17"]
        # selected_dates = ["2017-07-03"]
        for selected_date in selected_dates:
            selected_df = data_frame[selected_date]
            avoid_slump_run = AvoidSlump(symbol_name, selected_df, starting_cash=15000)
            avoid_slump_run.start()
            if see_pic:
                self.trend_pic(avoid_slump_run, selected_date)

            if see_rtn_compare:
                self.rtn_compare_pic(avoid_slump_run, selected_date)

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
                             legend=True, figsize=(12, 7), style="c.-",
                             ylim=(closes.min() - 0.5, closes.max() + 0.5))
            zhishun_line_pdser = pd.Series(avoid_slump_run.zhishun_line_befei, selected_df.index)
            zhishun_line_pdser.name = "zhishun_line"
            zhishun_line_pdser.plot(ax=ax, legend=True)
            ma = ta_indicators.get_rolling_mean(closes, avoid_slump_run.sma_window)
            ma.plot(ax=ax, legend=True)
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

