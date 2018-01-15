import collections
import unittest


import matplotlib.transforms as mtransforms
import pandas as pd
from matplotlib import pyplot as plt

from analytic import utility, ta_indicators, statistics
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

    @unittest.skip("just for experimental")
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

    @unittest.skip("just for experimental")
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
        """
        test calculation of rocp sma whether fitts result of pandas calculation
        :return:
        """
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

    # @unittest.skip("just for experimental")
    def test_show_intraday_rtn_trend(self):
        """
        try to see whether stocks supposed to expose to similar context respond similar in major down time
        """
        requested_col = ['time', 'high', 'low', 'open', 'close', 'volume']
        semiconductor_selected = ["MU", "AMAT", "ASML", "KLAC", "LRCX", "INTC", "NVDA", "TXN"]
        file_names = ["{}_2017-05-26-2018-01-05_1_min".format(symbol) for symbol in semiconductor_selected]
        data_frame = utility.get_cols_from_csv_names(file_names=file_names,
                                                     interested_col=requested_col,
                                                     join_spy_for_data_integrity=False,
                                                     keep_spy_if_not_having_spy=False,
                                                     base_dir="../../rawdata")

        def cal_intraday_rtn(open_close_df, *args, **kwargs):
            """

            :param open_close_df: should be a DataFrame having one day worth of open close minute data
            :param args: can be ignored
            :param kwargs: expecting symbol
            :return: introday rtn

            """
            if not isinstance(open_close_df, pd.DataFrame) or len(open_close_df) == 0:
                return None
            if "symbol" not in kwargs:
                raise ValueError("kwargs expecting field \"symbol\"")
            symbol = kwargs["symbol"]
            daily_open = open_close_df["{}_OPEN".format(symbol)].iloc[0]
            intraday_rtn = (open_close_df["{}_CLOSE".format(symbol)] - daily_open) / daily_open
            intraday_rtn.name = "{}_INTRADAY_RTN".format(symbol)
            return intraday_rtn

        def cal_minute_change(open_close_df, *args, **kwargs):
            """
            calculate minute change
            """
            if not isinstance(open_close_df, pd.DataFrame) or len(open_close_df) == 0:
                return None
            if "symbol" not in kwargs:
                raise ValueError("\"symbol\" is expected in field")
            symbol = kwargs["symbol"]
            minute_changes = ta_indicators.get_rocp(open_close_df["{}_CLOSE".format(symbol)],
                                                    window_size='60s',
                                                    name="{}_ROCP_60S".format(symbol),
                                                    expanding=True)
            first_close_in_day = open_close_df["{}_CLOSE".format(symbol)].iloc[0]
            day_open = open_close_df["{}_OPEN".format(symbol)].iloc[0]
            minute_changes.iloc[0] = (first_close_in_day - day_open) / day_open
            # print("cal_minute_change, len(minute_changes): {}".format(len(minute_changes)))
            return minute_changes

        selected_stock = ["AMAT", "ASML", "KLAC", "LRCX"]

        # a map mapping symbol name to pd.Series
        stock_intraday_rtns = collections.OrderedDict({})

        # a map mapping symbol name to pd.Series
        stock_rocp60 = collections.OrderedDict({})

        # a map mapping symbol to scalar values
        stock_min_change_means = collections.OrderedDict({})
        stock_min_change_stds = collections.OrderedDict({})

        for symbol in selected_stock:
            oc_df = data_frame[["{}_OPEN".format(symbol), "{}_CLOSE".format(symbol)]]
            # min bars grouped by days
            grouped_by_days = oc_df.groupby(pd.Grouper(level=0, freq='1B'))
            kwargs = {"symbol": symbol}
            # intraday return trend
            intraday_rtn = grouped_by_days.apply(cal_intraday_rtn, **kwargs).add(1)
            intraday_rtn.name = "{}_INTRADAY_RTN".format(symbol)
            intraday_rtn.dropna(inplace=True)
            print("intraday_rtn.name: {}, len(intraday_rtn): {}, len(data_frame): {}".format(
                intraday_rtn.name,
                len(intraday_rtn),
                len(data_frame)))
            print("put stock_intraday_rtns[{}]".format(symbol))
            stock_intraday_rtns[symbol] = pd.Series(intraday_rtn.values, index=data_frame.index, name=intraday_rtn.name)
            rocp_60s = grouped_by_days.apply(cal_minute_change, **kwargs)
            rocp_60s.name = "{}_ROCP_60S".format(symbol)
            rocp_60s.dropna(inplace=True)
            print("rocp_60s.name: {}, len(rocp_60s): {}, len(data_frame): {}".format(
                rocp_60s.name,
                len(rocp_60s),
                len(data_frame)))
            stock_rocp60[symbol] = pd.Series(rocp_60s.values, index=data_frame.index, name=rocp_60s.name)
            stock_min_change_stds[symbol] = rocp_60s.std()
            stock_min_change_means[symbol] = rocp_60s.mean()

        print("stock_min_change_stds: {}\nstock_min_change_means:{} ".format(stock_min_change_stds, stock_min_change_means))

        #  plotting
        dates = pd.date_range('2017-05-26', '2018-01-05', freq="B")
        for i in range(len(dates)):
            selected_date = dates[i].strftime("%Y-%m-%d")
            print("going to generate pic for %s" % selected_date)
            try:
                if len(stock_intraday_rtns['AMAT'].loc[selected_date]) == 0:
                    continue
                fig, axes = plt.subplots(nrows=2, ncols=1, squeeze=False, sharex="col")
                fig.set_figwidth(16)
                fig.set_figheight(18)
                ylim = [1 - 0.02, 1 + 0.02]
                axes[0, 0].set_xlabel("time")
                axes[0, 0].set_title("INTRADAY_RTN COMPARE {}".format(selected_date))
                legends1 = []
                for symbol, rocp_60 in stock_intraday_rtns.items():
                    rtns = stock_intraday_rtns[symbol].loc[selected_date]
                    ylim[0] = min(ylim[0], rtns.min())
                    ylim[1] = max(ylim[1], rtns.max())
                    legends1.append(rtns.name)
                    axes[0, 0].plot(rtns.index.values, rtns.values, '.-', alpha=0.5)
                axes[0, 0].set_ylim(bottom=ylim[0], top=ylim[1])
                axes[0, 0].legend(legends1)

                # prepare second subplot
                axes[1, 0].set_ylim(bottom=-12, top=0.5)
                axes[1, 0].set_title("MINIMUM Z SCORE COMPARE {}".format(selected_date))
                legends2 = []
                min_z_in_7mins = []
                for symbol, rocp_60 in stock_rocp60.items():
                    rocp60_of_selected = rocp_60.loc[selected_date]
                    kwargs = {
                        "mean2": stock_min_change_means[symbol],
                        "std2": stock_min_change_stds[symbol]
                    }
                    min_z_in_7min = rocp60_of_selected.rolling(window=7, min_periods=1) \
                        .apply(statistics.min_z_score, kwargs=kwargs)
                    min_z_in_7min.name = "{}_MINI_Z_IN_7_MIN".format(symbol)
                    min_z_in_7mins.append(min_z_in_7min)

                    legends2.append(min_z_in_7min.name)
                    axes[1, 0].plot(min_z_in_7min.index.values, min_z_in_7min.values, alpha=0.13)
                mean_z = pd.concat(min_z_in_7mins, axis=1).mean(axis=1)
                mean_z.name = "MEDIAN_Z_IN_7_MIN"
                axes[1, 0].plot(mean_z.index.values, mean_z.values)
                legends2.append(mean_z.name)
                axes[1, 0].legend(legends2)
                plt.xticks(rotation=90)
                plot_name = "two_in_one_{}".format(selected_date)
                plt.savefig("../../pictures/{}.png".format(plot_name))
                plt.close()
            except TypeError as te:
                if "no numeric data to plot" in str(te):
                    plt.close()
                    continue
                else:
                    raise te
        print(statistics.counter_min)
