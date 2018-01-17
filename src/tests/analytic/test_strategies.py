import collections
import unittest


import matplotlib.transforms as mtransforms
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

from analytic import utility, ta_indicators, statistics
from analytic.strategies.AvoidSlump import AvoidSlump


class TestStrategies(unittest.TestCase):

    @staticmethod
    def trend_z_pic(avoid_slump_run: AvoidSlump, date, z_threshold, ax):
        closes = avoid_slump_run.hist_data[date][avoid_slump_run.col_dict['close']]
        z_s = pd.Series(avoid_slump_run.z_s_of_day, index=closes.index)
        z_s.name = "z_s"
        z_s.plot(ax=ax, title="z {} {}".format(avoid_slump_run.symbol_name, date), legend=True, ylim=(-10, 1))
        threshold_line = np.multiply(np.ones((len(closes))), z_threshold)
        threshold_line = pd.Series(threshold_line, index=closes.index)
        threshold_line.name = "z_th={}".format(z_threshold)
        threshold_line.plot(ax=ax, legend=True)
        # ta_indicators.get_rolling_mean(z_s, window_size=7).plot(ax=ax, legend=True)

    @staticmethod
    def trend_pic(avoid_slump_run: AvoidSlump, date, ax):

        closes = avoid_slump_run.hist_data[date][avoid_slump_run.col_dict['close']]
        closes.plot(ax=ax, title="avoid slump strategy {} {}".format(avoid_slump_run.symbol_name, date),
                    legend=True,
                    ylim=(closes.min() - 0.5, max(closes.min() * 1.03, closes.max())), style="c.-")
        ma = ta_indicators.get_rolling_mean(closes, avoid_slump_run.sma_window)
        ma.plot(ax=ax, legend=True)
        zhishun_line_pdser = pd.Series(avoid_slump_run.zhishun_line_befei, closes.index)
        zhishun_line_pdser.name = "zhishun_line"
        zhishun_line_pdser.plot(ax=ax, legend=True)
        trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
        where_in_closed_position = avoid_slump_run.positions <= 0
        shifted_position = avoid_slump_run.positions.shift(1)
        where_starts_open = (shifted_position <= 0) & (0 < avoid_slump_run.positions)
        where_in_closed_position = where_in_closed_position | where_starts_open
        ax.fill_between(closes.index, 0, 1, where=where_in_closed_position.values,
                        facecolors="red",
                        alpha=0.2,
                        transform=trans)

    @staticmethod
    def rtn_compare_pic(avoid_slump_run: AvoidSlump, date, ax):
        closes = avoid_slump_run.hist_data[date][avoid_slump_run.col_dict['close']]
        closes_rtn = ta_indicators.get_rocp(closes, 2).add(1).cumprod()
        closes_rtn.name = 'closes_rtn'
        totals = avoid_slump_run.totals
        totals_rtn = ta_indicators.get_rocp(totals, 2).add(1).cumprod()
        totals_rtn.name = "totals_rtn"
        ylim = (min(0.97, min(closes_rtn.min(), totals_rtn.min())), max(1.03, closes_rtn.max()))
        closes_rtn.plot(ax=ax, legend=True,
                        title="B-N-H VS. AVOID SLUMP {} {}".format(avoid_slump_run.symbol_name, date),
                        ylim=ylim)
        totals_rtn.plot(ax=ax, legend=True)

        # plot_name = "{}_RTN_COMPARE_{}.png".format(avoid_slump_run.symbol_name, date)
        # plt.savefig("../../quantopian_algs_backup/{}".format(plot_name))
        # plt.close()
        # plt.show()

    @staticmethod
    def provide_avoid_slump(symbol_name, historical_data, zscore_threshold=0.0, buffer=0.004):
        """
        provide avoid slump with same parameters
        """
        return AvoidSlump(symbol_name, hist_data=historical_data,
                          starting_cash=15000,
                          rocp_mean=-1.2456320934141318e-06,
                          rocp_std=0.00084797841463923234,
                          zscore_threshold=zscore_threshold,
                          sma_threshold=0,
                          buffer=buffer)

    # @unittest.skip("just for experimental")
    def test_save_figs(self):
        save_rtn_compare = True
        save_trend_pic = True
        save_z_trend=True

        symbol_name = "AMAT"
        requested_col = ['time', 'high', 'low', 'open', 'close', 'volume']
        data_frame = utility.get_cols_from_csv_names(file_names=[utility.get_appropriate_file(symbol_name)],
                                                     interested_col=requested_col,
                                                     join_spy_for_data_integrity=False,
                                                     keep_spy_if_not_having_spy=False,
                                                     base_dir="../../rawdata")

        # selected_dates = ['2017-06-29']
        selected_dates = ['2017-11-20', '2017-07-07', '2017-06-13', '2017-06-29', '2017-09-18', '2017-10-26', '2017-06-27', '2017-08-25', '2017-08-11', '2017-06-21', '2017-10-09', '2017-12-15', '2017-08-14', '2017-10-31', '2017-07-17', '2017-07-19', '2017-08-04', '2017-10-16', '2017-07-03', '2017-12-19', '2017-08-28', '2017-08-22', '2017-06-19', '2017-11-24', '2017-09-11', '2017-08-07', '2017-12-18', '2017-12-13', '2017-10-20', '2017-09-15', '2017-11-21', '2017-08-30', '2017-10-06', '2017-11-08', '2017-06-05', '2017-07-05', '2017-09-27', '2017-06-07', '2017-10-13', '2017-12-12', '2017-12-29', '2017-08-08', '2017-07-21', '2017-09-06', '2017-07-06', '2017-11-15', '2017-08-23', '2017-07-26', '2017-06-28', '2017-09-07', '2017-11-10', '2017-06-12', '2017-09-08', '2017-11-06', '2017-06-06', '2017-07-14', '2017-09-01', '2017-08-31', '2017-10-03', '2017-11-02', '2017-12-11', '2017-09-22', '2017-12-28', '2017-08-15', '2017-06-20', '2017-11-03', '2017-12-08', '2017-10-11', '2017-08-29', '2017-10-19', '2017-06-22', '2017-05-26', '2017-12-14', '2017-06-23', '2017-06-16', '2017-11-13', '2017-07-13', '2017-10-24', '2017-06-02', '2017-06-01', '2017-05-31', '2017-10-02', '2017-11-27', '2017-08-03', '2017-11-28', '2017-11-07', '2017-07-20', '2017-12-22', '2017-07-24', '2017-07-11', '2017-09-19', '2017-09-28', '2017-08-21', '2017-10-05', '2017-08-17', '2017-11-16', '2017-10-30', '2017-07-10', '2017-10-17', '2017-06-08', '2017-07-18', '2018-01-02', '2017-10-12', '2017-11-30', '2017-06-14', '2017-09-14', '2017-09-20', '2017-09-21', '2017-09-05', '2017-07-28', '2017-08-09', '2017-11-01', '2017-09-12', '2017-08-24', '2017-05-30', '2018-01-04', '2017-10-04', '2017-09-13', '2017-08-16', '2017-10-23', '2017-10-10', '2017-09-29', '2017-12-07', '2017-12-27', '2017-12-26', '2018-01-03', '2017-11-14', '2017-09-26', '2017-12-04', '2018-01-05', '2017-11-22', '2017-12-06', '2017-08-01', '2017-10-27', '2017-06-15', '2017-10-18', '2017-07-25', '2017-06-30', '2017-09-25', '2017-08-10', '2017-12-21', '2017-07-12', '2017-07-31', '2017-10-25', '2017-08-02', '2017-12-20', '2017-07-27', '2017-11-09', '2017-06-26', '2017-12-01', '2017-08-18', '2017-06-09', '2017-12-05', '2017-11-17', '2017-11-29', '2017-05-29', '2017-07-04', '2017-09-04', '2017-11-23', '2017-12-25', '2018-01-01']
        for selected_date in selected_dates:
            fig, axes = plt.subplots(nrows=3, ncols=1, squeeze=False, sharex="col")
            fig.set_figwidth(11)
            fig.set_figheight(21)
            selected_df = data_frame[selected_date]
            if len(selected_df) == 0:
                continue
            avoid_slump_run = TestStrategies.provide_avoid_slump(symbol_name, selected_df,
                                                                 zscore_threshold=-1.0,
                                                                 buffer=0.003)
            avoid_slump_run.start()

            if save_rtn_compare:
                self.rtn_compare_pic(avoid_slump_run, selected_date, axes[0, 0])
            if save_trend_pic:
                self.trend_pic(avoid_slump_run, selected_date, axes[1, 0])
            if save_z_trend:
                self.trend_z_pic(avoid_slump_run, selected_date, avoid_slump_run.zscore_threshold, axes[2, 0])

            back_test_res_df = avoid_slump_run.generate_report()
            intraday_effect = back_test_res_df['INTRADAY_RTN'] - back_test_res_df['INTRADAY_{}_RTN'.format(symbol_name)]
            back_test_res_df = back_test_res_df.assign(intraday_effect=intraday_effect)
            print(back_test_res_df['intraday_effect'])

            plot_name = "trend_and_rtn_compare_{}".format(selected_date)
            if intraday_effect.iloc[0] > 0.005:
                plt.savefig("../../pictures/trend_and_compare_good/{}.png".format(plot_name))
            elif intraday_effect.iloc[0] > -0.005:
                plt.savefig("../../pictures/trend_and_compare_neutral/{}.png".format(plot_name))
            else:
                plt.savefig("../../pictures/trend_and_compare_bad/{}.png".format(plot_name))
            plt.close()

    # @unittest.skip("just for experimental")
    def test_avoid_slump(self):
        output_per_day_result = True
        output_daily_return_pic = True
        symbol_name = "AMAT"
        syms_to_ld = ["AMAT", "UVXY"]  # symbols to load
        requested_col = ['time', 'high', 'low', 'open', 'close', 'volume']
        data_frame = utility.get_cols_from_csv_names(file_names=[utility.get_appropriate_file(s) for s in syms_to_ld],
                                                     interested_col=requested_col,
                                                     join_spy_for_data_integrity=False,
                                                     keep_spy_if_not_having_spy=False,
                                                     base_dir="../../rawdata")
        # selected_df = data_frame.loc['2017-11-07': '2017-11-15']
        selected_df = data_frame
        hist_z = []
        hist_mean = []
        hist_std = []
        for z_threshold in np.arange(-1.0, -0.9, 0.1):
            hist_z.append(z_threshold)
            print("z_threshold: {}".format(z_threshold))
            # selected_df = data_frame
            avoid_slump_run = TestStrategies.provide_avoid_slump(symbol_name, selected_df,
                                                                 zscore_threshold=z_threshold,
                                                                 buffer=0.003)
            avoid_slump_run.start()
            # print(avoid_slump_run.positions.head(100))

            back_test_res_df = avoid_slump_run.generate_report()

            intraday_effect = back_test_res_df['INTRADAY_RTN'] - back_test_res_df['INTRADAY_{}_RTN'.format(symbol_name)]
            back_test_res_df = back_test_res_df.assign(intraday_effect=intraday_effect)
            back_test_res_df.sort_values(by="intraday_effect", inplace=True)
            col_names = ["time", "intraday_effect", "INTRADAY_RTN", "INTRADAY_{}_RTN".format(symbol_name)]

            #  did back come fist
            if output_per_day_result:
                temp_dates = []
                print("{}                 {}   {}   {}".format(*col_names))
                for i2 in range(len(back_test_res_df)):
                    print("{0}  {1:9.6f}%  {2:9.6f}%  {3:9.6f}%"
                          .format(back_test_res_df.index[i2],
                                  back_test_res_df[col_names[1]].iloc[i2] * 100,
                                  back_test_res_df[col_names[2]].iloc[i2] * 100,
                                  back_test_res_df[col_names[3]].iloc[i2] * 100))
                    temp_dates.append(back_test_res_df.index[i2].strftime("%Y-%m-%d"))
                print(np.array(temp_dates))

            if output_daily_return_pic:
                daily_last = avoid_slump_run.totals.groupby(pd.Grouper(level=0, freq='1B')).last()
                daily_last = daily_last.dropna()
                daily_last = daily_last / avoid_slump_run.starting_cash
                daily_last.name = "daily_last"
                daily_bnh = selected_df['AMAT_CLOSE'].groupby(pd.Grouper(level=0, freq='1B')).last()
                daily_bnh = daily_bnh.dropna()
                daily_bnh = daily_bnh / selected_df['AMAT_OPEN'].iloc[0]
                daily_bnh.name = "daily_bnh"

                ax = daily_last.plot(title="RETURN", legend=True)
                daily_bnh.plot(ax=ax, legend=True)
                plt.show()

            print("\nintraday_effect.mean(): {0:9.6f}, \n"
                  "intraday_effect.std(): {1:9.6f}, \n"
                  "total_rtn: {2:9.6f}".format(
                back_test_res_df[col_names[1]].mean(),
                back_test_res_df[col_names[1]].std(),
                avoid_slump_run.totals.iloc[-1] / avoid_slump_run.totals.iloc[0]))
            hist_mean.append(back_test_res_df[col_names[1]].mean())
            hist_std.append(back_test_res_df[col_names[1]].std())
        result_df = pd.DataFrame({"mean": hist_mean, "std": hist_std}, index=hist_z)
        result_df.index.name = "z_thld"
        print(result_df.head(100))
        result_df.to_csv("../../quantopian_algs_backup/back_search_result.csv")

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
        # prepare csv headers of file
        meanz_csv = open("../../rawdata/{}.csv".format("MEAN_Z_IN_7_MIN"), "w")
        meanz_csv.write("time, MEAN_Z_IN_7_MIN\n")
        meanz_csv.close()
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
                axes[1, 0].set_ylim(bottom=-10, top=10)
                axes[1, 0].set_yticks(np.arange(-10, 11, 1))
                axes[1, 0].set_title("SUM Z SCORE COMPARE {}".format(selected_date))
                legends2 = []
                sum_z_in_7mins = []
                for symbol, rocp_60 in stock_rocp60.items():
                    rocp60_of_selected = rocp_60.loc[selected_date]
                    kwargs = {
                        "mean2": stock_min_change_means[symbol],
                        "std2": stock_min_change_stds[symbol]
                    }
                    min_z_in_7min = rocp60_of_selected.rolling(window=7, min_periods=1) \
                        .apply(statistics.min_z_score, kwargs=kwargs)
                    max_z_in_7min = rocp60_of_selected.rolling(window=7, min_periods=1) \
                        .apply(statistics.max_z_score, kwargs=kwargs)
                    sum_z = min_z_in_7min.add(max_z_in_7min)
                    sum_z_in_7mins.append(sum_z)
                    sum_z.name = "{}_SUM_Z_7MIN".format(symbol)
                    legends2.append(sum_z.name)
                    axes[1, 0].plot(sum_z.index.values, sum_z.values, alpha=0.16)
                mean_z = pd.concat(sum_z_in_7mins, axis=1).mean(axis=1)
                mean_z.name = "MEAN_Z_IN_7_MIN"
                mean_z.to_csv("../../rawdata/{}.csv".format(mean_z.name), mode='a')
                axes[1, 0].plot(mean_z.index.values, mean_z.values)
                legends2.append(mean_z.name)
                axes[1, 0].legend(legends2)
                # add zero line
                axes[1, 0].plot(mean_z.index.values, np.zeros((len(mean_z))))
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
