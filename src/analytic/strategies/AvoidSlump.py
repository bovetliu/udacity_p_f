import collections
from analytic.performance import SingleStockStrategy
from analytic import statistics

import numpy as np


class AvoidSlump(SingleStockStrategy):

    def __init__(self, symbol_name, hist_data,
                 begin_datetime=None,
                 end_datetime=None,
                 starting_cash=5000,
                 rocp_mean=-0.000002602,
                 rocp_std=0.000831548,
                 drop_down_window=7,
                 sma_window=7):
        super().__init__(symbol_name, hist_data, begin_datetime, end_datetime, starting_cash)
        self.rocp_mean = rocp_mean
        self.rocp_std = rocp_std
        self.rocp_params = {
            "mean2": rocp_mean,
            "std2": rocp_std,
        }
        self.drop_down_window = drop_down_window
        self.sma_window = sma_window

        self.last_prices = []
        self.zhishun_line = []
        # start zhishun condition : <= self.zscore_threshold
        self.zscore_threshold = -1.0
        # stop zhishun condition: not prev_need_zhishun and rocp_sma >= self.sma_threshold
        self.sma_threshold = 0.0

    def before_trading(self):
        """
        """
        self.last_prices.clear()
        self.zhishun_line.clear()

    def handle_data(self, pr_open, pr_high, pr_low, pr_close, volume):
        self.last_prices.append(pr_close)
        len_prices = len(self.last_prices)

        left_bd = max(-self.drop_down_window, -len_prices)
        zscored_drop_down = statistics.drop_down(self.last_prices[left_bd:], **self.rocp_params)

        left_bd = max(-self.sma_window, -len_prices)
        cur_ma = np.sum(self.last_prices[left_bd:]) / min(len_prices, self.sma_window)

        prev_ma = None
        if len_prices > 1:
            left_bd = max(-self.sma_window - 1, -len_prices)
            prev_ma = np.sum(self.last_prices[left_bd:-1]) / len(self.last_prices[left_bd:-1])

        rocp_ma = 0 if prev_ma is None else (cur_ma - prev_ma) / prev_ma

        # now rocp_ma and zscored_drop_down have been obtained
        newest_zhishun = self.calc_zhishun(pr_close, zscored_drop_down, rocp_ma,
                          self.start_zhishun_condition,
                          self.end_zhishun_condition)
        if len(self.zhishun_line) != len_prices:
            raise ValueError("len(self.zhishun_line): {}, while len_prices: {}".format(
                len(self.zhishun_line), len_prices))

        if newest_zhishun is None:
            self.order_target_percent(1.0)
            return
        # at this time, zhishun line is present
        cur_pos = self.positions.loc[self.current_simu_time]
        if pr_close <= newest_zhishun:
            self.order_target_percent(0.0)
        elif cur_pos == 0 and rocp_ma <= 0.0:
            self.order_target_percent(0.0)
        else:
            self.order_target_percent(1.0)

    def calc_zhishun(self, cur_pr, zscored_drop_down, rocp_ma, should_start_zhishun, should_stop_zhishun, buffer=0.003):
        if len(self.zhishun_line) == 0:
            self.zhishun_line.append(None)
        else:
            prev_need_zhishun = self.zhishun_line[-1] is not None
            does_start_zhishun = should_start_zhishun(cur_pr, zscored_drop_down, rocp_ma)
            should_extend_zhishun = prev_need_zhishun and (not should_stop_zhishun(cur_pr, zscored_drop_down, rocp_ma))
            cur_need_zhishun = does_start_zhishun or should_extend_zhishun
            if not cur_need_zhishun:
                self.zhishun_line.append(None)
            else:
                if prev_need_zhishun:
                    self.zhishun_line.append(self.zhishun_line[-1])
                else:
                    self.zhishun_line.append(cur_pr * (1 - buffer))
        return self.zhishun_line[-1]

    def start_zhishun_condition(self, cur_pr, zscored_drop_down, rocp_ma):
        return zscored_drop_down <= self.zscore_threshold

    def end_zhishun_condition(self, cur_pr, zscored_drop_down, rocp_ma):
        return rocp_ma > self.sma_threshold

    def just_after_close(self):
        print("nothing to do after close {}".format(self.current_simu_time))
