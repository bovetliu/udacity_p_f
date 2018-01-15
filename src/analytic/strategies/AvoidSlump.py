from analytic.performance import SingleStockStrategy
from analytic import statistics

import numpy as np


class AvoidSlump(SingleStockStrategy):

    def __init__(self, symbol_name, hist_data,
                 begin_datetime=None,
                 end_datetime=None,
                 starting_cash=5000,
                 rocp_mean=-0.000002602,  # pre-calculated from previous 1 year minute date of AMAT
                 rocp_std=0.000831548,  # pre-calculated from previous 1 year minute date of AMAT
                 drop_down_window=7,
                 sma_window=7,
                 zscore_threshold=-1.0,
                 sma_threshold=0.0):
        super().__init__(symbol_name, hist_data, begin_datetime, end_datetime, starting_cash)
        self.rocp_mean = rocp_mean
        self.rocp_std = rocp_std
        self.rocp_params = {
            "mean2": rocp_mean,
            "std2": rocp_std,
        }
        self.drop_down_window = drop_down_window
        self.sma_window = sma_window

        self.last_rocps = []
        self.last_prices = []  # to be cleared before trading
        self.zhishun_line_befei = np.array([], dtype=np.float64)
        self.zhishun_line = []  # to be cleared before trading
        # start zhishun condition : <= self.zscore_threshold
        self.zscore_threshold = zscore_threshold
        # stop zhishun condition: not prev_need_zhishun and rocp_sma >= self.sma_threshold
        self.sma_threshold = sma_threshold

        self.__daily_loss_control = []
        self.daily_loss_control_beifen = []
        self.__open_pr = None

    def before_trading(self):
        """
        """
        self.__daily_loss_control.clear()
        self.last_rocps.clear()
        self.last_prices.clear()
        self.zhishun_line.clear()
        self.__open_pr = None

    def handle_data(self, pr_open, pr_high, pr_low, pr_close, volume, **kwargs):
        # print(pr_open, pr_high, pr_low, pr_close, volume)
        if self.__open_pr is None:
            self.__open_pr = pr_open
        if len(self.last_prices) == 0:
            self.last_rocps.append((pr_close - pr_open) / pr_open)
        else:
            self.last_rocps.append((pr_close - self.last_prices[-1]) / self.last_prices[-1])
        self.last_prices.append(pr_close)
        if len(self.__daily_loss_control) == 0:
            self.__daily_loss_control.append(pr_close * 0.985)
        else:
            self.__daily_loss_control.append(max(self.__daily_loss_control[-1], pr_close * 0.985))
        # print(self.__daily_loss_control[-1])

        len_prices = len(self.last_prices)

        left_bd = max(-self.drop_down_window, -len_prices)
        # zscored_drop_down = statistics.drop_down(self.last_prices[left_bd:], **self.rocp_params)
        cur_z_min = statistics.min_z_score(self.last_rocps[left_bd:], **self.rocp_params)
        cur_z_max = statistics.max_z_score(self.last_rocps[left_bd:], **self.rocp_params)
        cur_z_mean_over = cur_z_max + cur_z_min

        # # external mean
        # print("cur_z_min: {}, cur_z_max: {}, cur_z_mean_over: {}".format(
        #     cur_z_min, cur_z_max, cur_z_mean_over))
        # cur_z_mean_over = kwargs["MEAN_Z_IN_7_MIN"]
        rocp_ma = AvoidSlump.calc_rocp_of_sma(self.last_prices, self.sma_window)

        # now rocp_ma and zscored_drop_down have been obtained
        newest_zhishun = self.calc_zhishun(pr_close, cur_z_mean_over, rocp_ma,
                                           self.start_zhishun_condition,
                                           self.end_zhishun_condition,
                                           buffer=0.004)
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
        elif cur_pos == 0:    # if zhishun line presents, and have already touched zhishun line, then do not buy
            self.order_target_percent(0.0)
        else:
            self.order_target_percent(1.0)

    def just_after_close(self):
        self.zhishun_line_befei = np.append(self.zhishun_line_befei, self.zhishun_line)
        self.daily_loss_control_beifen = np.append(self.daily_loss_control_beifen, self.__daily_loss_control)
        print("finished day : {}".format(self.current_simu_time.strftime('%Y-%m-%d')))

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

    @staticmethod
    def calc_rocp_of_sma(arr, window):
        """
        return ROCP of sma having window size specified. Tested at test_strategies.TestStrategies#test_calc_rocp_of_sma

        :param arr: array like, which can be sliced.
        :param window: an positive integer expected,
        :return: rate of change percent of window sma
        """
        if len(arr) < 2:
            return 0
        left_bd = max(-window, -len(arr))
        ma_of_cur_window = np.sum(arr[left_bd:]) / min(window, len(arr))

        left_bd = max(-window - 1, -len(arr))
        ma_of_prev_window = np.sum(arr[left_bd:-1]) / len(arr[left_bd:-1])
        return (ma_of_cur_window - ma_of_prev_window) / ma_of_prev_window
