import math
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class SingleStockStrategy(ABC):
    """
    designed to handle single stock strategy
    """

    def __init__(self, symbol_name, hist_data,
                 begin_datetime=None,
                 end_datetime=None,
                 starting_cash=5000):
        """
        initialize one strategy
        :param symbol_name:
        :param hist_data: data frame, should contain OHLCV data of one symbol in time
        """
        if not symbol_name or not isinstance(symbol_name, str):
            raise ValueError("symbol name must be valid string")
        if not isinstance(hist_data, pd.DataFrame):
            raise TypeError("hist_data must be pd.DataFrame")
        self.symbol_name = symbol_name
        self.col_dict = {
            "open": "{}_OPEN".format(symbol_name),
            "high": "{}_HIGH".format(symbol_name),
            "low": "{}_LOW".format(symbol_name),
            "close": "{}_CLOSE".format(symbol_name),
            "wap": "{}_WAP".format(symbol_name),
            "volume": "{}_VOLUME".format(symbol_name),
        }
        self.hist_data = hist_data
        self.begin_datetime = begin_datetime
        self.end_datetime = end_datetime
        self.starting_cash = starting_cash

        # following will be updated after handle data
        self.positions = pd.Series(np.zeros((len(hist_data)), dtype=np.int32), index=hist_data.index)
        self.positions.iloc[0] = 0
        self.cashes = pd.Series(np.zeros((len(hist_data)), dtype=np.float64), index=hist_data.index)
        self.cashes.iloc[0] = self.starting_cash
        self.totals = pd.Series(np.zeros((len(hist_data)), dtype=np.float64), index=hist_data.index)
        self.totals.iloc[0] = self.starting_cash

        self.current_simu_time = None
        self.current_simu_time_i = None
        # TODO(Bowei) shrink following code, too clumsy
        found_period = False
        self.__one_period = self.hist_data.index[1] - self.hist_data.index[0]
        for i in range(1, 6):
            next_delta = self.hist_data.index[i + 1] - self.hist_data.index[i]
            if self.__one_period != next_delta:
                continue
            else:
                found_period = True
                break
        if not found_period:
            raise ValueError("Could not find period of hist_data")
        if self.__one_period != pd.Timedelta("1 minute"):
            raise ValueError("Currently only support minute level simulation")

        self.ib_commission_method = "Fixed"  # another available option is "Tiered"
        self.back_test_result = None  # expecting one data frame

    @abstractmethod
    def before_trading(self):
        pass

    @abstractmethod
    def handle_data(self, pr_open, pr_high, pr_low, pr_close, wap, volume, **kwargs):
        pass

    @abstractmethod
    def last_operation_of_trading_day(self):
        pass

    def start(self):
        self.__iterate_bars()

    def order_target_percent(self, target_percent: float, overriding_price: float=None):
        cur_price = overriding_price if (overriding_price is not None) \
            else self.hist_data.iloc[self.current_simu_time_i][self.col_dict['close']]
        if cur_price is None:
            raise ValueError("cur_price is None")
        cur_pos = self.positions.iloc[self.current_simu_time_i]
        cur_cash = self.cashes.iloc[self.current_simu_time_i]
        cur_total = cur_price * cur_pos + cur_cash
        self.order_target_value(cur_total * target_percent, cur_price)

    def order_target_value(self, target_holding_value: float, overriding_price: float=None):
        cur_price = overriding_price if (overriding_price is not None) \
            else self.hist_data.iloc[self.current_simu_time_i][self.col_dict['close']]
        abs_holding_val = abs(target_holding_value)
        target_pos = int(math.floor(abs_holding_val / cur_price)) * (1 if target_holding_value >= 0.0 else -1)
        self.order_target(target_pos, cur_price)

    def order_target(self, target_position: int, overriding_price: float=None):
        cur_price = overriding_price if (overriding_price is not None) \
            else self.hist_data.iloc[self.current_simu_time_i][self.col_dict['close']]
        cur_pos = self.positions.iloc[self.current_simu_time_i]
        self.order(target_position - cur_pos, cur_price)

    def order(self, pos: int, overriding_price: float):
        cur_price = overriding_price if (overriding_price is not None) \
            else self.hist_data.iloc[self.current_simu_time_i][self.col_dict['close']]
        cur_cash = self.cashes.iloc[self.current_simu_time_i]
        cur_pos = self.positions.iloc[self.current_simu_time_i]
        # if pos:
        #     print("time: {}, pos: {}, cur_price: {}".format( self.current_simu_time, pos, cur_price))
        value = pos * cur_price

        cur_cash_updated = cur_cash - self.__calc_commission(pos, cur_price) - value
        cur_pos_updated = cur_pos + pos
        self.cashes.iloc[self.current_simu_time_i] = cur_cash_updated
        self.positions.iloc[self.current_simu_time_i] = cur_pos_updated
        self.totals.iloc[self.current_simu_time_i] = cur_pos_updated * cur_price + cur_cash_updated

    def deduct_total_value_percent(self, to_be_ducted):
        """
        necessary function to offset possible future function effect, might used to calculate buffer
        """
        self.cashes.iloc[self.current_simu_time_i] *= (1 - to_be_ducted)

    def __calc_commission(self, pos, avg_share_price=0):
        """
        currently using Interactive brokers Stocks, ETFs (ETPs) and Warrants - Tiered Pricing Structure
        https://www.interactivebrokers.com/en/index.php?f=1590&p=stocks2
        :return: estimated commission fee for this order
        """
        if pos % 1.0 != 0.0:
            raise ValueError("pos must be integer")
        abs_pos = abs(pos)
        commission = 0.0
        if self.ib_commission_method.lower() == "tiered":
            commission = min(0.35, 0.0035 * abs_pos)

            # exchange fee have no idea how to calculate

            # clearing fee
            commission += 0.00020 * abs_pos
        elif self.ib_commission_method.lower() == "fixed":
            commission = min(1.0, 0.005 * abs_pos)
            if pos < 0:
                # USD 0.000119 * Quantity Sold: FINRA Trading Activity Fee
                commission += abs_pos * 0.000119
                # Transaction Fees
                commission += abs_pos * avg_share_price * 0.0000231
        return commission

    def __iterate_bars(self):
        begin_end_indice = self.hist_data.index.slice_locs(start=self.begin_datetime, end=self.end_datetime)
        print(begin_end_indice)
        for i in range(begin_end_indice[0], begin_end_indice[1]):
            time = self.hist_data.index[i]
            # one_period_before = time - self.__one_period
            one_period_after = time + self.__one_period
            if time.hour == 9 and time.minute == 30:
                # assume before trading
                self.current_simu_time = time - pd.Timedelta("15 minutes")
                self.before_trading()

            self.current_simu_time = time
            self.current_simu_time_i = i
            self.__update()

            row_ser = self.hist_data.iloc[i]
            row_dict = row_ser.to_dict()
            self.handle_data(
                row_dict[self.col_dict['open']],
                row_dict[self.col_dict['high']],
                row_dict[self.col_dict['low']],
                row_dict[self.col_dict['close']],
                row_dict[self.col_dict['wap']],
                row_dict[self.col_dict['volume']],
                **row_dict)

            if time.minute == 59 and (time.hour == 12 or time.hour == 15):
                if one_period_after not in self.hist_data.index:
                    self.last_operation_of_trading_day()
                    # self.current_simu_time = time + pd.Timedelta("1 minute")

    def __update(self):
        cur_simu_time_idx = self.current_simu_time_i
        if cur_simu_time_idx == 0:
            return
        prev_simu_time_idx = cur_simu_time_idx - 1
        prev_pos = self.positions.iloc[prev_simu_time_idx]
        prev_cash = self.cashes.iloc[prev_simu_time_idx]

        cur_price = self.hist_data.iloc[cur_simu_time_idx][self.col_dict['close']]
        cur_total_in_record = self.totals.iloc[cur_simu_time_idx]
        if cur_total_in_record == 0:
            self.cashes.iloc[self.current_simu_time_i] = prev_cash
            self.positions.iloc[self.current_simu_time_i] = prev_pos
            self.totals.iloc[self.current_simu_time_i] = prev_pos * cur_price + prev_cash

    def generate_report(self):
        """
        generate report
        :return:
        """
        operations = self.positions.diff()
        operations.iloc[0] = self.positions.iloc[0]
        operations = operations.apply(lambda pos_change: 0 if pos_change == 0 else 1)
        operations.name = "{}_OPERATION".format(self.symbol_name)
        daily_op_grp = operations.groupby(pd.Grouper(level=0, freq='1B'))

        operations_per_day = daily_op_grp.sum()
        operations_per_day.name = "OPERATION_CNT_PER_DAY"

        dgrp = self.hist_data.groupby(pd.Grouper(level=0, freq='1B'))

        intraday_symbol_rtn = (dgrp[self.col_dict['close']].last() - dgrp[self.col_dict['open']].first()) \
                    / dgrp[self.col_dict['open']].first()
        intraday_symbol_rtn.name = "INTRADAY_{}_RTN".format(self.symbol_name)
        # print(intraday_symbol_rtn.head())

        # print(self.totals.head(10))
        dgrp[self.col_dict['open']].first() / dgrp[self.col_dict['close']].first()
        totals_grp = self.totals.groupby(pd.Grouper(level=0, freq='1B'))
        corrected_totals_grp_first = totals_grp.first() * (dgrp[self.col_dict['open']].first() / dgrp[self.col_dict['close']].first())
        intraday_rtn = (totals_grp.last() - corrected_totals_grp_first) / corrected_totals_grp_first
        intraday_rtn.name = "INTRADAY_RTN"
        # print(intraday_rtn.head())
        self.back_test_result = pd.concat([operations_per_day, intraday_rtn, intraday_symbol_rtn], axis=1)
        return self.back_test_result
