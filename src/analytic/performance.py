import math
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
from analytic import ta_indicators


def get_relative_net_worth(data_frame: pd.DataFrame, symbol: str) -> pd.Series:
    """
    get relative net worth pandas series
    :param data_frame: data frame
    :param symbol: stock symbol
    :return: pandas series recording net worth change along bars and signals
    """

    if not isinstance(data_frame, pd.DataFrame):
        raise TypeError("data_frame can only be pd.DataFrame")
    if not symbol or not isinstance(symbol, str):
        raise TypeError("symbol not valid.")
    net_worth = 1.0
    current_position = 0  # 0 means emtpy, 1 means long, -1 means short
    prev_price = 0

    pending_action = 0
    tbr = pd.Series(data=[], index=[])
    for index, row in data_frame.iterrows():
        cur_price = row['{}_OPEN'.format(symbol)]
        # following is handling pening action at open of each bar
        if pending_action is not None and pending_action != current_position:
            # close position first
            if current_position != 0:
                ratio = (cur_price / prev_price) if prev_price != 0 else 1
                net_worth = ratio * net_worth if current_position > 0 else net_worth / ratio
            # simulate order_target_percent
            if pending_action in (-1, 0, 1):
                current_position = pending_action
            else:
                raise ValueError("Unrecognized sig value {}".format(pending_action))
            # prev_rtn always record a check point
            prev_price = cur_price

        # end of market opening, now it is end of market
        cur_sig = row['{}_SIGNAL'.format(symbol)]
        cur_price = row['{}_CLOSE'.format(symbol)]
        if cur_sig != current_position:
            pending_action = cur_sig
        else:
            pending_action = None
        # update net worth at end of each trading day
        if current_position != 0:
            ratio = (cur_price / prev_price) if prev_price != 0 else 1
            net_worth = ratio * net_worth if current_position > 0 else net_worth / ratio

        # https://github.com/pandas-dev/pandas/issues/2801  the contributor closed pandas inplace appending series.
        tbr = tbr.append(pd.Series(data=[net_worth], index=[index]))
        # after market close
        prev_price = cur_price
    return tbr


def get_sharp_ratio(val_ser: pd.Series, risk_free_daily_rtn:float = 0.0):
    drtn = ta_indicators.get_daily_return(val_ser)
    drtn.iloc[0] = 0
    mean_rtn = drtn.mean()
    return (mean_rtn - risk_free_daily_rtn) / drtn.std()


def backtest(hist_prices, positions, starting_cash: float=30000.0):
    """
    Give analysis like total return, return diff between buy and hold, sharp ratio, position change time, win ratio
    :param hist_prices: if DataFrame, then expecting columns of {STOCK_NAME}_CLOSE, if pd.Series, then assuming closes
    :param positions: if DataFrame, then expecting {STOCK}_POSITION, if pd.Series, then assuming positions
    :param starting_cash: float, indicating starting cash
    :return: do not know yet
    """
    # first only consider both parameters are series.
    if isinstance(hist_prices, pd.Series) and isinstance(positions, pd.Series):
        symbol_name = hist_prices.name.split("_")[0]
        print("symbol_name: {}".format(symbol_name))
        pos_diff = positions.diff()
        pos_diff.iloc[0] = positions.iloc[0]
        pos_diff.name = "{}_OPERATION".format(symbol_name)
        holdings = positions * hist_prices
        holdings.name = "{}_HOLDING".format(symbol_name)
        cash = starting_cash - (pos_diff * hist_prices).cumsum(axis=0)
        cash.name = "CASH"
        totals = holdings + cash
        totals.name = "PORTFOLIO_VALUE"
        rtn = totals.pct_change().fillna(0).add(1).cumprod().add(-1)
        rtn.name = "PORTFOLIO_RETURN"
        back_test_result_df = pd.concat([pos_diff, holdings, cash, totals, rtn], axis=1)
        return BackTestResult(back_test_result_df, len(pos_diff.nonzero()))

    if isinstance(hist_prices, pd.DataFrame) and isinstance(positions, pd.DataFrame):
        # levelrages = holdings / starting_cash
        raise Exception("both data frame not supported yet")
    raise TypeError("hist_prices, positions should either both pd.Series, or both pd.DataFrame")


class BackTestResult:

    def __init__(self, back_test_result_df, num_operation=None):
        self.back_test_result_df = back_test_result_df
        self.num_operation = num_operation

    def final_return(self):
        """
        return final return of back test
        """
        return self.back_test_result_df["PORTFOLIO_RETURN"].iloc[-1]


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
    def handle_data(self, pr_open, pr_high, pr_low, pr_close, volume, **kwargs):
        pass

    @abstractmethod
    def just_after_close(self):
        pass

    def start(self):
        self.__iterate_bars()

    def order_target_percent(self, target_percent: float, overriding_price: float=None):
        cur_price = overriding_price if (overriding_price is not None) \
            else self.hist_data.loc[self.current_simu_time][self.col_dict['close']]
        if cur_price is None:
            raise ValueError("cur_price is None")
        cur_pos = self.positions.loc[self.current_simu_time]
        cur_cash = self.cashes.loc[self.current_simu_time]
        cur_total = cur_price * cur_pos + cur_cash
        self.order_target_value(cur_total * target_percent, cur_price)

    def order_target_value(self, target_holding_value: float, overriding_price: float=None):
        cur_price = overriding_price if (overriding_price is not None) \
            else self.hist_data.loc[self.current_simu_time][self.col_dict['close']]
        abs_holding_val = abs(target_holding_value)
        target_pos = int(math.floor(abs_holding_val / cur_price)) * (1 if target_holding_value >= 0.0 else -1)
        self.order_target(target_pos, cur_price)

    def order_target(self, target_position: int, overriding_price: float=None):
        cur_price = overriding_price if (overriding_price is not None) \
            else self.hist_data.loc[self.current_simu_time][self.col_dict['close']]
        cur_pos = self.positions.loc[self.current_simu_time]
        self.order(target_position - cur_pos, cur_price)

    def order(self, pos: int, overriding_price: float):
        cur_price = overriding_price if (overriding_price is not None) \
            else self.hist_data.loc[self.current_simu_time][self.col_dict['close']]
        cur_cash = self.cashes.loc[self.current_simu_time]
        cur_pos = self.positions.loc[self.current_simu_time]

        value = pos * cur_price

        cur_cash_updated = cur_cash - self.__calc_commission(pos, cur_price) - value
        cur_pos_updated = cur_pos + pos
        self.cashes.loc[self.current_simu_time] = cur_cash_updated
        self.positions.loc[self.current_simu_time] = cur_pos_updated
        self.totals.loc[self.current_simu_time] = cur_pos_updated * cur_price + cur_cash_updated

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
        for time in self.hist_data.index[begin_end_indice[0]:begin_end_indice[1]]:
            one_period_before = time - self.__one_period
            one_period_after = time + self.__one_period
            if one_period_before not in self.hist_data.index:
                # assume before trading
                self.current_simu_time = time - pd.Timedelta("15 minutes")
                self.before_trading()
                # if self.current_simu_time.minute == 30:
                #     self.__offset = pd.Timedelta("1 minute")
                # elif self.current_simu_time == 31:
                #     self.__offset = pd.Timedelta("0 minute")

            self.current_simu_time = time
            self.__update()

            row_ser = self.hist_data.loc[time]
            row_dict = row_ser.to_dict()
            self.handle_data(
                row_dict[self.col_dict['open']],
                row_dict[self.col_dict['high']],
                row_dict[self.col_dict['low']],
                row_dict[self.col_dict['close']],
                row_dict[self.col_dict['volume']],
                **row_dict)

            if one_period_after not in self.hist_data.index:
                self.current_simu_time = time + pd.Timedelta("1 minute")
                self.just_after_close()

    def __update(self):
        cur_simu_time_idx = self.hist_data.index.get_loc(self.current_simu_time)
        if cur_simu_time_idx == 0:
            return
        prev_simu_time_idx = cur_simu_time_idx - 1
        prev_pos = self.positions.iloc[prev_simu_time_idx]
        prev_cash = self.cashes.iloc[prev_simu_time_idx]

        cur_price = self.hist_data.iloc[cur_simu_time_idx][self.col_dict['close']]
        cur_total_in_record = self.totals.iloc[cur_simu_time_idx]
        if cur_total_in_record == 0:
            self.cashes.loc[self.current_simu_time] = prev_cash
            self.positions.loc[self.current_simu_time] = prev_pos
            self.totals.loc[self.current_simu_time] = prev_pos * cur_price + prev_cash

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
        totals_grp = self.totals.groupby(pd.Grouper(level=0, freq='1B'))
        intraday_rtn = (totals_grp.last() - totals_grp.first()) / totals_grp.first()
        intraday_rtn.name = "INTRADAY_RTN"
        # print(intraday_rtn.head())
        self.back_test_result = pd.concat([operations_per_day, intraday_rtn, intraday_symbol_rtn], axis=1)
        return self.back_test_result
