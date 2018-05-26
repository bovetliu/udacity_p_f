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

