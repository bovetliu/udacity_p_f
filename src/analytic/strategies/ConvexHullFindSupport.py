from analytic.strategies.base import SingleStockStrategy


class ConvexHullFindSupport(SingleStockStrategy):

    def __init__(self, symbol_name, hist_data,
                 begin_datetime=None,
                 end_datetime=None,
                 starting_cash=5000):
        super().__init__(symbol_name, hist_data, begin_datetime, end_datetime, starting_cash)

    def before_trading(self):
        """
        core overriding method
        """
        pass

    def handle_data(self, pr_open, pr_high, pr_low, pr_close, wap, volume, **kwargs):
        """
        core overriding method
        """
        pass

    def last_operation_of_trading_day(self):
        """
        core overriding method
        """
        pass

