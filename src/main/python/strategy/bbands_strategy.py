import pandas as pd

class BBandsStrategy:

    def __init__(self):
        # last sell at price
        self.rtn = 1.0
        self.last_sell_at = 0
        self.prev_rtn = 0
        self.prev_upper_band = 0
        self.prev_sma = 0
        self.prev_lower_band = 0

        self.time_when_bought = None
        self.bought_at_rtn = 0

        self.time_when_closed = None
        self.close_at_rtn = 0

    def run(self, in_df: pd.Series):
        pass


    def handle(self, time, cur_rtn, upper_band, sma, lower_band):
        """
        handle row by row
        """
        do_close = False
        do_buy = False
        if (self.prev_rtn <= self.prev_lower_band) and (cur_rtn >= lower_band):
            # up crossing lower band
            do_buy = True

        if (self.prev_rtn >= self.prev_sma) and (cur_rtn < sma):
            # scenario down crossing sma
            do_close = True
        if (self.prev_rtn > self.prev_upper_band) and (cur_rtn <= upper_band):
            do_close = True

        if (self.prev_rtn >= self.bought_at_rtn * 0.97) and (cur_rtn < self.bought_at_rtn * 0.97):
            do_close = True

        if do_buy and do_close:
            raise Exception('need to take a look at this state')
        if do_close:
            self.close(time, cur_rtn)
        elif do_buy:
            self.buy(time, cur_rtn)

    def buy(self, time, rtn):
        if not self.bought_at_rtn:
            self.time_when_bought = time
            self.bought_at_rtn = rtn
            print('at {} buy@{}'.format(time, rtn))
        self.time_when_closed = None
        self.close_at_rtn = 0


    def close(self, time, rtn):
        if not self.bought_at_rtn:
            # if currently indeed holding
            self.rtn *= (rtn / self.bought_at_rtn)
            self.time_when_closed = time
            self.close_at_rtn = rtn
            print('at {} sell@{}'.format(time, rtn))
        self.time_when_bought = None
        self.bought_at_rtn = 0
