import unittest

import pandas as pd
import numpy as np

from analytic import performance
from analytic import utility


class TestPerformance(unittest.TestCase):

    def test_get_relative_net_worth(self):
        data_frame = utility.get_cols_from_csv_names(['AAPL_20100104-20171013'],
                                                     interested_col=['Date', 'Close', 'Open', 'Volume'],
                                                     join_spy_for_data_integrity=False,
                                                     keep_spy_if_not_having_spy=False,
                                                     base_dir="../../rawdata")

        data_frame = data_frame.loc['2017-09-28':'2017-10-13']
        signal = [0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0]
        data_frame = data_frame.assign(AAPL_SIGNAL=pd.Series(signal).values)
        net_worth_ser = performance.get_relative_net_worth(data_frame, 'AAPL')
        self.assertAlmostEqual(1, net_worth_ser.iloc[0], 6)
        self.assertAlmostEqual(1, net_worth_ser.iloc[1], 6)
        self.assertAlmostEqual(1, net_worth_ser.iloc[2], 6)
        self.assertAlmostEqual(1, net_worth_ser.iloc[3], 6)
        self.assertAlmostEqual(0.999024, net_worth_ser.iloc[4], 6)
        self.assertAlmostEqual(1.011456, net_worth_ser.iloc[5], 6)
        self.assertAlmostEqual(1.010870, net_worth_ser.iloc[6], 6)
        self.assertAlmostEqual(1.014190, net_worth_ser.iloc[7], 6)
        self.assertAlmostEqual(1.014190, net_worth_ser.iloc[8], 6)
        self.assertAlmostEqual(1.017961, net_worth_ser.iloc[9], 6)
        self.assertAlmostEqual(1.014385, net_worth_ser.iloc[10], 6)
        self.assertAlmostEqual(1.019132, net_worth_ser.iloc[11], 6)
        print(net_worth_ser)

    def test_backtest(self):
        symbol = "AMAT"
        requested_col = ['time', 'high', 'low', 'open', 'close']
        data_frame = utility.get_cols_from_csv_names(file_names=['AMAT_to_2018-01-05'],
                                                     interested_col=requested_col,
                                                     join_spy_for_data_integrity=False,
                                                     keep_spy_if_not_having_spy=False,
                                                     base_dir="../../rawdata")
        # select 2017-12-21, one go down day
        selected_date = "2017-12-21"
        closes = data_frame.loc[selected_date]["AMAT_CLOSE"]
        # a holding of all 1
        numpy_holdings = np.ones((len(closes)), dtype=np.float64)
        for i in range(len(numpy_holdings)):
            if i > 2:
                numpy_holdings[i] *= 1.8
        numpy_holdings = numpy_holdings * 310
        holdings = pd.Series(numpy_holdings, index=closes.index)

        back_test_result = performance.backtest(closes, holdings)
        back_test_result_df = back_test_result.back_test_result_df
        print("return of day: {:6.2f}%".format(back_test_result.final_return() * 100))
        # expected holding result of day is -2.12%
        self.assertAlmostEqual(-2.12, back_test_result.final_return() * 100, 2, "expected holding result is -2.12%")
