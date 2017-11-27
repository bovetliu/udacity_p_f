import unittest

import pandas as pd

from analytic import utility


class TestUtility(unittest.TestCase):

    def test_get_relative_net_worth(self):
        data_frame = utility.get_cols_from_csv_names(['AAPL_20100104-20171013'],
                                                     keep_spy_if_not_having_spy=False,
                                                     interested_col=['Date', 'Close', 'Open', 'Volume'],
                                                     base_dir="../../rawdata")

        data_frame = data_frame.loc['2017-09-28':'2017-10-13']
        signal = [0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0]
        data_frame = data_frame.assign(AAPL_SIGNAL=pd.Series(signal).values)
        net_worth_ser = utility.get_relative_net_worth(data_frame, 'AAPL')
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
