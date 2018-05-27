import os
import unittest

import pandas as pd

from analytic.data import tws_data_api_connector
from analytic.data.tws_data_api_connector import BarSize
from analytic.utility import RAW_DATA_PATH


class TestTwsDataApiConnector(unittest.TestCase):

    def test_get_historical_data_prices(self):
        symbol = "NVDA"
        end_local_date_str = "20180524 16:00:00"
        res_data = \
            tws_data_api_connector.get_historical_data_prices(symbol, end_local_date_str, 5, BarSize.DAY_1, True)
        last_row = res_data.iloc[-1]
        # print(last_row)
        self.assertEqual(last_row["m_close"], 247.69)
        self.assertEqual(last_row["m_count"], 35770.00)
        self.assertEqual(last_row["m_high"], 249.40)
        self.assertEqual(last_row["m_low"], 245.24)
        self.assertEqual(last_row["m_volume"], 77090)
        self.assertEqual(last_row["m_wap"], 247.63)

    def test_sync_sombol_to_local(self):
        symbol = "NVDA"
        target_file = os.path.join(RAW_DATA_PATH, "{}-TWS-DATA.csv".format(symbol))
        tws_data_api_connector.sync_sombol_to_local(symbol)
        df = pd.read_csv(target_file, parse_dates=True, index_col="m_time_iso")
        self.assertTrue("m_time_iso" == df.index.name)
        self.assertTrue("m_close" in df.columns)
        self.assertTrue("m_count" in df.columns)
        self.assertTrue("m_high" in df.columns)
        self.assertTrue("m_low" in df.columns)
        self.assertTrue("m_open" in df.columns)
        self.assertTrue("m_volume" in df.columns)
        self.assertTrue("m_wap" in df.columns)
        target_row = df.loc[pd.Timestamp(ts_input="2018-05-25 09:00:00")]
        # print(target_row)
        self.assertEqual(target_row["m_close"], 249.28)
        self.assertEqual(target_row["m_count"], 23654)
        self.assertEqual(target_row["m_high"], 249.94)
        self.assertEqual(target_row["m_low"], 246.76)
        self.assertEqual(target_row["m_open"], 248.20)
        self.assertEqual(target_row["m_volume"], 50867)
        self.assertAlmostEqual(target_row["m_wap"], 248.876, 2)
