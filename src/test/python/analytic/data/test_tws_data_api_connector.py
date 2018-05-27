import os
import shutil
import unittest

import pandas as pd

from analytic.data import tws_data_api_connector
from analytic.data.tws_data_api_connector import BarSize
from analytic.utility import RAW_DATA_PATH
from analytic.utility import TEST_RESOURCES_PATH


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
        target_file = os.path.join(RAW_DATA_PATH, "{}-TWS-DATA-DAILY.csv".format(symbol))
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

    def test_get_local_data(self):
        symbol = "NODATA"
        caught_expected_error = False
        try:
            tws_data_api_connector.get_local_data(symbol)
        except FileNotFoundError as fnf:
            caught_expected_error = True
            print("caught FileNotFoundError: {}".format(fnf))
        self.assertTrue(caught_expected_error)

        # now move a file from test resources to "rawdata" directory
        file_name = "NVDA-TWS-DATA-DAILY.csv"
        file_name_for_test = "NVDAFORTEST-TWS-DATA-DAILY.csv"
        target_file_path = os.path.join(TEST_RESOURCES_PATH, "test_data", file_name)
        dest_filie_path = os.path.join(RAW_DATA_PATH, file_name_for_test)
        shutil.copyfile(target_file_path, dest_filie_path)
        df = tws_data_api_connector.get_local_data("NVDAFORTEST")
        target_row = df.loc[pd.Timestamp(ts_input="2018-05-25 09:00:00")]
        self.assertEqual(target_row["m_close"], 249.28)
        self.assertEqual(target_row["m_count"], 23654)
        self.assertEqual(target_row["m_high"], 249.94)
        self.assertEqual(target_row["m_low"], 246.76)
        self.assertEqual(target_row["m_open"], 248.20)
        self.assertEqual(target_row["m_volume"], 50867)
        self.assertAlmostEqual(target_row["m_wap"], 248.876, 2)
        os.remove(dest_filie_path)
