import os
import shutil
import unittest

import pandas as pd
import requests
from requests import HTTPError

from analytic.data import tws_data_api_connector
from analytic.data.tws_data_api_connector import BarSize
from analytic.utility import RAW_DATA_PATH
from analytic.utility import TEST_RESOURCES_PATH


class TestTwsDataApiConnector(unittest.TestCase):

    def test_get_historical_data_prices(self):
        symbol = "NVDA"
        end_local_date_str = "20180524 16:01:00"
        num_of_bars = 550
        res_data = tws_data_api_connector.get_historical_data_prices(symbol, end_local_date_str, num_of_bars, BarSize.DAY_1, True)
        self.assertEqual(num_of_bars, len(res_data))
        last_row = res_data.iloc[-1]
        # print(last_row)
        self.assertEqual(last_row["m_close"], 247.69)
        self.assertEqual(last_row["m_count"], 35770.00)
        self.assertEqual(last_row["m_high"], 249.40)
        self.assertEqual(last_row["m_low"], 245.24)
        self.assertEqual(last_row["m_volume"], 77090)
        self.assertEqual(last_row["m_wap"], 247.63)

        num_of_bars = 400
        res_data = tws_data_api_connector.get_historical_data_prices(symbol, end_local_date_str, num_of_bars, BarSize.MIN_1, True)
        self.assertEqual(23, res_data.index[0].day)
        # print(res_data)

    def test_sync_sombol_to_local(self):
        symbol = "NVDA"
        target_file = os.path.join(RAW_DATA_PATH, "{}-TWS-DATA-DAILY.csv".format(symbol))
        tws_data_api_connector.get_local_synced(symbol)
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

    def test_query_index(self):
        symbols = tws_data_api_connector.query_symbol_list("sp500")
        self.assertEqual(505, len(symbols), "without supplying query, should get a length of 505")

        # search companies of which the name has "apple" or "advanced"
        symbols = tws_data_api_connector.query_symbol_list("sp500", query="apple")
        self.assertTrue("AAPL" in symbols)
        self.assertFalse("DPS" in symbols)
        symbols = tws_data_api_connector.query_symbol_list("sp500", query=["apple", "advanced", "asdf"])
        self.assertTrue("AAPL" in symbols)
        self.assertTrue("AMD" in symbols)
        symbols = tws_data_api_connector.query_symbol_list("sp500", query=["apple", "advanced", "asdf"],
                                                           queried_column="Company")
        self.assertTrue("AAPL" in symbols)
        self.assertTrue("AMD" in symbols)

    # @unittest.skip  # no reason needed
    def test_syn_sp500(self):
        symbols = set(tws_data_api_connector.query_symbol_list("spdr_sectors", return_df=False))
        symbols.add("SPY")
        problematic_symbols = []
        for symbol in symbols:
            print("going to sync {};".format(symbol))
            try:
                tws_data_api_connector.get_local_synced(symbol, num_of_days_needed=600)
            except (HTTPError, requests.exceptions.Timeout) as err:
                print("one error, symbol : {}".format(symbol))
                print("error: {}".format(str(err)))
                problematic_symbols.append(symbol)
        if len(problematic_symbols) > 0:
            print("problematic symbols: ")
            print(problematic_symbols)
