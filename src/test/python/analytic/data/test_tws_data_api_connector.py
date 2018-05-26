import unittest
from analytic.data.tws_data_api_connector import BarSize
from analytic.data import tws_data_api_connector


class TestTwsDataApiConnector(unittest.TestCase):

    def test_get_historical_data_prices(self):
        symbol = "NVDA"
        end_local_date_str = "20180524 16:00:00"
        res_data = \
            tws_data_api_connector.get_historical_data_prices(symbol, end_local_date_str, 5, BarSize.DAY_1, True)
        # TODO(Bowei) assertions pending

    def test_sync_sombol_to_local(self):
        symbol = "NVDA"
        tws_data_api_connector.sync_sombol_to_local(symbol)
        # TODO(Bowei) assertions pending
