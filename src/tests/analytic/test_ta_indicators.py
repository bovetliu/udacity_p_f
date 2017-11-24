import unittest

import pandas as pd

from analytic import ta_indicators


class TestTAIndicators(unittest.TestCase):

    def test_get_frws(self):
        series = pd.Series([15, 15, 15, 16, 17, 18, 17, 16, 15, 15.5, 19, 21, 4], name='DEMO')
        frs = ta_indicators.get_frws(series, 3)
        print(frs)