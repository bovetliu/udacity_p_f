import unittest

import pandas as pd

from analytic import ta_indicators


class TestTAIndicators(unittest.TestCase):

    def test_get_frws(self):
        series = pd.Series([15, 15, 15, 16, 17, 18, 17, 16, 15, 15.5, 19, 21, 4], name='DEMO')
        future_rtn = ta_indicators.get_frws(series)

        self.assertAlmostEqual(0.0000, future_rtn[0], 4)
        self.assertAlmostEqual(0.0000, future_rtn[1], 4)
        self.assertAlmostEqual(0.0667, future_rtn[2], 4)
        self.assertAlmostEqual(0.0625, future_rtn[3], 4)
        self.assertAlmostEqual(0.0588, future_rtn[4], 4)
