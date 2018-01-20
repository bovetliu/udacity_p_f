import unittest
import math
from datetime import timedelta, datetime, timezone
from analytic import time_utils
from analytic.models.convex_hull import PricePoint, Accuracy


class TestConvexHull(unittest.TestCase):

    def test_py_datetime_module(self):
        datetime_with_tzinfo_01 = datetime(2018, 1, 20, 14, 45, 0, tzinfo=time_utils.tz_est)
        datetime_without_tzinfo_02 = datetime(2018, 1, 20, 14, 45, 0)
        print(datetime_with_tzinfo_01)
        print(datetime_without_tzinfo_02)

        datetime_with_tzinfo_02 = datetime(2018, 1, 19, 15, 0, 0, tzinfo=time_utils.tz_est)
        delta = datetime_with_tzinfo_02 - datetime_with_tzinfo_01
        print(delta)

    def test_price_point(self):
        self.assertAlmostEqual(0.25 * math.pi, math.atan2(1, 1), 2)
        # print(math.atan2(1, 0))
        start_datetime = datetime(2018, 1, 20, 9, 30, 0)
        time_01 = datetime(2018, 1, 20, 9, 45)
        time_02 = datetime(2018, 1, 20, 10, 41)
        accu = Accuracy.MINUTE
        # 2018-01-20T09:45:00  price 10.00
        pp01 = PricePoint(start_datetime, time_01, 10, accu)

        # 2018-01-20T10:41:00  price 10.9
        pp02 = PricePoint(start_datetime, time_02, 10.9, accu)

        self.assertEqual(56, PricePoint.time_diff(time_01, time_02, accu))
        self.assertAlmostEqual(0.0160700450850256, pp01.angle_to(pp02), 5)

