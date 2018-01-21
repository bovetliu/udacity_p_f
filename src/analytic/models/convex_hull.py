"""
Feel that convex might generate meaningful results to identify where is nearest support.
"""
import math
from enum import Enum
import collections

from datetime import datetime
from analytic.models import convex_hull

import matplotlib.pyplot as plt


class Accuracy(Enum):
    SECOND = 1
    MINUTE = 2
    HOUR = 3
    DAY = 4


class HalfScan:

    def __init__(self, find_support=True):
        """

        :param initial_prices:
        :param find_support: means find counter clockwise turns
        """
        self.__all_prices = collections.OrderedDict()
        self.__last_pr_pt = None
        self.__hull = collections.deque([])
        self.__find_support = find_support

    def add(self, price_point):
        """
        add one point to all prices, and adjust existing points on hull, and add latest point to hull
        :param price_point:
        :return:
        """
        if not isinstance(price_point, PricePoint):
            raise TypeError("price_point must be an instance of convex_hull.PricePoint")
        if self.__last_pr_pt is not None and self.__last_pr_pt.timestamp() >= price_point.timestamp():
            raise ValueError("price point must be added in a strictly monotonely increasing regarding timestamp.")
        self.__last_pr_pt = price_point
        self.__all_prices[self.__last_pr_pt.timestamp()] = self.__last_pr_pt
        if len(self.__hull) < 2:
            self.__hull.append(price_point)
            return

        top = self.__hull.pop()
        if self.__find_support:
            while PricePoint.ccw(self.__hull[-1], top, price_point) <= 0:
                # TODO(Bowei) in future, add pop condition, to find local support or resistance
                top = self.__hull.pop()
        else:
            while PricePoint.ccw(self.__hull[-1], top, price_point) >= 0:
                top = self.__hull.pop()
        self.__hull.append(top)
        self.__hull.append(price_point)

    def add_all(self, price_points):
        raise Error("Not yet implemented")
        # defensive_cp = []
        # for pr in price_points:
        #     if not isinstance(pr, PricePoint):
        #         raise ValueError("element in price_points is not valid")
        #     if len(defensive_cp) > 0:
        #         if [defensive_cp[-1].timestamp() < pr.timestamp()]:
        #             raise ValueError("Price point is not in non-decreasing order")
        #     defensive_cp.append(pr)
        # for pr in defensive_cp:
        #     self.__all_prices[pr.timestamp()] = pr

    def quick_plot(self, ax=None):
        timestamps_01 = []
        pr_pts_02 = []
        for k, v in self.__all_prices.items():
            timestamps_01.append(k)
            pr_pts_02.append(v)
        prices_01 = [pt.price() for pt in pr_pts_02]
        timestamps_02 = []
        prices_02 = []
        for pr_pt in self.__hull:
            timestamps_02.append(pr_pt.timestamp())
            prices_02.append(pr_pt.price())
        if ax is not None:
            ax.plot(timestamps_01, prices_01, 'b-', timestamps_02, prices_02, 'g-')
        else:
            plt.plot(timestamps_01, prices_01, 'b-', timestamps_02, prices_02, 'g-')
        plt.show()


class PricePoint:

    def __init__(self, starting_timestamp: datetime, timestamp: datetime, price: float, accu: Accuracy):
        if not isinstance(starting_timestamp, datetime):
            raise TypeError("starting_timestamp must be type of datetime.datetime")
        if (not isinstance(timestamp, datetime)) or (price < 0.0):
            raise TypeError("datetime must be python datetime, and price must be float number")
        if not isinstance(accu, Accuracy):
            raise TypeError("accu must one of enum Accuracy, SECOND, MINUTE, DAY")
        self.__starting_timestamp = starting_timestamp
        self.__timestamp = timestamp
        self.__accu = accu
        self.__x = PricePoint.time_diff(self.__starting_timestamp, self.__timestamp, self.__accu)
        self.set_accu(accu)
        self.__price = float(price)

    def timestamp(self):
        return self.__timestamp

    def price(self):
        return self.__price

    def r(self) -> float:
        return math.sqrt(self.__x ** 2.0 + self.__price ** 2.0)

    def set_accu(self, accu):
        self.__accu = accu
        self.__x = PricePoint.time_diff(self.__starting_timestamp, self.__timestamp, accu)

    def theta(self):
        return math.atan2(self.__price, self.__x)

    def angle_to(self, that) -> float:
        if not isinstance(that, convex_hull.PricePoint):
            raise TypeError("only PricePoint accepted")
        if that.__accu != self.__accu:
            raise ValueError("only PricePoint with same accuracy can be calculated")
        return math.atan2(that.__price - self.__price, that.__x - self.__x)

    def dist_to(self, that) -> float:
        if not isinstance(that, convex_hull.PricePoint):
            raise TypeError("only PricePoint accepted")
        if that.__accu != self.__accu:
            raise ValueError("only PricePoint with same accuracy can be calculated")
        return math.sqrt((that.__x - self.__x) ** 2 + (that.__price - self.__price) ** 2)

    def compare_to(self, that) -> int:
        if self.__price < that.__price:
            return -1
        if self.__price > that.__price:
            return +1
        if self.__x < that.__x:
            return -1
        if self.__x > that.__x:
            return +1
        return 0

    def __eq__(self, other):
        if not isinstance(other, convex_hull.PricePoint):
            return False
        return self.__price == other.__price and self.__accu == other.__accu and self.__x == other.__x

    def __str__(self):
        return "PricePoint(starting_timestamp={}, timestamp={}, price={}, accu={})".format(
            self.__starting_timestamp, self.__timestamp, self.__price, self.__accu)

    @classmethod
    def time_diff(cls, starting_time, end_time, accu: Accuracy=None) -> int:
        """
        return a integer represent floor integer of timedelta using accuracy represented by accu
        :param starting_time: starting datetime
        :param end_time: ending datetime
        :param accu: accuracy enum
        :return: one floor rounded integer of timedelta
        """
        total_sec_delta = (end_time - starting_time).total_seconds()
        if accu == Accuracy.SECOND:
            return int(total_sec_delta)
        elif accu == Accuracy.DAY:
            return int(total_sec_delta / 86400)
        elif accu == Accuracy.HOUR:
            return int(total_sec_delta / 3600)
        elif accu == Accuracy.MINUTE:
            return int(total_sec_delta / 60)

    @classmethod
    def area2(cls, a, b, c):
        if not isinstance(a, convex_hull.PricePoint) or \
                (not isinstance(b, convex_hull.PricePoint)) or \
                (not isinstance(c, convex_hull.PricePoint)):
            raise TypeError("a, b, c must be all convex_hull.PricePoint")
        if (a.__accu != b.__accu) or (b.__accu != c.__accu):
            raise ValueError("a, b, c accuracy must be all the same.")
        return (b.__x - a.__x) * (c.__price - a.__price) - (b.__price - a.__price) * (c.__x - a.__x)

    @classmethod
    def ccw(cls, a, b, c) -> int:
        """
        Returns true if a→b→c is a counterclockwise turn.
        :param a:first point
        :param b:second point
        :param c:third point
        :return:{ -1, 0, +1 } if a→b→c is a { clockwise, collinear; counterclocwise } turn.
        """
        area2 = convex_hull.PricePoint.area2(a, b, c)
        return -1 if area2 < 0 else (0 if area2 == 0 else 1)
