from datetime import datetime, timezone, timedelta

__minus_five_hour = timedelta(minutes=-300)
__minus_four_hour = timedelta(minutes=-240)

tz_est = timezone(__minus_five_hour, "Eastern Standard Time")
tz_edt = timezone(__minus_four_hour, "Eastern Daylight Time")

IB_LOCAL_DATETIME_FORMAT = "%Y%m%d %H:%M:%S"
