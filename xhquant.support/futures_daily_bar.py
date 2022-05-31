#!/usr/bin/env python3

from datetime import datetime
import akshare as ak


def futures_daily_bar():
    today = datetime.today()
    start_date = today.strftime("%Y%m%d")
    end_date = today.strftime("%Y%m%d")
    exchange = ["CFFEX", "CZCE", "SHFE", "DCE", "INE"]
    print(ak.get_futures_daily(start_date=start_date, end_date=end_date))


if __name__ == "__main__":
    futures_daily_bar()
