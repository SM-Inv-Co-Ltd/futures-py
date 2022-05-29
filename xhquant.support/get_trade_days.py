#!/usr/bin/env python3


import datetime
import time
import jqdatasdk


def get_trade_days():
    date_list = jqdatasdk.get_trade_days(start_date="2022-01-01", end_date="2023-01-01")
    for each_date in date_list:
        result = datetime.date.strftime(each_date, "%Y%m%d")
        result += "150000"
        print(result)


if __name__ == "__main__":
    jqdatasdk.auth("18560457063", "457063")
    get_trade_days()
