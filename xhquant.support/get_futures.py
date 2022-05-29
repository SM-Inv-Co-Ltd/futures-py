#!/usr/bin/env python3

from datetime import datetime, timedelta, date
from dateutil.relativedelta import relativedelta


def get_futures():
    today = datetime.today()
    # today = date(2022, 3, 15)
    if datetime.now().hour > 16:
        today += timedelta(days=1)

    all_dta = {}
    for i in range(12):
        dta = today + relativedelta(months=i)
        year = str(dta.year)[2:]
        month = str(dta.month)
        if len(month) == 1:
            month = '0' + month
        all_dta[year + month] = month
    # 生成按月份排序的日期
    all_dta = sorted(all_dta.items(), key=lambda asd: asd[1])

    # 读取price_tick文件
    price_tick_file = open("/opt/sharedir/Meta/price_tick.txt", "r")
    tkrs = []
    exchange = []
    multi = []
    base_point = []
    for each_line in price_tick_file:
        each_line = each_line.strip().split()
        tkrs.append(each_line[0])
        exchange.append(each_line[1])
        multi.append(each_line[2].strip())
        base_point.append(each_line[3].strip())

    universe_file = open("/opt/sharedir/Meta/universe.txt", "w")
    for iNum in range(len(tkrs)):
        for each_dta in all_dta:
            contract = tkrs[iNum] + each_dta[0]
            j = 0
            for each_char in contract:
                if '0' <= each_char <= '9':
                    break
                j += 1
            if exchange[iNum] == "SHFE" or exchange[iNum] == "INE" or exchange[iNum] == "DCE":
                contract = contract.lower()
            if exchange[iNum] == "CZCE":
                contract = contract[0:j] + contract[j + 1:]
            print("{0:10} {1:10} {2:10} {3:10}".format(contract, exchange[iNum], multi[iNum], base_point[iNum]), file=universe_file)


if __name__ == "__main__":
    get_futures()
