#!/usr/bin/env python3


def futures_contract():
    contract = []
    file = open("/opt/sharedir/Meta/universe.txt", "r")
    for line in file:
        line = line.strip("\n")
        contract.append(line.split()[0])
    return contract


def futures_exchange():
    exchange = []
    file = open("/opt/sharedir/Meta/universe.txt", "r")
    for line in file:
        line = line.strip("\n")
        exchange.append(line.split()[1])
    return exchange


def futures_daily_time_list():
    daily_time_list = []
    file = open("/opt/sharedir/Meta/daily_time_list.txt", "r")
    for line in file:
        line = line.strip("\n")
        daily_time_list.append(int(line) // 1000000)
    return daily_time_list


if __name__ == "__main__":
    print(futures_contract())
    print(futures_exchange())
    print(futures_daily_time_list())
