#!/usr/bin/env python3

from datetime import datetime, timedelta, date
import os
import subprocess


def futures_fake_data():
    today = int(datetime.today().strftime("%Y%m%d"))
    data_dir = "/opt/sharedir/TickPool/"

    tick_pool_data = []
    files = os.listdir(data_dir)
    for each_file in files:
        if each_file.startswith("202"):
            tick_pool_data.append(int(each_file.split('.')[0]))

    date_data = []
    date_file = "/opt/sharedir/Meta/daily_time_list.txt"
    for each_line in open(date_file, "r"):
        each_line = int(each_line) // 1000000
        if each_line < today:
            date_data.append(each_line)

    return date_data, tick_pool_data


def gen(date_data: list, tick_pool_data: list):
    data_dir = "/opt/sharedir/TickPool/"
    for each_data in date_data:
        if each_data not in tick_pool_data:
            subprocess.call(("ln", "-s", data_dir + "00000000.dat", data_dir + str(each_data) + ".dat"))


def del_f():
    data_dir = "/opt/sharedir/TickPool/"
    for each_file in os.listdir(data_dir):
        if os.path.islink(data_dir + each_file):
            subprocess.call(("rm", data_dir + each_file))


if __name__ == "__main__":
    all_data, real_data = futures_fake_data()
    # gen(all_data, real_data)
    del_f()
