#!/usr/bin/env python3

import akshare as ak
import pandas as pd
import numpy as np


def futures_daily_info():
    # 郑商所
    data_frame = ak.get_czce_daily(date="20220608")
    data_frame.drop(columns=["date", "variety"], inplace=True)
    data_frame.fillna(0, inplace=True)
    data_frame.to_csv("czce.csv", index=None)

    # 金融期货
    data_frame = ak.get_cffex_daily(date="20220608")
    data_frame.drop(columns=["date", "variety"], inplace=True)
    data_frame.fillna(0, inplace=True)
    data_frame.to_csv("cffex.csv", index=None)


if __name__ == "__main__":
    futures_daily_info()
