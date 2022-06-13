#!/usr/bin/env python3

from datetime import datetime
import os

import akshare as ak
import pandas as pd
import numpy as np

import xhsupport.futures_common as futures_common


def futures_exchange_daily_info(date: str):
    data_frame_list = []

    # 郑商所
    data_frame = ak.get_czce_daily(date=date)
    data_frame.drop(columns=["date", "variety"], inplace=True)
    data_frame.set_index("symbol", inplace=True)
    data_frame_list.append(data_frame)

    # 金融期货
    data_frame = ak.get_cffex_daily(date=date)
    data_frame.drop(columns=["date", "variety"], inplace=True)
    data_frame.set_index("symbol", inplace=True)
    data_frame_list.append(data_frame)

    # 大商所
    data_frame = ak.get_dce_daily(date=date)
    data_frame.drop(columns=["date", "variety"], inplace=True)
    data_frame.set_index("symbol", inplace=True)
    data_frame.index = data_frame.index.str.lower()
    data_frame_list.append(data_frame)

    # 上期所
    data_frame = ak.get_shfe_daily(date=date)
    data_frame.drop(columns=["date", "variety"], inplace=True)
    data_frame.set_index("symbol", inplace=True)
    data_frame.index = data_frame.index.str.lower()
    data_frame_list.append(data_frame)

    exchange_daily_info = pd.concat(data_frame_list, axis=0)
    return exchange_daily_info


def futures_daily_info(date: str):
    exchange_daily_info = futures_exchange_daily_info(date=date)

    contract = futures_common.futures_contract()
    array_value = np.full([len(contract), 9], pd.NA)
    array_column = ["open", "high", "low", "close", "volume", "open_interest", "turnover", "settle", "pre_settle"]
    nan_data_info = pd.DataFrame(array_value, columns=array_column, index=contract)
    nan_data_info.index.name = "symbol"
    for each_contract in exchange_daily_info.index:
        if each_contract not in nan_data_info.index:
            exchange_daily_info.drop([each_contract], inplace=True)

    daily_info = nan_data_info.combine_first(exchange_daily_info)
    daily_info.fillna(-1, inplace=True)

    # 整理顺序
    daily_info = daily_info.reset_index()
    daily_info["symbol"] = daily_info["symbol"].astype('category')
    daily_info["symbol"].cat.reorder_categories(contract, inplace=True)
    daily_info.sort_values('symbol', inplace=True)
    daily_info.set_index("symbol", inplace=True)

    return daily_info


if __name__ == "__main__":
    # today = datetime.today().strftime("%Y%m%d")
    today = "20220610"
    file_name = "/opt/sharedir/DailyCSV/" + today + "_daily_bar.csv"

    daily_information = futures_daily_info(today)
    daily_information.to_csv(file_name)

    # 处理nan
    daily_information = pd.read_csv(file_name, index_col="symbol")
    daily_information.fillna(-1, inplace=True)
    daily_information.to_csv(file_name)
