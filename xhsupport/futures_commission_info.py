#!/usr/bin/env python3

from datetime import datetime
import os

import akshare as ak
import pandas as pd
import numpy as np

import xhsupport.futures_common as futures_common


def futures_exchange_commission_info():
    data_frame = ak.futures_comm_info()
    # 选取重要字段
    data_frame = data_frame[["合约代码", "保证金-买开", "保证金-卖开", "每跳毛利", "手续费", "每跳净利"]]
    data_frame.set_index("合约代码", inplace=True)

    return data_frame


def futures_commission_info():
    exchange_commission_info = futures_exchange_commission_info()

    contract = futures_common.futures_contract()
    array_value = np.full([len(contract), 5], pd.NA)
    array_column = ["保证金-买开", "保证金-卖开", "每跳毛利", "手续费", "每跳净利"]
    nan_data_info = pd.DataFrame(array_value, columns=array_column, index=contract)
    nan_data_info.index.name = "合约代码"
    for each_contract in exchange_commission_info.index:
        if each_contract not in nan_data_info.index:
            exchange_commission_info.drop([each_contract], inplace=True)

    commission_info = nan_data_info.combine_first(exchange_commission_info)
    commission_info.fillna(-1, inplace=True)

    # 整理顺序
    commission_info = commission_info.reset_index()
    commission_info["合约代码"] = commission_info["合约代码"].astype('category')
    commission_info['合约代码'].cat.reorder_categories(contract, inplace=True)
    commission_info.sort_values('合约代码', inplace=True)
    commission_info.set_index("合约代码", inplace=True)

    return commission_info


if __name__ == "__main__":
    commission_information = futures_commission_info()
    today = datetime.today().strftime("%Y%m%d")
    commission_information.to_csv("/opt/sharedir/DailyCSV/" + today + "_daily_commission.csv")
