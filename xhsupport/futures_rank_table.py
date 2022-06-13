#!/usr/bin/env python3

from datetime import datetime
import os

import akshare as ak
import pandas as pd
import numpy as np

import xhsupport.futures_common as futures_common


def futures_exchange_rank_table(date: str):
    exchange_rank_table = ak.get_rank_sum_daily(start_day=date, end_day=date)
    columns = ["symbol",
               "vol_top5", "vol_chg_top5",
               "long_open_interest_top5", "long_open_interest_chg_top5", "short_open_interest_top5", "short_open_interest_chg_top5",
               "vol_top10,vol_chg_top10",
               "long_open_interest_top10", "long_open_interest_chg_top10", "short_open_interest_top10", "short_open_interest_chg_top10",
               "vol_top15", "vol_chg_top15",
               "long_open_interest_top15", "long_open_interest_chg_top15", "short_open_interest_top15", "short_open_interest_chg_top15",
               "vol_top20,vol_chg_top20",
               "long_open_interest_top20", "long_open_interest_chg_top20", "short_open_interest_top20", "short_open_interest_chg_top20"]
    exchange_rank_table = exchange_rank_table[columns]
    exchange_rank_table.set_index("symbol", inplace=True)

    return exchange_rank_table


if __name__ == "__main__":
    futures_exchange_rank_table("20220610")
