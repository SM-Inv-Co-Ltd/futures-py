#!/usr/bin/env python3

# 先引入后面可能用到的包（package）
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import akshare as ak
import talib as ta

'''人气指标（AR）和意愿指标（BR）
AR指标是通过比较某个周期内开盘价与最高、最低价，来反映市场买卖人气。
计算公式为：N日AR=(N日内（H－O）之和）/(N日内（O－L）之和)*100。
BR指标是通过比较一段周期内收盘价在该周期价格波动中的地位，来反映市场买卖意愿程度。
计算公式为：N日BR=（N日内（H－YC）之和）/N日内（YC－L）之和）*100。
其中，O 为当日开盘价，H 为当日最高价，L 为当日最低价，YC 为前一交易日的收盘价，N 为设定的时间参数，一般原始参数日设定为 26 日，计算周期可以根据自己的经验或回测结果进行修正。
 
双方的分界线是 100，100 以上是多方优势，100 以下是空方优势。
买入信号：
BR通常运行在AR上方，一旦BR跌破AR并在AR之下运行时，表明市场开始筑底，视为买进信号；
BR<40,AR<60: 空方力量较强，但随时可能反转上涨，考虑买进。
 
卖出信号：
BR>400,AR>180，多方力量极强，但随时可能反转下跌，考虑卖出；
BR快速上升，AR并未上升而是小幅下降或横盘，视为卖出信号。
 
背离信号：
AR、BR指标的曲线走势与股价K线图上的走势正好相反。
 
顶背离：
当股价K线图上的股票走势一峰比一峰高，股价一直向上涨，而AR、BR指标图上的走势却一峰比一峰低，说明出现顶背离，股价短期内将高位反转，是比较强烈的卖出信号。
 
底背离：
当股价K线图上的股票走势一底比一底低，股价一直向下跌，而AR、BR指标图上的走势却一底比一底高，说明出现底背离，股价短期内将低位反转，是比较强烈的买入信号。
'''

# 正常显示画图时出现的中文和负号
from pylab import mpl

mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False

# 引入TA-Lib库
import talib as ta
from datetime import datetime, timedelta

index2 = {'上证综指': '000001', '深证成指': '399001',
          '沪深300': '000300', '创业板指': '399006',
          '上证50': '000016', '中证500': '000905',
          '中小板指': '399005', '上证180': '000010'}
index={'上证综指': '000001'}
n = 250
tineperiod=26
for code in index.values():
    t = datetime.now()
    t0 = t - timedelta(n)
    start = t0.strftime('%Y%m%d')
    end = t.strftime('%Y%m%d')
    index_zh_a_hist_df = ak.index_zh_a_hist(symbol=code, period="daily", start_date=start, end_date=end)
    # print(index_zh_a_hist_df)
    index_zh_a_hist_df.index = pd.to_datetime(index_zh_a_hist_df['日期'])
    df = index_zh_a_hist_df.sort_index()
    df['HO'] = df['最高'] - df['开盘']
    df['OL'] = df['开盘'] - df['最低']
    df['HCY'] = df['最高'] - df['收盘'].shift(1)
    df['CYL'] = df['收盘'].shift(1) - df['最低']
    # 计算AR、BR指标
    df['AR'] = ta.SUM(df.HO, timeperiod=tineperiod) / ta.SUM(df.OL, timeperiod=tineperiod) * 100
    df['BR'] = ta.SUM(df.HCY, timeperiod=tineperiod) / ta.SUM(df.CYL, timeperiod=tineperiod) * 100
    # print(df['AR'].values[tineperiod-1:])
    # print(df['BR'].values[tineperiod-1:])
    for i in df['AR'].values[tineperiod-1:]:
        for j in df['BR'].values[tineperiod-1:]:
            if i<=100 and j<=100:
                print('空方优势')
                if 60>i and i>j and j<40:
                    print('随时可能反转上涨，考虑买进')
            elif i>=100 and j>=100:
                print('多方优势')
                if 180 < i and i < j and j > 400:
                    print('可能反转下跌，考虑卖出')
    result= df[['收盘', 'AR', 'BR']].dropna()
    result['收盘'].plot(color='g', figsize=(14, 5))
    plt.xlabel('')
    plt.title(code + '价格走势', fontsize=15)
    df[['AR', 'BR']].plot(figsize=(14, 5))
    plt.xlabel('')
    plt.show()