#!/usr/bin/env python3

import warnings

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import graphviz

from scipy.stats import rankdata
import scipy.stats as stats

from gplearn import genetic
from gplearn.functions import make_function
from gplearn.genetic import SymbolicTransformer, SymbolicRegressor
from gplearn.fitness import make_fitness

from sklearn.utils import check_random_state
from sklearn.model_selection import train_test_split
import baostock as bs
from ta.volume import VolumeWeightedAveragePrice
import statsmodels.api as sm
from scipy.stats.mstats import zscore
import talib

from scipy.stats import spearmanr
import datetime
from numba import jit
import scipy.stats as stats
from sklearn.linear_model import LinearRegression
import math
import cloudpickle
import matplotlib.pyplot as plt

# 初始函数族
init_function = ['add', 'sub', 'mul', 'div', 'sqrt', 'log', 'inv', 'sin', 'max', 'min']


def _ts_beta(x1, x2, n):
    with np.errstate(divide='ignore', invalid='ignore'):
        try:
            if n[0] == n[1] and n[1] == n[2]:
                u'''need：(list1,list2,number)  return：number
                前 n 期样本 A 对 B 做回归所得回归系数'''
                list1 = x1.flatten().tolist()
                list2 = x2.flatten().tolist()
                n = int(n[0])
                list1 = np.array(list1[-n - 1:]).reshape(-1, 1)
                list2 = np.array(list2[-n - 1:]).reshape(-1, 1)
                linear_regression = LinearRegression()
                model = linear_regression.fit(list1, list2)
                res = linear_regression.coef_.tolist()[0]
                res = np.array(res + [0] * (len(x1) - len(linear_regression.coef_))).flatten()
                return res
            else:
                return np.zeros(x1.shape[0])
        except:
            return np.zeros(x1.shape[0])


def _ts_resid(x1, x2, n):
    with np.errstate(divide='ignore', invalid='ignore'):
        try:
            if n[0] == n[1] and n[1] == n[2]:
                u'''need：(list1,list2,number)  return：number
                前 n 期样本 A 对 B 做回归所得的残差'''
                list1 = x1.flatten().tolist()
                list2 = x2.flatten().tolist()
                n = int(n[0])
                list1 = np.array(list1[-n - 1:]).reshape(-1, 1)
                list2 = np.array(list2[-n - 1:]).reshape(-1, 1)
                linear_regression = LinearRegression()
                model = linear_regression.fit(list1, list2)
                res = list(linear_regression.intercept_)
                res = np.array(res + [0] * (len(x1) - len([linear_regression.intercept_]))).flatten()
                return res
            else:
                return np.zeros(x1.shape[0])
        except:
            return np.zeros(x1.shape[0])


def _corr(data1, data2, n):
    with np.errstate(divide='ignore', invalid='ignore'):
        try:
            if n[0] == n[1] and n[1] == n[2]:
                window = n[0]
                x1 = pd.Series(data1.flatten())
                x2 = pd.Series(data2.flatten())
                df = pd.concat([x1, x2], axis=1)
                temp = pd.Series()
                for i in range(len(df)):
                    if i <= window - 2:
                        temp[str(i)] = np.nan
                    else:
                        df2 = df.iloc[i - window + 1:i, :]
                        temp[str(i)] = df2.corr('spearman').iloc[1, 0]
                return np.nan_to_num(temp)
            else:
                return np.zeros(data1.shape[0])
        except:
            return np.zeros(data1.shape[0])


def SEQUENCE(n):
    with np.errstate(divide='ignore', invalid='ignore'):
        try:
            if n[0] == n[1] and n[1] == n[2]:
                a = n[0]
                res = np.array(list(range(1, a + 1)))
                res = res.tolist() + [0] * (len(n) - len(res))
                return np.array(res).flatten()
            else:
                return np.zeros(n.shape[0])
        except:
            return np.zeros(n.shape[0])


def _ts_prod(x, n):
    with np.errstate(divide='ignore', invalid='ignore'):
        try:
            if n[0] == n[1] and n[1] == n[2]:
                x = pd.Series(x.flatten())
                n = n[0]
                res = x.prod(min_count=n)
                res = np.array([res] + [0] * (len(x) - 1))
                return res
            else:
                return np.zeros(x.shape[0])
        except:
            return np.zeros(x.shape[0])


# def _ts_cov(x, y, window):
#     with np.errstate(divide='ignore', invalid='ignore'):
#         try:
#             if n[0] == n[1] and n1[1] == n[2]:
#                 x = pd.Series(x.flatten())
#                 y = pd.Series(y.flatten())
#                 res = x.rolling(window).cov(y).fillna(0).values
#                 return res
#             else:
#                 return np.zeros(x.shape[0])
#         except:
#             return np.zeros(x.shape[0])


def _ts_lowday(x1, n1):
    with np.errstate(divide='ignore', invalid='ignore'):
        try:
            if n1[0] == n1[1] and n1[1] == n1[2]:
                '''need：(list,number)  return：number
                计算 A 前 n 期时间序列中最小值距离当前时点的间隔'''
                n = int(n1[0])
                x = x1.flatten().tolist()
                lowday = n - x[-n:].index(min(list(x)[-n:])) - 1
                return np.array([lowday] + [0] * (len(x) - 1))
            else:
                return np.zeros(x1.shape[0])
        except:
            return np.zeros(x1.shape[0])


def _ts_highday(x1, n1):
    with np.errstate(divide='ignore', invalid='ignore'):
        try:
            if n1[0] == n1[1] and n1[1] == n1[2]:
                u'''need：(list,number)  return：number
                计算 A 前 n 期时间序列中最大值距离当前时点的间隔'''
                n = int(n1[0])
                x = x1.flatten().tolist()
                highday = n - x[-n:].index(max(list(x)[-n:])) - 1
                return np.array([highday] + [0] * (len(x) - 1))
            else:
                return np.zeros(x1.shape[0])
        except:
            return np.zeros(x1.shape[0])


# def _ts_adv(x, d):
#     with np.errstate(divide='ignore', invalid='ignore'):
#         try:
#             if d[0] == d[1] and d[1] == d[2]:
#                 d = int(d[0])
#                 reslist(map(lambda t: np.mean(x.flatten()[t:t + 20]), range(len(x.flatten()) - d + 1)))
#                 res = res[np.isneginf(res)] = 0
#                 return np.array(res)
#             else:
#                 return np.zeros(x.shape[0])
#         except:
#             return np.zeros(x.shape[0])


def _ts_wma(x1, d):
    with np.errstate(divide='ignore', invalid='ignore'):
        try:
            if d[0] == d[1] and d[1] == d[2]:
                d = int(d[0])
                x1 = pd.Series(x1.flatten())
                weights = np.arange(1, d + 1)
                wma = x1.rolling(d).apply(lambda prices: np.dot(prices, weights) / weights.sum(), raw=True)
                wma = wma.fillna(0).values
                return wma
            else:
                return np.zeros(x1.shape[0])
        except:
            return np.zeros(x1.shape[0])


def _ts_mean(X, d):
    with np.errstate(divide='ignore', invalid='ignore'):
        try:
            if d[0] == d[1] and d[1] == d[2]:
                X = pd.Series(X.flatten())
                res = X.rolling(window=d[0]).mean()
                res = res.fillna(0).values
                return res
            else:
                return np.zeros(X.shape[0])
        except:
            return np.zeros(X.shape[0])


def _decay_linear(data1, n):
    with np.errstate(divide='ignore', invalid='ignore'):
        try:
            if n[0] == n[1] and n[1] == n[2]:
                n = int(n[0])
                w = np.arange(1, n + 1)
                w = w[::-1]
                w = np.array(w) / np.sum(w)
                res = list(map(lambda t: np.sum(np.multiply(np.array(data1.flatten()[t:t + n]), w)), range(len(data1.flatten()) - n + 1)))
                return np.array(res).flatten()
            else:
                return np.zeros(data1.shape[0])
        except:
            return np.zeros(data1.shape[0])


def _abs(x1):
    return abs(x1.flatten())


def _ts_abs(data1, n):
    with np.errstate(divide='ignore', invalid='ignore'):
        try:
            if n[0] == n[1] and n[1] == n[2]:
                window = int(n[0])
                x1 = data1.flatten()
                temp1 = x1[-window:]
                temp1 = np.abs(temp1)
                temp2 = x1[:-window]
                result = np.append(temp2, temp1)
                return result
            else:
                return np.zeros(data1.shape[0])
        except:
            return np.zeros(data1.shape[0])


def _ts_scale(data1, n):
    with np.errstate(divide='ignore', invalid='ignore'):
        try:
            if n[0] == n[1] and n[1] == n[2]:
                return int(n[0]) * data1 / np.nansum(np.abs(data1.flatten()))
            else:
                return np.zeros(data1.shape[0])
        except:
            return np.zeros(data1.shape[0])


def _signedpower(data1, n):
    with np.errstate(divide='ignore', invalid='ignore'):
        try:
            if n[0] == n[1] and n[1] == n[2]:
                return np.sign(data1.flatten()) * np.power(np.abs(data1.flatten()), int(n[0]))
            else:
                return np.zeros(data1.shape[0])
        except:
            return np.zeros(data1.shape[0])


def _ts_delta(data1, n):
    with np.errstate(divide='ignore', invalid='ignore'):
        try:
            if n[0] == n[1] and n[1] == n[2]:
                a1 = data1 - pd.Series(data1.flatten()).shift(periods=int(n[0]))
                a1 = a1.fillna(0).values
                return a1
            else:
                return np.zeros(data1.shape[0])
        except:
            return np.zeros(data1.shape[0])


def _ts_delay(data1, n):
    with np.errstate(divide='ignore', invalid='ignore'):
        try:
            if n[0] == n[1] and n[1] == n[2]:
                a1 = pd.Series(data1.flatten())
                a1 = a1.shift(periods=int(n[0]))
                a1 = a1.fillna(0).values
                return a1
            else:
                return np.zeros(data1.shape[0])
        except:
            return np.zeros(data1.shape[0])


def _ts_corr(data1, data2, n):
    with np.errstate(divide='ignore', invalid='ignore'):
        try:
            if n[0] == n[1] and n[1] == n[2]:
                window = int(n[0])
                x1 = pd.Series(data1.flatten())
                x2 = pd.Series(data2.flatten())
                df = pd.concat([x1, x2], axis=1)
                temp = pd.Series()
                for i in range(len(df)):
                    if i <= window - 2:
                        temp[str(i)] = np.nan
                    else:
                        df2 = df.iloc[i - window + 1:i, :]
                        temp[str(i)] = df2.corr('spearman').iloc[1, 0]
                return np.nan_to_num(temp)
            else:
                return np.zeros(data1.shape[0])
        except:
            return np.zeros(data1.shape[0])


def _ts_sum(data, n):
    with np.errstate(divide='ignore', invalid='ignore'):
        try:
            if n[0] == n[1] and n[1] == n[2]:
                window = int(n[0])
                value = np.array(pd.Series(data.flatten()).rolling(window).sum().tolist())
                value = np.nan_to_num(value)
                return value
            else:
                return np.zeros(data.shape[0])
        except:
            return np.zeros(data.shape[0])


def _ts_sma(data, n):
    with np.errstate(divide='ignore', invalid='ignore'):
        try:
            if n[0] == n[1] and n[1] == n[2]:
                window = int(n[0])
                value = np.array(pd.Series(data.flatten()).rolling(window).mean().tolist())
                value = np.nan_to_num(value)
                return value
            else:
                return np.zeros(data.shape[0])
        except:
            return np.zeros(data.shape[0])


def _ts_stddev(data, n):
    with np.errstate(divide='ignore', invalid='ignore'):
        try:
            if n[0] == n[1] and n[1] == n[2]:
                window = int(n[0])
                value = np.array(pd.Series(data.flatten()).rolling(window).std().tolist())
                value = np.nan_to_num(value)
                return value
            else:
                return np.zeros(data.shape[0])
        except:
            return np.zeros(data.shape[0])


def _ts_rank(x1, n):
    with np.errstate(divide='ignore', invalid='ignore'):
        try:
            if n[0] == n[1] and n[1] == n[2]:
                n = int(n[0])
                x2 = pd.Series(x1.flatten()[-n:])
                res = x2.rank(pct=True).values
                res = res.tolist() + [0] * (x1.shape[0] - res.shape[0])
                return np.array(res)
            else:
                return np.zeros(x1.shape[0])
        except:
            return np.zeros(x1.shape[0])


def _ts_argmin(data, n):
    with np.errstate(divide='ignore', invalid='ignore'):
        try:
            if n[0] == n[1] and n[1] == n[2]:
                window = int(n[0])
                value = pd.Series(data.flatten()).rolling(window).apply(np.argmin) + 1
                value = np.nan_to_num(value)
                return value
            else:
                return np.zeros(data.shape[0])
        except:
            return np.zeros(data.shape[0])


def _ts_argmax(data, n):
    with np.errstate(divide='ignore', invalid='ignore'):
        try:
            if n[0] == n[1] and n[1] == n[2]:
                window = int(n[0])
                value = pd.Series(data.flatten()).rolling(window).apply(np.argmax) + 1
                value = np.nan_to_num(value)
                return value
            else:
                return np.zeros(data.shape[0])
        except:
            return np.zeros(data.shape[0])


def _ts_min(data, n):
    import numpy as np
    with np.errstate(divide='ignore', invalid='ignore'):
        try:
            if n[0] == n[1] and n[1] == n[2]:
                window = int(n[0])
                value = np.array(pd.Series(data.flatten()).rolling(window).min().tolist())
                value = np.nan_to_num(value)
                return value
            else:
                return np.zeros(data.shape[0])
        except:
            return np.zeros(data.shape[0])


def _ts_max(data, n):
    with np.errstate(divide='ignore', invalid='ignore'):
        try:
            if n[0] == n[1] and n[1] == n[2]:
                window = int(n[0])
                value = np.array(pd.Series(data.flatten()).rolling(window).max().tolist())
                value = np.nan_to_num(value)
                return value
            else:
                return np.zeros(data.shape[0])
        except:
            return np.zeros(data.shape[0])


@jit
def _beta(x1, x2):
    list1 = x1.flatten().tolist()
    list2 = x2.flatten().tolist()
    list1 = np.array(list1).reshape(-1, 1)
    list2 = np.array(list2).reshape(-1, 1)
    linreg = LinearRegression()
    model = linreg.fit(list1, list2)
    res = linreg.coef_.tolist()[0]
    res = np.array(res + [0] * (len(x1) - len(linreg.coef_))).flatten()
    return res


def _resid(x1, x2):
    list1 = x1.flatten().tolist()
    list2 = x2.flatten().tolist()
    list1 = np.array(list1).reshape(-1, 1)
    list2 = np.array(list2).reshape(-1, 1)
    linreg = LinearRegression()
    model = linreg.fit(list1, list2)
    res = linreg.coef_.tolist()[0]
    res = np.array(res + [0] * (len(x1) - len(linreg.intercept_))).flatten()
    return res


def _beta(x1, x2):
    list1 = x1.flatten().tolist()
    list2 = x2.flatten().tolist()
    list1 = np.array(list1).reshape(-1, 1)
    list2 = np.array(list2).reshape(-1, 1)
    linreg = LinearRegression()
    model = linreg.fit(list1, list2)
    res = linreg.coef_.tolist()[0]
    res = np.array(res + [0] * (len(x1) - len(linreg.coef_))).flatten()
    return res


def _abs(x1):
    return abs(x1.flatten())


def _rank(x1):
    x1 = pd.Series(x1.flatten())
    return x1.rank(pct=True).values


def _cube(data):
    return np.square(data.flatten()) * data.flatten()


def _square(data):
    return np.square(data.flatten())


def _ts_argmaxmin(data, n):
    try:
        return _ts_argmax(data, n) - _ts_argmin(data, n)
    except:
        return np.zeros(data.shape[0])


ts_beta = make_function(function=_ts_beta, name='ts_beta', arity=3, wrap=False)
beta = make_function(function=_beta, name='beta', arity=2, wrap=False)
ts_resid = make_function(function=_ts_resid, name='ts_resid', arity=3, wrap=False)
resid = make_function(function=_resid, name='resid', arity=2, wrap=False)
corr = make_function(function=_corr, name='corr', arity=3, wrap=False)
SEQUENCE = make_function(function=SEQUENCE, name='SEQUENCE', arity=1, wrap=False)
# ts_cov = make_function(function=_ts_cov, name='ts_cov', arity=3, wrap=False)
ts_lowday = make_function(function=_ts_lowday, name='ts_lowday', arity=2, wrap=False)
ts_highday = make_function(function=_ts_highday, name='ts_highday', arity=2, wrap=False)
# ts_adv = make_function(function=_ts_adv, name='ts_adv', arity=2, wrap=False)
ts_wma = make_function(function=_ts_wma, name='ts_wma', arity=2, wrap=False)
ts_mean = make_function(function=_ts_mean, name='ts_mean', arity=2, wrap=False)
decay_linear = make_function(function=_decay_linear, name='decay_linear', arity=2, wrap=False)
_abs = make_function(function=_abs, name='_abs', arity=1, wrap=False)
ts_abs = make_function(function=_ts_abs, name='ts_abs', arity=2, wrap=False)
rank = make_function(function=_rank, name='rank', arity=1, wrap=False)
ts_scale = make_function(function=_ts_scale, name='ts_scale', arity=2, wrap=False)
ts_delta = make_function(function=_ts_delta, name='ts_delta', arity=2, wrap=False)
ts_delay = make_function(function=_ts_delay, name='ts_delay', arity=2, wrap=False)
ts_sma = make_function(function=_ts_sma, name='ts_sma', arity=2, wrap=False)
ts_std = make_function(function=_ts_stddev, name='ts_std', arity=2, wrap=False)
ts_rank = make_function(function=_ts_rank, name='ts_rank', arity=2, wrap=False)
ts_stddev = make_function(function=_ts_stddev, name='ts_stddev', arity=2, wrap=False)
ts_sum = make_function(function=_ts_sum, name='ts_sum', arity=2, wrap=False)
ts_corr = make_function(function=_ts_corr, name='ts_corr', arity=3, wrap=False)
ts_min = make_function(function=_ts_min, name='ts_min', arity=2, wrap=False)
cube = make_function(function=_cube, name='cube', arity=1, wrap=False)
square = make_function(function=_square, name='square', arity=1, wrap=False)
ts_argmaxmin = make_function(function=_ts_argmaxmin, name='ts_argmaxmin', arity=2, wrap=False)
ts_argmax = make_function(function=_ts_argmax, name='ts_argmax', arity=2)
ts_argmin = make_function(function=_ts_argmin, name='ts_argmin', arity=2, wrap=False)
ts_min = make_function(function=_ts_min, name='ts_min', arity=2, wrap=False)
ts_max = make_function(function=_ts_max, name='ts_max', arity=2, wrap=False)
# decay_linear,ts_corr(replaced by corr),

init_function = ['add', 'sub', 'mul', 'div', 'sqrt', 'log', 'inv', 'sin', 'max', 'min']
user_function = [SEQUENCE, ts_lowday, ts_highday, ts_wma, ts_mean, _abs, rank, ts_scale, ts_delta, ts_delay,
                 ts_sma, ts_std, ts_rank, ts_sum, corr, ts_min, cube, square, ts_argmaxmin, ts_argmax, ts_argmin, ts_min, ts_max, ts_beta,
                 beta, ts_resid, resid]
