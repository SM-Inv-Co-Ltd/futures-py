import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


def factors_correlation_coefficient(factor_data_1: pd.DataFrame, factor_data_2: pd.DataFrame):
    """
    因子之间的自相关系数
    :param factor_data_1: Dataframe格式的因子1
    :param factor_data_2: Dataframe格式的因子2
    :return:
    """
    if len(factor_data_1) != len(factor_data_2):
        print("两个因子的时间长度不一致,请检查.")

    columns = list(factor_data_1.columns.intersection(factor_data_2.columns))
    factor_data_1 = factor_data_1[columns]
    factor_data_2 = factor_data_2[columns]

    correlation_coefficient = []
    for i in range(len(factor_data_1)):
        correlation_coefficient.append(stats.spearmanr(factor_data_1.iloc[i].values, factor_data_2.iloc[i].values)[0])

    plt.figure(figsize=(12, 6))
    plt.title("Factor Correlation Coefficient")
    plt.plot(correlation_coefficient, color="forestgreen", linestyle='-', linewidth=1, label="Corr")
    plt.legend(loc="best")

    plt.savefig("factor_correlation_coefficient.pdf", bbox_inches='tight')

    return np.mean(correlation_coefficient)
