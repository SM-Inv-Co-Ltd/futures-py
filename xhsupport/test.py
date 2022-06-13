#!/usr/bin/env python3

from datetime import datetime
import os

import pandas as pd
import numpy as np


def test():
    s = pd.Series({'a': 1, 'b': 2, 'c': 3})
    list_custom = ['b', 'a', 'c']
    df = pd.DataFrame(s)
    print(df)
    df = df.reset_index()
    print(df)
    df.columns = ['words', 'number']
    print(df)
    df['words'] = df['words'].astype('category')
    df['words'].cat.reorder_categories(list_custom, inplace=True)
    print(df)

    # inplace = True，使 df生效
    df.sort_values('words', inplace=True)
    print(df)
    df.set_index("words", inplace=True)
    print(df)


if __name__ == "__main__":
    test()
