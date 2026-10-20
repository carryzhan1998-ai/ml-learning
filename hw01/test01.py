import pandas as pd
df = pd.read_csv('covid_train.csv')
print(df.shape)                    # 多少行、多少列
print(df.columns.tolist())         # 所有列名
print(df.iloc[:3, :8])             # 前 3 行、前 8 列看一眼
print(df.dtypes.value_counts())    # 哪些是 float，哪些是 int

df = pd.read_csv('covid_train.csv')
for i, c in enumerate(df.columns):
    print(i, c)

import numpy as np
X = df.values[:, :-1]
for i in range(X.shape[1]):
    if X[:, i].std() < 1e-6:
        print(f'列 {i} ({df.columns[i]}) 几乎没有变化，可以删')