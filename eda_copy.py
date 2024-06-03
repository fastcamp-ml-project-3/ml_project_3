import os
import gc
import time

import numpy as np
import pandas as pd
import statsmodels.api as sm

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from cachetools import cached, LRUCache

import warnings

cache = LRUCache(maxsize=1000)
@cached(cache)
def load():
    pass

pd.set_option('display.max_columns', None)
pd.options.display.float_format = '{:.2f}'.format
warnings.filterwarnings('ignore')

# Import
train = pd.read_csv("../data/train.csv")
test = pd.read_csv("../data//test.csv")
stores = pd.read_csv("../data//stores.csv")
#sub = pd.read_csv("../input/store-sales-time-series-forecasting/sample_submission.csv")   
transactions = pd.read_csv("../data//transactions.csv").sort_values(["store_nbr", "date"])

# Datetime
train["date"] = pd.to_datetime(train.date)
test["date"] = pd.to_datetime(test.date)
transactions["date"] = pd.to_datetime(transactions.date)

# Data types
train.onpromotion = train.onpromotion.astype("float16")
train.sales = train.sales.astype("float32")
stores.cluster = stores.cluster.astype("int8")

print(train.head())

# 데이터 병합
temp = pd.merge(train.groupby(["date", "store_nbr"]).sales.sum().reset_index(), transactions, how="left")

# 스피어만 상관 관계 계산
spearman_corr = temp.corr(method="spearman").loc["transactions", "sales"]
print(f"Spearman Correlation between Total Sales and Transactions: {spearman_corr:.4f}")

# 라인 차트 그리기
fig = px.line(transactions.sort_values(["store_nbr", "date"]), x='date', y='transactions', color='store_nbr', title="Transactions")
fig.show()


a = transactions.copy()
a["year"] = a.date.dt.year
a["month"] = a.date.dt.month
fig1 = px.box(a, x="year", y="transactions" , color = "month", title = "Transactions")
fig1.show()