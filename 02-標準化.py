#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = "WCH"

import pandas as pd
import numpy as np
import myfun_CrimeRates
from sklearn.preprocessing import MinMaxScaler
from pandas import ExcelWriter  # 匯入 excel writer
myfun_CrimeRates.matplot_中文字()

#### 資料來源 https://www.kaggle.com/datasets/mguzmann/swedishcrime
df = pd.read_excel("CrimeRates_資料清洗後.xlsx",0)
print(df.shape)
print(df.info)
print(df.describe())

print("Year:",df["Year"].tolist())
print("crimes.person:",df["crimes.person"].tolist())
print("population:",df["population"].tolist())
print("drunk.driving:",df["drunk.driving"].tolist())

"""
listcrimes.person=myfun_CrimeRates.pandas_取得裡面的種類(df,"crimes.person")
#listcrimes.person.sort()
listGender=myfun_CrimeRates.pandas_取得裡面的種類(df,"Gender")
listSatisfy=myfun_CrimeRates.pandas_取得裡面的種類(df,"JobSatisfaction")
listJobRole=myfun_CrimeRates.pandas_取得裡面的種類(df,"JobRole")
"""

print("=============數據 標準化=============")
#   Pandas  x 轉 numpy
x=df.to_numpy()
print(x)

#  標準化 x
scaler = MinMaxScaler(feature_range=(0,1))     # 初始化 # 設定縮放的區間上下限
scaler.fit(x)                                   # 找標準化範圍
x= scaler.transform(x)                          # 把資料轉換
print("標準化:",x)

# numpy 轉 Pandas
df = pd.DataFrame(x, columns=df.columns)

writer = ExcelWriter('CrimeRates資料清洗後標準化.xlsx', engine='xlsxwriter')      # 另存為資料清洗後
df.to_excel(writer, sheet_name='資料清洗後標準化',index=False,header=1)   # 分頁欄位的名稱 header=1 要印表頭
writer.save()





