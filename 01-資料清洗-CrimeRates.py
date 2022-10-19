#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = "WCH"

import pandas as pd
import numpy as np
import myfun_CrimeRates
import itertools
from pandas import ExcelWriter  # 匯入 excel writer


#### 資料來源 https://www.kaggle.com/datasets/mguzmann/swedishcrime
df1 = pd.read_excel("reported.xlsx")            # read_some file formats 的type為dataframe
print(df1.columns)                              # 全部欄位名稱
print(df1.info())
print(df1.describe())

# 把空的資料補上平均

myfun_CrimeRates.pandas_nan_col_替換(df1,"house.theft",df1["house.theft"].mean().round(0))
myfun_CrimeRates.pandas_nan_col_替換(df1,"vehicle.theft",df1["vehicle.theft"].mean().round(0))
myfun_CrimeRates.pandas_nan_col_替換(df1,"out.of.vehicle.theft",df1["out.of.vehicle.theft"].mean().round(0))
myfun_CrimeRates.pandas_nan_col_替換(df1,"shop.theft",df1["shop.theft"].mean().round(0))
myfun_CrimeRates.pandas_nan_col_替換(df1,"narcotics",df1["narcotics"].mean().round(0))
myfun_CrimeRates.pandas_nan_col_刪掉表單內空值(df1)


writer = ExcelWriter('CrimeRates_資料清洗後.xlsx', engine='xlsxwriter')      # 另存為資料清洗後
df1.to_excel(writer, sheet_name='CrimeRates_資料清洗後',index=False,header=1)   # 分頁欄位的名稱 header=1 要印表頭
writer.save()




