#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = "WCH"

import pandas as pd
import numpy as np
import myfun_CrimeRates
from sklearn.preprocessing import MinMaxScaler
from pandas import ExcelWriter  # 匯入 excel writer
myfun_CrimeRates.matplot_中文字()
import seaborn as sns
import seaborn.objects as so
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use("TKAgg")


#### 資料來源 https://www.kaggle.com/datasets/mguzmann/swedishcrime

#### 參考資料
# https://www.kaggle.com/code/daesunryu/eda-crime-rate-in-sweden
# https://www.kaggle.com/code/sashirajc/swedish-crime-rates-data-exploration

df = pd.read_excel("CrimeRates_資料清洗後.xlsx",0)

print(df.shape)
print(df.info)
print(df.describe())

print("Year:",df["Year"].tolist())
print("crimes.person:",df["crimes.person"].tolist())
print("population:",df["population"].tolist())
print("drunk.driving:",df["drunk.driving"].tolist())

################
sns.relplot(data=df, x='Year', y='crimes.person', hue='population')

plt.subplots_adjust(left=None, bottom=None, right=None, top=0.9, wspace=None, hspace=None)  #  調整上面的 出血邊
plt.title('Crimes against Person', size=10)
plt.xlabel("Year", size=10)
plt.ylabel("Crime Rate", size=10)
plt.savefig("Crimes against Person.png")
plt.show()

#################
plt.plot(df['Year'],df['robbery'],label='robbery')
plt.plot(df['Year'],df['burglary'],label='burglary')
plt.plot(df['Year'],df['vehicle.theft'],label='vehicle.theft')
plt.plot(df['Year'],df['house.theft'],label='house.theft')
plt.plot(df['Year'],df['shop.theft'],label='shop.theft')
plt.plot(df['Year'],df['out.of.vehicle.theft'],label='out.of.vehicle.theft')
plt.xlabel('Year')
plt.ylabel('Number of thefts/robberies')
plt.legend()
plt.savefig("Number of thefts_robberies.png")
plt.show()

#################
plt.plot(df['Year'],df['robbery'],label='robbery')
plt.plot(df['Year'],df['burglary'],label='burglary')
plt.plot(df['Year'],df['vehicle.theft'],label='vehicle.theft')
plt.plot(df['Year'],df['house.theft'],label='house.theft')
plt.plot(df['Year'],df['shop.theft'],label='shop.theft')
plt.plot(df['Year'],df['out.of.vehicle.theft'],label='out.of.vehicle.theft')
plt.plot(df['Year'],df['crimes.total'],label='crimes.total')
plt.xlabel('Year')
plt.ylabel('Number of thefts/robberies')
plt.legend()
plt.savefig("crimes.total-Number of thefts_robberies.png")
plt.show()

################

data=df
sns.pairplot(data[["Year","crimes.penal.code","crimes.person","assault","stealing.general"]], diag_kind="crimes.total")
plt.savefig("1.png")
plt.show()

################

g = sns.PairGrid(data[["Year","crimes.penal.code","crimes.person","assault","stealing.general"]])
g.map_diag(sns.kdeplot)
g.map_offdiag(sns.kdeplot,n_levels=6)
plt.savefig("2.png")
plt.show()