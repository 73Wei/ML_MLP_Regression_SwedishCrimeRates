#!/usr/bin/env python
# -*- coding=utf-8 -*-
__author__ = "WCH"
import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
from tensorflow.keras.datasets import boston_housing
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn import preprocessing
import myfun_CrimeRates
myfun_CrimeRates.matplot_中文字()

df = pd.read_excel("CrimeRates資料清洗後標準化.xlsx")  # 讀取 pandas資料
print(df.columns)
print(df.shape)
colX=df.columns[0:-1].tolist()                    #'crimes.total'=label
classes = colX

train_x, test_x, train_y, test_y=\
              myfun_CrimeRates.ML_read_excel("CrimeRates資料清洗後標準化.xlsx",colX,['crimes.total'],0.2)



model =  tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(1000,
                                activation='tanh',
                                input_shape=[train_x.shape[1]]))
model.add(tf.keras.layers.Dense(1000, activation='tanh'))
model.add(tf.keras.layers.Dense(1))


model.compile(loss='mse',
              optimizer='sgd',
              metrics=['mae'])

history=model.fit(train_x, train_y,
          epochs=500,
          batch_size=100)

#保存模型架構
with open("model_Boston.json", "w") as json_file:
   json_file.write(model.to_json())
#保存模型權重
model.save_weights("model_Boston.h5")


# testing
print("start testing")
cost = model.evaluate(test_x, test_y)
print("test cost: {}".format(cost))

Y_pred2 = model.predict(test_x)  # Y predict

print(Y_pred2[:10])
print(test_y[:10])
# 印出測試的結果
Y_pred = model.predict(test_x)
print("預測:",Y_pred )
print("實際:",test_y)
print('MAE:', mean_absolute_error(Y_pred, test_y))
print('MSE:', mean_squared_error(Y_pred, test_y))

################################################
#print(history.history.keys())
plt.plot(history.history['mae'])  # mean_absolute_error
plt.title('CrimeRate')
plt.ylabel('mae')
plt.xlabel('epoch')
plt.legend(['train mae'], loc='upper right')
plt.savefig('CrimeRate')
plt.show()



