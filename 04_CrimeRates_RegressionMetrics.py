#!/usr/bin/env python
# -*- coding=utf-8 -*-
__author__ = "WCH"

from sklearn import datasets
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
import myfun_CrimeRates
myfun_CrimeRates.matplot_中文字()

df = pd.read_excel("CrimeRates資料清洗後標準化.xlsx")  # 讀取 pandas資料
print(df.columns)
print(df.shape)
colX=df.columns[0:-1].tolist()                    #'crimes.total'=label
classes = colX

train_x, test_x, train_y, test_y=\
              myfun_CrimeRates.ML_read_excel("CrimeRates資料清洗後標準化.xlsx",colX,['crimes.total'],0.2)


category=2
dim=20


# 建立模型
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(units=100,
                                #activation=tf.nn.tanh,
                                input_dim=dim))
model.add(tf.keras.layers.Dense(units=category))


model.compile(loss='mse',
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
              metrics=['mse', 'mae',
                       #'mape',
                       tf.compat.v1.keras.losses.cosine_proximity])


history=model.fit(train_x, train_y,
       epochs=400,
       batch_size=100)


# testing
cost = model.evaluate(test_x, test_y)
print("準確率 score: ",cost)
weights, biases = model.layers[0].get_weights()
print("權重Weights =" ,weights, "偏移bias = ",biases)

# 印出測試的結果
Y_pred = model.predict(test_x)
print("預測:",Y_pred )
print("實際:",test_y )
#print('MAE:', mean_absolute_error(Y_pred, test_y))
#print('MSE:', mean_squared_error(Y_pred, test_y))


plt.plot(history.history['mse']) #mean_squared_error
plt.plot(history.history['mae']) # mean_absolute_error
# plt.plot(history.history['mape']) # mean_absolute_percentage_error
plt.plot(history.history['cosine_similarity']) # cosine_proximity
plt.title('Regression Metrics')
plt.ylabel('')
plt.xlabel('epoch')
plt.legend(['mean_squared_error','mean_absolute_error',
            #'mean_absolute_percentage_error',
            'cosine_proximity'],

           loc='upper right')
plt.savefig("save.jpg")
plt.show()


