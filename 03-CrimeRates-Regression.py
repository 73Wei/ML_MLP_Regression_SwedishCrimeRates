#!/usr/bin/env python
# -*- coding=utf-8 -*-
__author__ = "WCH"

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets, linear_model
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import myfun_CrimeRates
myfun_CrimeRates.matplot_中文字()


df = pd.read_excel("CrimeRates資料清洗後標準化.xlsx")  # 讀取 pandas資料
print(df.columns)
print(df.shape)

colX=df.columns[0:-1].tolist()     #'crimes.total'=label
print(colX)

train_x, test_x, train_y, test_y=\
              myfun_CrimeRates.ML_read_excel("CrimeRates資料清洗後標準化.xlsx",colX,['crimes.total'],0.2)

model = linear_model.LinearRegression()
model.fit(train_x, train_y)
predict_y=model.predict(test_x)
print("預測的y:",predict_y)
print("實際的y:",test_y)


print('score: %.2f' % model.score(test_x, test_y))
print('Coefficients: \n', model.coef_)
print("權重：",model.coef_)
print("截距：",model.intercept_)
print("MAE:", mean_absolute_error(test_y, model.predict(test_x)))   # 實際的Ｙ和算出來的Ｙ
print("MSE:", mean_squared_error(test_y, model.predict(test_x)))


print("model 算出來的公式為 y=",str(model.intercept_),"+")
for i in range(len(model.coef_)):
    print("(特徵值",i," * ",model.coef_[i],")+")



# ### 單一圖
for p in range(len(model.coef_)):
    plt.scatter(test_x[:,p], test_y,label=" 預測的結果")
    plt.scatter(train_x[:,p], train_y,label=" 訓練資料")
    plt.title("特徵值"+str(p))
    plt.legend()
    plt.savefig("LinearRegression 特徵值" + str(p))
    plt.show()



myfun_CrimeRates.ML_Regression_PolynomialFeaturesLinearRegression(train_x, test_x, train_y, test_y,0)
myfun_CrimeRates.ML_Regression_SVR(train_x, test_x, train_y, test_y,0)
myfun_CrimeRates.ML_Regression_BayesianRidgePolynomialRegression(train_x, test_x, train_y, test_y,0)


############### ML PCA
from sklearn.decomposition import PCA
model = PCA(n_components=4)      # <----- 組件的數量。 如果為 None，則保留所有非零分量。)must be between 0 and min(n_samples, n_features)=4

test_x_pca = model.fit(test_x).transform(test_x)
# 沒有predict,print("model 預測:", model.predict([[9,9],[9.2,9.2]]))

print(test_x_pca[:, 0])
print(test_x_pca[:, 1])
print(test_x_pca[:, 2])   # 因為降維，所以3維 變成 2維

######
from sklearn.decomposition import KernelPCA
kernel_pca = KernelPCA(
    n_components=2,      # <----- 組件的數量。 如果為 None，則保留所有非零分量。
    kernel="rbf", gamma=10, fit_inverse_transform=True, alpha=0.1
)

test_x_kernel_pca = kernel_pca.fit(train_x).transform(train_x)
# 沒有predict,  print("kernel_pca 預測:", kernel_pca.predict([[9,9],[9.2,9.2]]))


# 畫 3D圖片
ax = plt.axes(projection='3d')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

# test_x[:,0], test_x[:,1], test_x[:,2],c=test_y
ax.scatter(train_x[:,0], train_x[:,1], train_x[:,2],c=train_y,label='訓練資料')
#ax.plot_surface(X, Y, Z, cmap = cm.Reds,label="預測後的結果")
plt.show()


#  畫圖
fig, (orig_data_ax, pca_proj_ax, kernel_pca_proj_ax) = plt.subplots(
    ncols=3, figsize=(14, 4)
)
orig_data_ax.scatter(test_x[:, 0], test_x[:, 1], c=test_y)
orig_data_ax.set_ylabel("Feature #1")
orig_data_ax.set_xlabel("Feature #0")
orig_data_ax.set_title("Testing data")
plt.savefig("1.png")

pca_proj_ax.scatter(test_x_pca[:, 0], test_x_pca[:, 1], c=test_y)
pca_proj_ax.set_ylabel("Principal component #1")
pca_proj_ax.set_xlabel("Principal component #0")
pca_proj_ax.set_title("Projection of testing data\n using PCA")
plt.savefig("2.png")

kernel_pca_proj_ax.scatter(test_x_kernel_pca[:,0], test_x_kernel_pca[:, 1], c=test_y)
kernel_pca_proj_ax.set_ylabel("Principal component #1")
kernel_pca_proj_ax.set_xlabel("Principal component #0")
kernel_pca_proj_ax.set_title("Projection of testing data\n using KernelPCA")

plt.savefig("3.png")
plt.show()

