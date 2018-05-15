# -*- coding:utf-8 -*-
# 处理含有车类型的数据集，X实例中第3第4第5个特征实际是车辆类型的向量化。
from numpy import genfromtxt  # genfromtxt函数创建数组表格数据。主要执行两个循环运算。
# 第一个循环将文件的每一行转换成字符串序列；
# 第二个循环将每个字符串序列转换为相应的数据类型。
# genfromtxt能够考虑缺失的数据，但其他更快和更简单的函数像loadtxt不能考虑缺失值。
import numpy as np
from sklearn import datasets,linear_model

import os
base_dir = os.path.dirname(os.path.abspath(__file__))
dataPath = os.path.join(base_dir,'DeliveryDummyDone.csv')

deliveryData = genfromtxt(dataPath,delimiter =',')
print(deliveryData)
# print(type(deliveryData))
# print(deliveryData.shape)

X = deliveryData[:,:-1] # [行,列]
Y = deliveryData[:,-1]
print(X)
print('-'*50)
print(Y)

regr = linear_model.LinearRegression()
regr.fit(X,Y)
print('coefficients:',regr.coef_)    # 相关系数b1,b2
print('intercept:',regr.intercept_)  # 截距b0

# 测试集验证
xPred = np.array([102,6,1,0,0]).reshape(1,5)
print(xPred)
yPred = regr.predict(xPred)
# X : {array-like, sparse matrix}, shape = (n_samples, n_features)
#             Samples.
#
#         Returns

#         -------
#         C : array, shape = (n_samples,)
#             Returns predicted values.
print('predicted y:',yPred)