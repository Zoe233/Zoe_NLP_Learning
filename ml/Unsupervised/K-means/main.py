# -*- coding:utf-8 -*-
# 参考来源: https://blog.csdn.net/zzz_cming/article/details/79859490
# k-means聚类程序展示
import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets as ds
import matplotlib.colors
from sklearn.cluster import KMeans

# 随机生成数据
points_number = 3000
centers = 20
data, laber = ds.make_blobs(points_number, centers = centers, random_state = 0)
# data.shape (3000,2)
# laber.shape (3000,)

# 创建Figure
fig = plt.figure()
# 用来正常显示中文标签
matplotlib.rcParams['font.sans-serif'] = [u'SimHei']
# 用来正常显示负号
matplotlib.rcParams['axes.unicode_minus'] = False

# 原始点的分布
ax1 = fig.add_subplot(211)
plt.scatter(data[:,0],data[:,1],c=laber)
plt.title(u'原始数据分布')
plt.sca(ax1)

# K-means聚类后
N = 5
model = KMeans(n_clusters=N, init='k-means++')
y_pre = model.fit_predict(data)
ax2 = fig.add_subplot(212)
plt.scatter(data[:,0],data[:,1],c=y_pre)
plt.title(u'K-Means聚类')
plt.sca(ax2)

plt.show()