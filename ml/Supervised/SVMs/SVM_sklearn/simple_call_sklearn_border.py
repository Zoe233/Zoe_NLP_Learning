# 用sklearn画出决定边界

import pylab as pl  # pylab 画图的包
from sklearn import svm
import numpy as np

# 创建实例X
# np.random.randn(a,b)生成a*b的矩阵，从标准正态分布中返回一个或多个样本值。
# np.randn.rand(a,b)生成a*b的矩阵， 从[0, 1)中返回随机值。
np.random.seed(1)
X = np.r_[np.random.randn(20,2)-[2,2],np.random.randn(20,2)+[2,2],]  # 生成40*2的数组
# 生成与X对应的 目标对象向量  40*1的数组

Y = np.array([0]*20 + [1]*20).reshape(40,1)
Y = np.array([0]*20 + [1]*20)  # svm要求的格式为：array-like, shape (n_samples,)

# print(X)
# print(Y)

# 调用sklearn svm 模型
clf = svm.SVC(kernel ='linear')
clf.fit(X,Y)

# 获取超平面 separating hyperplane
w = clf.coef_[0]
a = -w[0]/w[1]
xx = np.linspace(-5,5)
yy = a*xx-(clf.intercept_[0])/w[1]

# 画图
b = clf.support_vectors_[0] # 获取第一个 支持向量
yy_down = a*xx + (b[1] - a*b[0])

b = clf.support_vectors_[-1] # 获取最后一个 支持向量
yy_up = a*xx + (b[1] -a*b[0])

print("w:",w)
print("a:",a)
# print('xx:',xx)
# print('yy:',yy)
print('support_vectors_:',clf.support_vectors_)
print('clf.coef_:',clf.coef_)
# 在sklearn中，coef_属性存储着线性模型的超平面的向量。
# It has shape (n_classes, n_features) if n_classes > 1 (multi-class one-vs-all) and (1, n_features) for binary classification.
# 在此例中，n_fetures为2，因此w = coef_[0]是向量正交于超平面


