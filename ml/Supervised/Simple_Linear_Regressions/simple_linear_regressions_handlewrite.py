# -*- coding:utf-8 -*-
import numpy as np
X = [1, 3, 2, 1, 3]
Y = [14, 24, 18, 17, 27]

def fitSLR(X,Y):
    '''
    定义简单线性回归方程训练模型
    入参：X,Y
    出参：b0,b1
    '''
    n = len(X)
    numerator = 0 # 分子
    denominator =0 # 分母

    for i in range(n):
        numerator +=(X[i]-np.mean(X))*(Y[i]-np.mean(Y))
        denominator += pow(X[i]-np.mean(X),2)

    print('numerator:',numerator)
    print('denominator:',denominator)

    b1 = numerator/denominator
    b0 = np.mean(Y)-b1*np.mean(X)
    print('b0:',b0,'b1:',b1)
    return  b0,b1

def predict(x,b0,b1):
    '''
    根据训练的模型，将x值代入线性回归估计方程，计算y值
    :param x:测试数据
    :return:y-head
    '''
    y = b0+b1*x
    print(y)
    return y

if __name__=='__main__':
    fitSLR(X,Y)
    predict(5,10,5)