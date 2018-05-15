# 计算简单线性回归中的相关性和R平方值应用
# 手写，重复计算过程
# 公式见图 "r计算公式（线性回归).png"
import numpy as np
import math

def computeCorrelation(X,Y):
    '''
    计算回归中的相关性
    :param X: 实例
    :param Y: 目标值向量
    :return:
    '''
    xBar = np.mean(X)  # X的平均值
    yBar = np.mean(Y)  # Y的平均值
    fenzi = 0
    varX = 0
    varY = 0
    for i in range(0,len(X)):
        diffXXBar = X[i] - xBar
        diffYYBar = Y[i] - yBar
        fenzi += (diffXXBar*diffYYBar)
        varX += diffXXBar**2
        varY += diffYYBar**2

    fenmu = math.sqrt(varX * varY)
    return fenzi/fenmu

if __name__ == '__main__':
    testX = [1,3,8,7,9]
    testY = [10,12,24,21,34]

    r = computeCorrelation(testX,testY)
    r_square =  pow(r,2)
    print(r)
    print(r_square)