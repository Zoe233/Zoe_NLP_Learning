# -*- coding:utf-8 -*-
import numpy as np
import random

# 麦子学院的 7.6的梯度下降介绍的太简单了，没彻底明白。


def gradientDescent(X,Y,theta,alpha,m,numIterations):
    '''
    自定义 梯度下降法计算方法
    :param X: 实例集X
    :param Y: 因变量（目标变量）
    :param theta: θ向量，里面包含了θ0，θ1，θ2，...，θn （ n = len(X) )
    :param alpha: 是梯度下降中 更新法则的 α
    :param m: m 是实例的数量,len(X)
    :param numIterations: 指定的重复更新θ的次数
    :return:
    '''
    # ！！！重要！！！
    XTrans = X.transpose()
    # 老师讲的是将θ的向量转置为[θ0,θ1,θ2,θ3,...,θn]
    # 但实际上是把X中的示例给 转置 了，为啥咧？
    # 因为 theta = np.ones(n) 生成的是n*1的一维向量，是行向量。已经满足要求。
    # 而X中的一个实例为1个行向量，我们在做向量之间的内积的时候需要将X的单条实例变为 列向量。
    print('XTrans:')
    print(XTrans)

    for i in range(0,numIterations):
        hypothesis = np.dot(X,theta)  # 做向量内积
        # print('hypothesis:',hypothesis)
        loss = hypothesis - Y  # 预测值 - 实际Y值
        # print('loss:',loss)

        # avg cost per example ( the 2 in 2*m doesn't really matter here.
        # but to be consistent with the gradient, I include it.
        cost = np.sum(pow(loss,2)/(2*m))  # 此处的cost函数的定义和 前面介绍的公式不完全一致，这边是通用型的。
        print('Iteration %d | Cost:%f'%(i,cost))

        # avg gradient per example
        gradient = np.dot(XTrans,loss)/m

        # update 更新法则
        theta = theta - alpha*gradient
    return theta






def genData(numPoints,bias,variance):
    '''
    生成数据集方法
    :param numPoints: 数据集行数（即数据集实例数）
    :param bias:目标变量y的偏差（实际的估计值和y之间的偏差值）
    :param variance:方差，用于数据的离散程度
    :return: X,Y
    '''
    X = np.zeros( shape = (numPoints,2))  # 生成numPoints*2的0数组
    Y = np.zeros( shape = numPoints)  # 生成numPonts的0向量
    # print(x)
    # print(y)
    for i in range(numPoints):
        # 将X数组中的每行的第0个元素变为1，每行的第1个元素变为i值
        X[i][0] = 1
        X[i][1] = i

        # 将Y向量中的结果生成i+bias+呈正态分布的[0,1]之间的随机数*variance
        # Y[i] = (i+bias) + random.uniform(0,1)* variance
        # 根据上面的bias和variance的定义，老师的例子并不严谨。向量的方差是固定的，但是每个y_hat和y的偏差值是不同的
        # 所以将 Y[i]生成数改成这样：
        Y[i] = (i + variance) + random.uniform(0, 1) * bias

    print(X.shape)
    print(Y.shape)
    return X,Y

if __name__=='__main__':
    # 生成数据集 X,Y
    X,Y = genData(10,25,10)

    print('*'*50)
    m,n = np.shape(X)
    numIterations = 50000
    alpha = 0.0005
    theta = np.ones(n)
    # 梯度下降的思想：
    # theta从某个组开始，可以是0，
    # 保持该（组）值持续减小，如果是一组值就要保证他们同步更新，直到找到我们希望找到的最小值
    # 我们要找到一条最快下山的路径，我们走的每一步大小就是α 。
    theta = gradientDescent(X,Y,theta,alpha,m,numIterations)

    print(theta)