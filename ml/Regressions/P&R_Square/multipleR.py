import numpy as np

def polyfit(x,y,degree):
    '''
    计算多元线性回归中的r^2的值
    :param x:
    :param y:
    :param degree: 某些方程中有平方，此处均为1
    :return:
    '''
    results = {}
    coeffs = np.polyfit(x,y,degree)  # coeffs 返回的是b0,b1,b2..等相关的系数

    # Polynomial Coefficients
    results['polynomial'] = coeffs.tolist()  # 将其转换为list

    # r-squared
    p = np.poly1d(coeffs)  # 1维的

    # fit values, and mean
    yhat = p(x)
    ybar = np.sum(y)/len(y)