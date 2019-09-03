import numpy as np
import matplotlib.pyplot as plt
import math


DATA = np.array([[0.697, 0.774, 0.634, 0.608, 0.556, 0.403, 0.481, 0.437, 0.666, 0.243, 0.245, 0.343, 0.639, 0.657, 0.360, 0.593, 0.719],
                 [0.460, 0.376, 0.264, 0.318, 0.215, 0.237, 0.149, 0.211, 0.091, 0.267, 0.057, 0.099, 0.161, 0.198, 0.370, 0.042, 0.103]])
LABEL = [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]


def sigmod(z):
    y = 1 / (1+math.exp(z))
    return y


def cost_function(X, label, beta):
    """目标函数值"""
    (m, n) = X.shape
    num = 0
    for i in range(0, n):
        z = np.dot(beta, X[:, i])
        num = num - label[i]*z + np.log(1+np.exp(z))
    return num


def logist_regress(X, label):
    """
    开始用牛顿法求解遇到了一些情况，改成了普通的梯度方法
    :param X: 目标数据集，每一列是一个样本
    :param label: 数据的标签
    :return:
    """
    (m, n) = X.shape
    beta = np.ones((1, m))
    last_value = 0

    # 用牛顿法求解
    # while True:
    #     current_value = cost_function(X, label, beta)
    #     if np.abs(current_value - last_value) < 0.001:
    #         break
    #
    #     last_value = current_value
    #     first_d = np.zeros((1, m))
    #     second_d = np.zeros((m, m))
    #
    #     for i in range(0, n):
    #         z = np.dot(beta, X[:, i])
    #         p1 = np.exp(z) / (1+np.exp(z))
    #         first_d = first_d - X[:, i] * (label[i] - p1)
    #         second_d = second_d + np.outer(X[:, i], np.transpose(X[:, i]))*p1*(1-p1)
    #
    #     print('一阶导是：', first_d)
    #     print('二阶导是，', second_d)
    #     beta = beta - np.dot(np.linalg.inv(second_d), np.transpose(first_d))

    # 用普通的梯度方法求解
    last_value = cost_function(X, label, beta)
    alpha = 0.1
    error_list = []
    index = 0
    while True:
        index = index + 1
        if index > 3000:
            break
        first_d = np.zeros((1, m))
        for i in range(0, n):
            z = np.dot(beta, X[:, i])
            p1 = np.exp(z) / (1+np.exp(z))
            first_d = first_d - X[:, i] * (label[i] - p1)
        beta = beta - alpha * first_d
        current_value = cost_function(X, label, beta)
        error_list.append(current_value)
        if np.abs(last_value - current_value) < 0.01:
            break

    return beta
    # print(beta)
    # print('求解完毕')
    # print(index)
    # plt.plot(error_list)
    # plt.show()


def run():
    (m, n) = DATA.shape
    X = np.ones((m+1, n))
    X[0:m, :] = DATA[:, :]
    beta = logist_regress(X, LABEL)
    print('beta = ', beta)

    for i in range(0, n):
        color = 'c'
        if LABEL[i] == 1:
            color = 'm'
        plt.scatter(DATA[0, i], DATA[1, i], c=color, marker='o')
    x = np.linspace(0, 1, 10)
    y = -beta[0, 0]/beta[0, 1]*x - beta[0, 2]/beta[0, 1]
    plt.plot(x, y)
    plt.title('Logistic Regression')
    plt.show()


if __name__ == '__main__':
    run()
