import numpy as np
import matplotlib.pyplot as plt


def kernel(x, y, kernel_name='line', delta=0.5):
    """
    核函数
    :param x: 第一个样本向量，是行向量
    :param y: 第二个样本向量
    :param kernel_name: 核函数的名字
    :param delta: 高斯核的参数
    :return:
    """
    z = 0
    if kernel_name == 'line':
        z = np.dot(x, y)
    elif kernel_name == 'guass':
        z = np.exp(-np.dot(x-y, x-y)/(2*delta*delta))

    return z


def update_wb(X, label, alpha, kernel_name='line'):
    """
    根据 alpha 的值更新 w 和 b
    :param X: 数据矩阵
    :param label: 数据标签
    :param alpha:
    :param kernel_name:
    :return:
    """
    (n, dim) = X.shape
    w = np.zeros((n, 1))

    for i in range(0, n):
        w = w + alpha[i] * label[i] * X[i, :]

    b = 0
    count = 0
    for i in range(0, n):  # 寻找所有的支持向量
        if alpha[i] > 0:
            b = b + label[i] - np.dot(X[i, :], w)
            count += 1

    if count > 0:
        b = b / count

    return w, b


def svm(X, label, C):
    """
    支持向量机实现
    :param X: 数据矩阵，每一行是一个样本
    :param label: 数据的标签
    :param C: 软间隔的常数
    :return:
    """
    (n, dim) = X.shape

    alpha = np.zeros((n, 1))
    w = np.zeros((1, dim))



