# 不进行正则化的谱聚类
import numpy as np
import matplotlib.pyplot as plt


class SpectralCluster:
    def __init__(self, X, k):
        """

        :param X: 数据矩阵，每一行是一个样本
        :param k: 要聚的类数
        """
        self.X = X
        self.k = k
        (n, dim) = X.shape
        self.y = np.zeros((n, 1))

    def gauss_similar(self, X, delta=1.0):
        """
        计算高斯的相似度
        :param X: 数据矩阵，每一行是一个样本
        :param delta: 高斯分布的方差
        :return:
        """
        (n, dim) = X.shape
        W = np.zeros((n, n))

        for i in range(0, n):
            for j in range(0, n):
                d = np.linalg.norm(X[i, :] - X[j, :])
                W[i, j] = np.exp(-d * d / (2 * delta * delta))
        return W

    def laplacian_matrix(self, X, delta=1.0):
        """
        计算拉普拉斯矩阵
        :param X: 数据矩阵，每一行是一个样本
        :param delta: 高斯分布的方差
        :return:
        """
        (n, dim) = X.shape
        W = self.gauss_similar(X, delta=delta)
        D = np.zeros((n, n))
        for i in range(0, n):
            D[i, i] = np.sum(W[i, :])

        L = D - W
        return L

    def fit_transform(self, X, k, delta=1.0):
        """
        执行谱聚类
        :param X:
        :param k:
        :param delta:
        :return:
        """
        (n, dim) = X.shape
        L = self.laplacian_matrix(X, delta=delta)


