# 谱图分割
# 算法来自 Ng, Jordan, and Weiss, 2002
import numpy as np
from Ch09Clustering import kmeans
from sklearn.neighbors import NearestNeighbors

import matplotlib.pyplot as plt
from sklearn import datasets


def spectral_graph_part(W, k):
    """

    :param W: 相似度矩阵
    :param k: 聚类的簇数
    :return:
    """
    (n, m) = W.shape
    if n != m:
        print('相似度矩阵必须是方阵')
        return

    Z = np.zeros((n, n))
    delta = np.zeros((n, 1))
    for i in range(0, n):
        delta[i] = np.sum(W[i, :])

    for i in range(0, n):
        for j in range(0, n):
            Z[i, j] = W[i, j] / np.sqrt(delta[i]*delta[j])

    values, vectors = np.linalg.eig(Z)
    eig_idx = np.argpartition(values, k)[-k:]

    eig_idx = eig_idx[np.argsort(-values[eig_idx])]
    x = vectors[:, eig_idx]

    k_means = kmeans.K_means(x, k)
    label = k_means.fit_transform()

    return label


def connect_component_cov(X, r, epsilon, eta):
    """
    Algorithm 2 in 'Spectral Clustering Based on Local PCA'
    :param X: 数据矩阵，每一行是一个样本
    :param r: 邻域半径
    :param epsilon: 谱范围，要求大于0
    :param eta: 协方差范围，要求大于0
    :return:
    """
    (n, dim) = X.shape

    # 计算R近邻
    nbr_object = NearestNeighbors(radius=r).fit(X)
    distance, indexs = nbr_object.radius_neighbors(X)


def test():
    # 用数据进行测试
    print('test')


if __name__ == '__main__':
    test()
