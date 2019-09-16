# 谱图分割
# 算法来自 Ng, Jordan, and Weiss, 2002
import numpy as np

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

    return x


def test():
    # 用数据进行测试
    print('test')


if __name__ == '__main__':
    test()
