# PCA降维方法
import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from Tools import PreProcess


class PCA:
    def __init__(self, X, n_dims):
        """

        :param X: 要进行降维的数据矩阵，每一行是一个样本
        :param n_dims: 降维之后的维度数
        """
        self.X = X
        self.n_dims = n_dims
        self.Y = None  # 降维结果

    def fit_transform(self):
        (n, dim) = self.X.shape
        X2 = self.X - np.mean(self.X, axis=0)
        C = np.dot(X2.T, X2)

        eg_values, eg_vectors = np.linalg.eig(C)
        idx = (eg_values*-1).argsort()

        eg_vectors = eg_vectors[:, idx]
        P = eg_vectors[:, 0:self.n_dims]

        self.Y = np.matmul(X2, P)
        return self.Y


def test():
    data = datasets.load_wine()
    X = data.data
    X = PreProcess.normalize(X)
    label = data.target

    pca = PCA(X, 2)
    Y = pca.fit_transform()
    plt.scatter(Y[:, 0], Y[:, 1], c=label)

    plt.show()


if __name__ == '__main__':
    test()
