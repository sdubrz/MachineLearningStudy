# 不进行正则化的谱聚类
import numpy as np
import matplotlib.pyplot as plt

from Ch09Clustering import kmeans
from sklearn import datasets
from Tools import PreProcess
from Ch10DimensionReduction import PCA
from sklearn.cluster import spectral_clustering
from sklearn.cluster import SpectralClustering


class SpectralCluster:
    def __init__(self, X, k, delta=1.0):
        """

        :param X: 数据矩阵，每一行是一个样本
        :param k: 要聚的类数
        """
        self.X = X
        self.k = k
        (n, dim) = X.shape
        self.delta = delta
        self.label = np.zeros((n, 1))

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
            W[i, i] = 0
        np.savetxt('E:\\Project\\MachineLearning\\spectralClustering\\W.csv', W, fmt='%f', delimiter=',')
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
        np.savetxt('E:\\Project\\MachineLearning\\spectralClustering\\L.csv', L, fmt='%f', delimiter=',')
        return L

    def clustering(self, X, k, delta=1.0):
        """
        聚类
        :param X: 数据矩阵，每一行是一个样本
        :param k: 簇的数目
        :param delta: 计算高斯相似度时用的方差，从t-SNE的经验来看，所有的数据都共用一个方差其实是很不合适的
        :return:
        """
        (n, dim) = X.shape
        L = self.laplacian_matrix(X, delta=delta)
        eg_values, eg_vectors = np.linalg.eig(L)
        idx = eg_values.argsort()
        eg_vectors = eg_vectors[:, idx]

        print(eg_values)

        U = eg_vectors[:, 0:k]
        # U = eg_vectors[:, dim-k:dim]  # 一个实验，是不对的
        np.savetxt('E:\\Project\\MachineLearning\\spectralClustering\\U.csv', U, fmt='%f', delimiter=',')
        k_means = kmeans.K_means(U, k)
        label = k_means.fit_transform()

        return label

    def fit_transform(self):
        self.label = self.clustering(self.X, self.k, self.delta)
        return self.label


def test():
    """用实际数据进行测试"""
    # wine = datasets.load_iris()
    # X = wine.data
    # label_true = wine.target

    read_path = 'E:\\Project\\MachineLearning\\datasets\\Wine\\'
    data_reader = np.loadtxt(read_path+'data.csv', dtype=np.str, delimiter=',')
    label_reader = np.loadtxt(read_path+'label.csv', dtype=np.str, delimiter=',')
    X = data_reader[:, :].astype(np.float)
    label_true = label_reader.astype(np.int)

    X = PreProcess.normalize(X)
    (n, dim) = X.shape
    k = 3
    delta = 0.5

    spectral_cluster = SpectralCluster(X, k, delta=delta)
    label = spectral_cluster.fit_transform()
    pca = PCA.PCA(X, 2)
    Y = pca.fit_transform()

    colors = ['c', 'm', 'y', 'b', 'r', 'g']
    shapes = ['s', 'o', '^', 'p', '+', '*']
    for i in range(0, n):
        plt.scatter(Y[i, 0], Y[i, 1], c=colors[int(label[i])], marker=shapes[int(label_true[i])])

    plt.show()


def sklearn_test():
    """
    用sklearn中的谱聚类方法实验
    :return:
    """
    # wine = datasets.load_iris()
    # X = wine.data
    # label_true = wine.target

    read_path = 'F:\\result2019-2\\result0812\\datasets\\digits5_8\\'
    data_reader = np.loadtxt(read_path + 'data.csv', dtype=np.str, delimiter=',')
    label_reader = np.loadtxt(read_path + 'label.csv', dtype=np.str, delimiter=',')
    X = data_reader[:, :].astype(np.float)
    label_true = label_reader.astype(np.int)

    X = PreProcess.normalize(X)
    (n, dim) = X.shape
    k = 5
    delta = 1.0

    affinity = SpectralCluster.gauss_similar(None, X=X, delta=delta)
    label = spectral_clustering(affinity, n_clusters=k)

    pca = PCA.PCA(X, 2)
    Y = pca.fit_transform()

    colors = ['c', 'm', 'y', 'b', 'r', 'g']
    shapes = ['s', 'o', '^', 'p', '+', '*']
    for i in range(0, n):
        plt.scatter(Y[i, 0], Y[i, 1], c=colors[int(label[i])], marker=shapes[int(label_true[i])])

    plt.show()


def sklearn_test2():
    read_path = 'F:\\result2019-2\\result0812\\datasets\\Wine\\'
    data_reader = np.loadtxt(read_path + 'data.csv', dtype=np.str, delimiter=',')
    label_reader = np.loadtxt(read_path + 'label.csv', dtype=np.str, delimiter=',')
    X = data_reader[:, :].astype(np.float)
    label_true = label_reader.astype(np.int)

    X = PreProcess.normalize(X)
    (n, dim) = X.shape
    k = 3
    delta = 1.0

    sc = SpectralClustering(n_clusters=k)
    sc.fit(X)
    label = sc.labels_

    pca = PCA.PCA(X, 2)
    Y = pca.fit_transform()

    colors = ['c', 'm', 'y', 'b', 'r', 'g']
    shapes = ['s', 'o', '^', 'p', '+', '*']
    for i in range(0, n):
        plt.scatter(Y[i, 0], Y[i, 1], c=colors[int(label[i])], marker=shapes[int(label_true[i])])

    plt.show()


if __name__ == '__main__':
    test()
    # sklearn_test2()





