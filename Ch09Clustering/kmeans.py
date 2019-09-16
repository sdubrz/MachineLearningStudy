# k-means聚类方法
import numpy as np
import random

from sklearn import datasets
import matplotlib.pyplot as plt


class K_means:

    def __init__(self, X, k):
        """
        初始化方法
        :param X: 数据矩阵，每一行是一个样本
        :param k: 聚类的簇数
        """
        self.X = X
        self.k = k
        (n, dim) = X.shape
        self.center_points = np.zeros((k, dim))  # 存放中心点
        self.label = np.zeros((n, 1))  # 记录每个样本属于哪一个类
        self.loop_time = 0  # 迭代次数

    def run(self, X, k):
        # 随机选取k个样本作为初始的样本中心点
        (n, dim) = X.shape
        center = np.zeros((k, dim))
        label = np.zeros((n, 1))
        last_label = np.ones((n, 1))

        center_index = []
        while len(center_index) < k:
            temp_index = random.randint(0, n-1)
            if not(temp_index in center_index):
                center_index.append(temp_index)
                center[len(center_index)-1, :] = X[temp_index, :]

        while not self.no_change(last_label, label):
            last_label = label.copy()
            center_sum = np.zeros((k, dim))
            center_count = np.zeros((k, 1))
            for i in range(0, n):
                distance = []
                for j in range(0, k):
                    d = np.linalg.norm(X[i, :] - center[j, :])
                    distance.append(d)
                i_label = distance.index(min(distance))  # 更新每一个样本的类别
                label[i] = i_label
                center_sum[i_label, :] = center_sum[i_label, :] + X[i, :]
                center_count[i_label] = center_count[i_label] + 1

            for i in range(0, k):
                center[i, :] = center_sum[i, :] / center_count[i]

            self.loop_time += 1

        self.center_points = center
        self.label = label

    def fit_transform(self):
        self.run(self.X, self.k)
        return self.label

    def no_change(self, last_label, label):
        return (last_label == label).all()


def test():
    """
    用具体的数据进行测试
    :return:
    """
    iris = datasets.load_iris()
    X0 = iris.data
    (n, dim) = X0.shape
    X = X0[:, 0:2]
    k = 3
    k_means_test = K_means(X, k)
    y = k_means_test.fit_transform()
    center = k_means_test.center_points

    print('共迭代了 '+str(k_means_test.loop_time) + ' 次')

    colors = ['r', 'g', 'b']
    for i in range(0, n):
        label = int(y[i])
        plt.scatter(X[i, 0], X[i, 1], marker='o', c=colors[label])

    for i in range(0, k):
        plt.scatter(center[i, 0], center[i, 1], marker='+', c=colors[i], s=120)

    plt.show()


if __name__ == '__main__':
    test()
