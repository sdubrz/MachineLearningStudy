# 学习向量量化(Learning Vector Quantization)聚类方法
import numpy as np
import random
import matplotlib.pyplot as plt


class LVQ:
    def __init__(self, X, y, k, eta=0.05, max_inter=3000):
        """
        学习向量量化聚类方法
        :param X: 训练数据集，每一行是一个样本
        :param y: 训练数据集的类别，要求从0开始
        :param k: 聚类的簇数
        :param eta: 学习率
        :param max_inter: 最大迭代次数
        """
        self.X = X
        self.k = k
        self.y = y
        self.eta = eta
        self.max_inter = max_inter
        self.label = y.copy()
        (n, dim) = X.shape
        self.center_point = np.zeros((k, dim))  # 原型向量

    def run(self, X, y, k, eta=0.05):
        (n, dim) = X.shape

        # 随机选取一组初始的原型向量
        center = np.zeros((k, dim))
        center_index = []
        while len(center_index) < k:
            temp_index = random.randint(0, n-1)
            if not (temp_index in center_index):
                center_index.append(temp_index)
                center[len(center_index)-1, :] = X[temp_index, :]

        # 开始迭代
        for i in range(0, self.max_inter):
            current_index = random.randint(0, n-1)
            distance = []
            for j in range(0, k):
                d = np.linalg.norm(X[current_index, :] - center[j, :])
                distance.append(d)

            p_index = distance.index(min(distance))
            if p_index == y[current_index]:
                center[p_index] = center[p_index, :] + eta * (X[current_index, :] - center[p_index, :])
            else:
                center[p_index] = center[p_index, :] - eta * (X[current_index, :] - center[p_index, :])

        self.center_point = center

    def fit(self):
        self.run(self.X, self.y, self.k, eta=self.eta)
        return self.center_point
