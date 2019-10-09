# 用于部分数据的预处理
import numpy as np
import os


def standardize_array(array):
    standardized_array = (array - np.mean(array, axis=0)) / np.std(array, axis=0)
    return np.nan_to_num(standardized_array)


def normalize(X, low=-1.0, up=1.0):
    """
    将数据的每一个维度都线性映射到 [low, up] 之间
    :param X: 数据矩阵，每一行是一个样本
    :param low: 下界
    :param up: 上界
    :return:
    """
    (n, dim) = X.shape
    Y = np.zeros((n, dim))
    for j in range(0, dim):
        j_max = np.max(X[:, j])
        j_min = np.min(X[:, j])

        if j_max == j_min:
            Y[:, j] = 0
        else:
            Y[:, j] = (X[:, j] - j_min)/(j_max-j_min)*(up-low) + low
            # for i in range(0, n):
            #     Y[i, j] = (Y[i, j] - j_min) / (j_max - j_min) * (up - low) + low

    return Y


def check_filepath(path):
    """
    通过向一个路径下存放一个很小的文件来检查该文件夹是否存在
    避免出现存储结果时报错，以至于浪费时间的情况
    :param path: 要检查的文件路径
    :return:
    """
    if not os.path.exists(path):
        os.makedirs(path)
    data = np.array([[1, 2, 3],
                     [4, 5, 6],
                     [7, 8, 9]])
    np.savetxt(path + "check_path.csv", data, fmt="%d", delimiter=",")
