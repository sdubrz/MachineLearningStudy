# 线性判别式分析
import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as LA

DATA = np.array([[0.697, 0.774, 0.634, 0.608, 0.556, 0.403, 0.481, 0.437, 0.666, 0.243, 0.245, 0.343, 0.639, 0.657, 0.360, 0.593, 0.719],
                 [0.460, 0.376, 0.264, 0.318, 0.215, 0.237, 0.149, 0.211, 0.091, 0.267, 0.057, 0.099, 0.161, 0.198, 0.370, 0.042, 0.103]])
LABEL = [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]


def max_indexs(a_list0, num_head=2):
    """获得前几个大的数的索引号"""
    k_list = []
    a_list = []
    for i in a_list0:
        a_list.append(i.real)

    n = len(a_list)
    for i in range(0, n):
        if len(k_list) < num_head:
            k_list.append(i)
            index = len(k_list) - 1
            while index > 0:
                if a_list[k_list[index]] > a_list[k_list[index-1]]:
                    temp = k_list[index]
                    k_list[index] = k_list[index-1]
                    k_list[index-1] = temp
                    index = index - 1
                else:
                    break
        else:
            if a_list[k_list[num_head-1]] < a_list[i]:
                k_list[num_head - 1] = i
                index = len(k_list) - 1
                while index > 0:
                    if a_list[k_list[index]] > a_list[k_list[index-1]]:
                        temp = k_list[index]
                        k_list[index] = k_list[index - 1]
                        k_list[index - 1] = temp
                        index = index - 1
                    else:
                        break
    return k_list


def lda(X, label, no_dims=2):
    """
    LDA方法降维算法实现
    :param X: 高维数据矩阵，每一列是一个样本
    :param label: 数据的标签
    :param no_dims: 降维之后的维度数
    :return:
    """
    (dim, n) = X.shape
    mean_X = np.mean(X, axis=1)

    # 对数据按照类别分类
    cluster_list = []
    label_list = []
    for i in range(0, n):
        if not (label[i] in label_list):
            a_list = []
            a_list.append(X[:, i])
            cluster_list.append(a_list)
            label_list.append(label[i])
        else:
            index = label_list.index(label[i])
            a_list = cluster_list[index]
            a_list.append(X[:, i])
    no_labels = label_list.__len__()
    array_list = []
    mean_matrix = np.zeros((dim, no_labels))

    temp_index = 0
    for cluster in cluster_list:
        this_cluster = np.transpose(np.array(cluster))
        array_list.append(this_cluster)
        mean_matrix[:, temp_index] = np.mean(this_cluster, axis=1)
        temp_index = temp_index + 1

    Sb = np.zeros((dim, dim))  # 类间散度
    Sw = np.zeros((dim, dim))  # 类内散度

    print(np.matmul(mean_matrix[:, 0]-mean_X, np.transpose(mean_matrix[:, 0]-mean_X)))

    for i in range(0, no_labels):
        Sb = Sb + len(cluster_list[i]) * np.outer(mean_matrix[:, i]-mean_X, np.transpose(mean_matrix[:, i]-mean_X))
    for i in range(0, n):
        label_index = label_list.index(label[i])
        Sw = Sw + np.outer(X[:, i]-mean_matrix[:, label_index], np.transpose(X[:, i]-mean_matrix[:, label_index]))

    Q = np.dot(LA.inv(Sw), Sb)
    values, vectors = LA.eig(Q)
    selected_index = max_indexs(values, num_head=no_dims)
    P = np.zeros((dim, no_dims))
    for i in range(0, no_dims):
        P[:, i] = vectors[:, selected_index[i]]

    Y = np.dot(np.transpose(P), X)
    return Y, P


def run():
    y, P = lda(DATA, LABEL, no_dims=1)
    print(P)
    plt.scatter(y[0, 0:8], np.zeros((1, 8)), c='c', marker='o', alpha=0.6)
    plt.scatter(y[0, 8:17], np.zeros((1, 9)), c='m', marker='o', alpha=0.6)
    plt.title('LDA')
    plt.show()


if __name__ == '__main__':
    run()
