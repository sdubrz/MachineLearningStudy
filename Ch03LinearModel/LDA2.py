import numpy as np
import matplotlib.pyplot as plt


DATA = np.array([[0.697, 0.774, 0.634, 0.608, 0.556, 0.403, 0.481, 0.437, 0.666, 0.243, 0.245, 0.343, 0.639, 0.657, 0.360, 0.593, 0.719],
                 [0.460, 0.376, 0.264, 0.318, 0.215, 0.237, 0.149, 0.211, 0.091, 0.267, 0.057, 0.099, 0.161, 0.198, 0.370, 0.042, 0.103]])
LABEL = [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]


def lda(X1, X2):
    """
    二分类的LDA
    :param X1: 第一部分数据，每一列是一个样本
    :param X2:
    :return:
    """
    (m1, n1) = X1.shape
    (m2, n2) = X2.shape

    x1_mean = np.mean(X1, axis=1)
    x2_mean = np.mean(X2, axis=1)

    # Sb = np.outer(x1_mean, np.transpose(x2_mean))  # 类间散度
    Sw = np.zeros((m1, m1))  # 类内散度

    for i in range(0, n1):
        Sw = Sw + np.outer(X1[:, i], np.transpose(x1_mean))
    for i in range(0, n2):
        Sw = Sw + np.outer(X2[:, i], np.transpose(x2_mean))

    w = np.dot(np.linalg.inv(Sw), x1_mean-x2_mean)

    return w


def run():
    X1 = DATA[:, 0:8]
    X2 = DATA[:, 8:17]
    w = lda(X1, X2)
    print('w = ', w)

    y1 = np.dot(w, X1)
    y2 = np.dot(w, X2)

    ax1 = plt.subplot(121)
    ax1.scatter(X1[0, :], X1[1, :], c='c', marker='o', label='good')
    ax1.scatter(X2[0, :], X2[1, :], c='m', marker='o', label='bad')
    temp_x = np.linspace(0, 1, 10)
    temp_y = temp_x * (-w[0]/w[1])
    ax1.plot(temp_x, temp_y)
    ax1.legend()
    ax2 = plt.subplot(122)
    ax2.scatter(y1, np.zeros((1, 8)), c='c', marker='o', alpha=0.6, label='good')
    ax2.scatter(y2, np.zeros((1, 9)), c='m', marker='o', alpha=0.6, label='bad')
    ax2.legend()
    # plt.scatter(X1[0, :], X1[1, :], c='c', marker='o')
    # plt.scatter(X2[0, :], X2[1, :], c='m', marker='o')
    plt.show()


if __name__ == '__main__':
    run()
