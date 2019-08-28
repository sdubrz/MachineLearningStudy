# 3.3编程实现对率回归，并给出西瓜数据集3.0α上的结果


import numpy as np
import matplotlib.pyplot as plt


Data = np.array([[0.697, 0.774, 0.634, 0.608, 0.556, 0.403, 0.481, 0.437, 0.666, 0.243, 0.245, 0.343, 0.639, 0.657, 0.360, 0.593, 0.719],
                 [0.460, 0.376, 0.264, 0.318, 0.215, 0.237, 0.149, 0.211, 0.091, 0.267, 0.057, 0.099, 0.161, 0.198, 0.370, 0.042, 0.103]])
Label = [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
(m, n) = Data.shape
X = np.concatenate((Data, np.ones((1, n))), axis=0)
beta = np.random.random((m+1, 1))
thresold = 0.0001
value_list = []  # 每次迭代的目标函数值


def l_value(beta):
    """计算目标函数值"""
    num = 0
    for i in range(0, n):
        temp = np.dot(np.transpose(beta), X[:, i])
        num = num - Label[i]*temp + np.log(1+np.exp(temp))
    return num[0]


def logistic_regression(beta):
    last_value = l_value(beta)
    value_list.append(last_value)

    while True:
        first_d = np.zeros((m+1, 1))
        second_d = np.zeros((m+1, m+1))

        for i in range(0, n):
            p1 = np.exp(np.dot(np.transpose(beta), X[:, i])) / (1+np.exp(np.dot(np.transpose(beta), X[:, i])))
            first_d = first_d - X[:, i]*(Label[i]-p1)
            second_d = second_d + np.outer(X[:, i], np.transpose(X[:, i]))*p1*(1-p1)

        beta = beta - np.dot(np.linalg.inv(second_d), first_d)
        this_value = l_value(beta)
        value_list.append(this_value)

        if last_value - this_value < thresold:
            break

        last_value = this_value

    test_label = []
    for i in range(0, n):
        temp = np.dot(np.transpose(beta), X[:, i])
        if temp[0] > 0.5:
            test_label.append(1)
        else:
            test_label.append(0)
    print(Label)
    print(test_label)

    for i in range(0, n):
        color = 'c'
        shape = 'o'
        if Label[i] == 1:
            color = 'm'

    plt.scatter(X[0, 0:8], X[1, 0:8], c='c', marker='o')
    plt.scatter(X[0, 8:n], X[1, 8:n], c='m', marker='o')

    plt.show()
    plt.plot(value_list)
    plt.show()

    print(value_list)


def test():
    # a = np.array([[1, 2, 3]])
    # b = np.array([[1],
    #               [2],
    #               [3]])
    # c = np.dot(b, a)
    # print(c)
    # print(np.log(2.7))
    print(X[:, 1])
    a = np.multiply(np.transpose(X[:, 1]), X[:, 1])
    print(a)
    print(X[:, 1]@np.transpose(X[:, 1]))
    print(np.outer(X[:, 1], np.transpose(X[:, 1])))


if __name__ == '__main__':
    logistic_regression(beta)
    # test()
