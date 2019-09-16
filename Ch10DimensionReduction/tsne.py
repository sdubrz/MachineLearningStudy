#
#  tsne.py
#
# Implementation of t-SNE in Python. The implementation was tested on Python
# 2.7.10, and it requires a working installation of NumPy. The implementation
# comes with an example on the MNIST dataset. In order to plot the
# results of this example, a working installation of matplotlib is required.
#
# The example can be run by executing: `ipython tsne.py`
#
#
#  Created by Laurens van der Maaten on 20-12-08.
#  Copyright (c) 2008 Tilburg University. All rights reserved.

import numpy as np
import pylab
import matplotlib.pyplot as plt


out_path = "F:\\t-PCAtest\\dataset\\test20190725\\Wine\\"


def Hbeta(D=np.array([]), beta=1.0):
    """
        Compute the perplexity and the P-row for a specific value of the
        precision of a Gaussian distribution.
    """

    # Compute P-row and corresponding perplexity
    P = np.exp(-D.copy() * beta)  # 一个向量
    sumP = sum(P)  # 改成Wine数据之后部分数据的sumP出现了值为0的情况
    H = np.log(sumP) + beta * np.sum(D * P) / sumP  # D与P是可以相乘的，结果就是数量积。最后的结果H是一个数
    P = P / sumP
    return H, P


def x2p(X=np.array([]), tol=1e-5, perplexity=30.0):
    """
        Performs a binary search to get P-values in such a way that each
        conditional Gaussian has the same perplexity.
    """

    # Initialize some variables
    print("Computing pairwise distances...")
    (n, d) = X.shape
    sum_X = np.sum(np.square(X), 1)  # 每一行的平方和，组成一个向量，好像是个行向量。 这里的D矩阵是距离的平方矩阵
    D = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X)  # 一个矩阵与一个行向量相加，这里的处理方式是让这个矩阵与一个每一行都是那个行向量的矩阵相加
    np.savetxt(out_path+"tsnedistance.csv", D, fmt='%f', delimiter=',')
    P = np.zeros((n, n))
    beta = np.ones((n, 1))
    logU = np.log(perplexity)

    # Loop over all datapoints
    for i in range(n):

        # Print progress
        if i % 500 == 0:
            print("Computing P-values for point %d of %d..." % (i, n))

        # Compute the Gaussian kernel and entropy for the current precision
        betamin = -np.inf
        betamax = np.inf
        Di = D[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))]  # np.r_是合并两个矩阵，列数相等。这一行的作用就是获取D中的一行，并且不包括这一行中对角线上的那个元素
        (H, thisP) = Hbeta(Di, beta[i])  # 这里的thisP是所有的点以i为中心的概率值，thisP是一个向量的形式

        # Evaluate whether the perplexity is within tolerance
        Hdiff = H - logU
        tries = 0
        while np.abs(Hdiff) > tol and tries < 50:

            # If not, increase or decrease precision， 这里就真的是二分法了
            if Hdiff > 0:
                betamin = beta[i].copy()
                if betamax == np.inf or betamax == -np.inf:
                    beta[i] = beta[i] * 2.
                else:
                    beta[i] = (beta[i] + betamax) / 2.
            else:
                betamax = beta[i].copy()
                if betamin == np.inf or betamin == -np.inf:
                    beta[i] = beta[i] / 2.
                else:
                    beta[i] = (beta[i] + betamin) / 2.

            # Recompute the values
            (H, thisP) = Hbeta(Di, beta[i])
            Hdiff = H - logU
            tries += 1

        # Set the final row of P
        P[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))] = thisP

    # Return final P-matrix
    print("Mean value of sigma: %f" % np.mean(np.sqrt(1 / beta)))
    return P


def pca(X=np.array([]), no_dims=50):
    """
        Runs PCA on the NxD array X in order to reduce its dimensionality to
        no_dims dimensions.
    """

    print("Preprocessing the data using PCA...")
    (n, d) = X.shape
    X = X - np.tile(np.mean(X, 0), (n, 1))
    (l, M) = np.linalg.eig(np.dot(X.T, X))
    Y = np.dot(X, M[:, 0:no_dims])
    return Y


def tsne(X=np.array([]), no_dims=2, initial_dims=50, perplexity=30.0, save_path=None):
    """
        Runs t-SNE on the dataset in the NxD array X to reduce its
        dimensionality to no_dims dimensions. The syntaxis of the function is
        `Y = tsne.tsne(X, no_dims, perplexity), where X is an NxD NumPy array.
    """

    # Check inputs
    if isinstance(no_dims, float):
        print("Error: array X should have type float.")
        return -1
    if round(no_dims) != no_dims:
        print("Error: number of dimensions should be an integer.")
        return -1

    # Initialize variables
    X = pca(X, initial_dims).real
    (n, d) = X.shape
    max_iter = 1000
    initial_momentum = 0.5  # 开始阶段的α
    final_momentum = 0.8  # 后期的α
    eta = 500
    min_gain = 0.01
    Y = np.random.randn(n, no_dims)
    dY = np.zeros((n, no_dims))
    iY = np.zeros((n, no_dims))
    gains = np.ones((n, no_dims))

    # Compute P-values
    P = x2p(X, 1e-5, perplexity)
    P = P + np.transpose(P)  # 对称化处理
    P = P / np.sum(P)
    P = P * 4.									# early exaggeration 早期夸大处理
    P = np.maximum(P, 1e-12)  # 就是将小于1e-12的数变为1e-12
    if not(save_path is None):
        np.savetxt(save_path+"tSNE_P.csv", P, fmt='%f', delimiter=',')

    # Run iterations
    for iter in range(max_iter):

        # Compute pairwise affinities
        sum_Y = np.sum(np.square(Y), 1)
        num = -2. * np.dot(Y, Y.T)
        num = 1. / (1. + np.add(np.add(num, sum_Y).T, sum_Y))
        num[range(n), range(n)] = 0.
        Q = num / np.sum(num)
        Q = np.maximum(Q, 1e-12)

        # Compute gradient
        PQ = P - Q
        for i in range(n):
            dY[i, :] = np.sum(np.tile(PQ[:, i] * num[:, i], (no_dims, 1)).T * (Y[i, :] - Y), 0)

        # Perform the update
        if iter < 20:
            momentum = initial_momentum
        else:
            momentum = final_momentum
        gains = (gains + 0.2) * ((dY > 0.) != (iY > 0.)) + \
                (gains * 0.8) * ((dY > 0.) == (iY > 0.))
        gains[gains < min_gain] = min_gain
        iY = momentum * iY - eta * (gains * dY)
        Y = Y + iY
        Y = Y - np.tile(np.mean(Y, 0), (n, 1))

        # Compute current value of cost function
        if (iter + 1) % 10 == 0:
            C = np.sum(P * np.log(P / Q))
            print("Iteration %d: error is %f" % (iter + 1, C))

            # plt.scatter(Y[0:50, 0], Y[0:50, 1], c='r', marker='o')
            # plt.scatter(Y[50:100, 0], Y[50:100, 1], c='g', marker='o')
            # plt.scatter(Y[100:150, 0], Y[100:150, 1], c='b', marker='o')
            # plt.show()

        # Stop lying about P-values
        if iter == 100:  # 早期的放大处理的恢复
            P = P / 4.

    if not(save_path is None):
        np.savetxt(save_path + 'Y.csv', Y, fmt='%f', delimiter=',')
        plt.scatter(Y[:, 0], Y[:, 1])
        plt.savefig(save_path+'Y.png')
        plt.close()

    # Return solution
    return Y


if __name__ == "__main__":
    print("Run Y = tsne.tsne(X, no_dims, perplexity) to perform t-SNE on your dataset.")
    print("Running example on 2,500 MNIST digits...")
    X = np.loadtxt("mnist2500_X.txt")
    labels = np.loadtxt("mnist2500_labels.txt")

    # path = "F:\\result2019\\result0513\\datasets\\Wine\\"
    # X_reader = np.loadtxt(path + "data.csv", dtype=np.str, delimiter=',')
    # label_reader = np.loadtxt(path + 'label.csv', np.str, delimiter=',')
    # X = X_reader[:, :].astype(np.float)
    # labels = label_reader.astype(np.int)

    Y = tsne(X, 2, 50, 20.0)
    pylab.scatter(Y[:, 0], Y[:, 1], 20, labels)
    pylab.show()
