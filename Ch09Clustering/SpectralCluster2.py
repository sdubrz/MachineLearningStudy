# K近邻的谱聚类实现
import numpy as np
from sklearn.neighbors import NearestNeighbors
from Ch09Clustering.kmeans import K_means
from Ch10DimensionReduction import PCA
from Tools import PreProcess
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


"""
    计算拉普拉斯矩阵的时候，如果采用的是标准的 L = D - W
    则应该将 L 特征值分解所得到的最小的 k 个特征值所对应的特征向量作为 k-means 算法的输入；
    而如果采用的是 L = D + W
    则应该将最大的 k 个特征值所对应的特征向量作为 k-means 算法的输入。
    在计算 W 时应该进行处理，使得 W 中每一行的和都为 1
"""


def similar(data, i, j):
    """计算相似度"""
    norm = np.linalg.norm(data[i, :] - data[j, :])
    delta = 1.0  # 感觉所有的点都用相同的方差值会有问题的
    return np.exp(-norm*norm/(2*delta*delta))


def laplace_matrix(data, n_nbrs, save_path=None):
    """
    计算拉普拉斯矩阵
    :param data: 数据矩阵，每一行是一个样本
    :param n_nbrs: K近邻的K
    :param save_path: 保存中间结果的文件目录，用于程序调试
    :return:
    """
    (n, dim) = data.shape
    W = np.zeros((n, n))

    neighbors = NearestNeighbors(n_neighbors=n_nbrs, algorithm='ball_tree').fit(data)
    distance, nbs_index = neighbors.kneighbors(data)

    for i in range(0, n):
        for j in range(i+1, n):
            if j in nbs_index[i, :]:
                W[i, j] = similar(data, i, j)
                W[j, i] = W[i, j]
        W[i, i] = 0

    if not(save_path is None):
        np.savetxt(save_path+"similar_matrix.csv", W, fmt='%f', delimiter=',')

    for i in range(0, n):
        s = np.sum(W[i, :])
        if s == 0:
            continue
        W[i, :] = -1 * W[i, :] / s
        W[i, i] = 1

    return W


def spectral_cluster(data, n_cluster=3, n_nbrs=15, save_path=None):
    """
    K近邻图的谱聚类方法，未进行正则化
    :param data: 数据矩阵，每一行是一个样本
    :param n_cluster: 聚类的簇数
    :param n_nbrs: 近邻数
    :param save_path: 保存中间结果的文件路径，用于程序调试
    :return:
    """
    (n, dim) = data.shape
    L = laplace_matrix(data, n_nbrs, save_path=save_path)
    np.savetxt(save_path+"laplace matrix.csv", L, fmt='%f', delimiter=',')

    eigen_values, eigen_vectors = np.linalg.eig(L)
    idx = eigen_values.argsort()
    eigen_vectors = eigen_vectors[:, idx]

    np.savetxt(save_path+"eigenvectors.csv", eigen_vectors, fmt='%f', delimiter=',')
    np.savetxt(save_path+"eigenvalues.csv", eigen_values[idx], fmt='%f', delimiter=',')

    # eigen_data = eigen_vectors[:, n-n_cluster:n]
    eigen_data = eigen_vectors[:, 0:n_cluster]
    np.savetxt(save_path+"eigendata.csv", eigen_data, fmt='%f', delimiter=',')
    kmeans = K_means(eigen_data, n_cluster)
    label = kmeans.fit_transform()

    return label


def run_test():
    main_path = 'E:\\Project\\result2019\\TPCA1008\\'
    data_name = 'digits5_8'
    read_path = main_path + "datasets\\" + data_name + "\\"
    data_reader = np.loadtxt(read_path + 'data.csv', dtype=np.str, delimiter=',')
    label_reader = np.loadtxt(read_path + 'label.csv', dtype=np.str, delimiter=',')
    X = data_reader[:, :].astype(np.float)
    label_true = label_reader.astype(np.int)

    X = PreProcess.normalize(X)
    (n, dim) = X.shape
    n_clusters = 5
    n_nbrs = 15

    save_path = main_path + "spectralCluster\\" + data_name + "\\"
    PreProcess.check_filepath(save_path)

    label = spectral_cluster(X, n_clusters, n_nbrs, save_path=save_path)

    tsne = TSNE(n_components=2, perplexity=30.0)
    Y = tsne.fit_transform(X)

    colors = ['c', 'm', 'y', 'b', 'r', 'g']
    shapes = ['s', 'o', '^', 'p', '+', '*']
    for i in range(0, n):
        plt.scatter(Y[i, 0], Y[i, 1], c=colors[int(label[i])], marker=shapes[int(label_true[i])])

    plt.show()


def test():
    m = 10
    a = [0 for x in range(m)]
    print(a)


if __name__ == '__main__':
    run_test()
