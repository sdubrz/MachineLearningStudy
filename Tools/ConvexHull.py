import matplotlib.pyplot as plt
import math

import sklearn.datasets as datasets

"""
使用Graham扫描法计算凸包
网上的代码好多运行效果并不好
算法参见《算法导论》第三版 第605页
"""


def get_bottom_point(points):
    """
    返回points中纵坐标最小的点的索引，如果有多个纵坐标最小的点则返回其中横坐标最小的那个
    :param points:
    :return:
    """
    min_index = 0
    n = len(points)
    for i in range(0, n):
        if points[i][1] < points[min_index][1] or (
                points[i][1] == points[min_index][1] and points[i][0] < points[min_index][0]):
            min_index = i
    return min_index


def sort_polar_angle_cos(points, center_point):
    """
    按照与中心点的极角进行排序，使用的是余弦的方法
    :param points: 需要排序的点
    :param center_point: 中心点
    :return:
    """
    n = len(points)
    cos_value = []
    rank = []
    norm_list = []
    for i in range(0, n):
        point_ = points[i]
        point = [point_[0] - center_point[0], point_[1] - center_point[1]]
        rank.append(i)
        norm_value = math.sqrt(point[0] * point[0] + point[1] * point[1])
        norm_list.append(norm_value)
        if norm_value == 0:
            cos_value.append(1)
        else:
            cos_value.append(point[0] / norm_value)

    for i in range(0, n - 1):
        index = i + 1
        while index > 0:
            if cos_value[index] > cos_value[index - 1] or (
                    cos_value[index] == cos_value[index - 1] and norm_list[index] > norm_list[index - 1]):
                temp = cos_value[index]
                temp_rank = rank[index]
                temp_norm = norm_list[index]
                cos_value[index] = cos_value[index - 1]
                rank[index] = rank[index - 1]
                norm_list[index] = norm_list[index - 1]
                cos_value[index - 1] = temp
                rank[index - 1] = temp_rank
                norm_list[index - 1] = temp_norm
                index = index - 1
            else:
                break
    sorted_points = []
    for i in rank:
        sorted_points.append(points[i])

    return sorted_points


def vector_angle(vector):
    """
    返回一个向量与向量 [1, 0]之间的夹角， 这个夹角是指从[1, 0]沿逆时针方向旋转多少度能到达这个向量
    :param vector:
    :return:
    """
    norm_ = math.sqrt(vector[0] * vector[0] + vector[1] * vector[1])
    if norm_ == 0:
        return 0

    angle = math.acos(vector[0] / norm_)
    if vector[1] >= 0:
        return angle
    else:
        return 2 * math.pi - angle


def coss_multi(v1, v2):
    """
    计算两个向量的叉乘
    :param v1:
    :param v2:
    :return:
    """
    return v1[0] * v2[1] - v1[1] * v2[0]


def graham_scan(points):
    # print("Graham扫描法计算凸包")
    bottom_index = get_bottom_point(points)
    bottom_point = points.pop(bottom_index)
    sorted_points = sort_polar_angle_cos(points, bottom_point)

    m = len(sorted_points)
    if m < 2:
        print("点的数量过少，无法构成凸包")
        return

    stack = []
    stack.append(bottom_point)
    stack.append(sorted_points[0])
    stack.append(sorted_points[1])

    for i in range(2, m):
        length = len(stack)
        top = stack[length - 1]
        next_top = stack[length - 2]
        v1 = [sorted_points[i][0] - next_top[0], sorted_points[i][1] - next_top[1]]
        v2 = [top[0] - next_top[0], top[1] - next_top[1]]

        while coss_multi(v1, v2) >= 0:
            stack.pop()
            length = len(stack)
            top = stack[length - 1]
            next_top = stack[length - 2]
            v1 = [sorted_points[i][0] - next_top[0], sorted_points[i][1] - next_top[1]]
            v2 = [top[0] - next_top[0], top[1] - next_top[1]]

        stack.append(sorted_points[i])

    return stack


def test1():
    points = [[1.1, 3.6],
              [2.1, 5.4],
              [2.5, 1.8],
              [3.3, 3.98],
              [4.8, 6.2],
              [4.3, 4.1],
              [4.2, 2.4],
              [5.9, 3.5],
              [6.2, 5.3],
              [6.1, 2.56],
              [7.4, 3.7],
              [7.1, 4.3],
              [7, 4.1]]

    for point in points:
        plt.scatter(point[0], point[1], marker='o', c='y')

    result = graham_scan(points)

    length = len(result)
    for i in range(0, length - 1):
        plt.plot([result[i][0], result[i + 1][0]], [result[i][1], result[i + 1][1]], c='r')
    plt.plot([result[0][0], result[length - 1][0]], [result[0][1], result[length - 1][1]], c='r')

    plt.show()


def test2():
    """
    使用复杂一些的数据测试程序运行效果
    :return:
    """
    iris = datasets.load_iris()
    data = iris.data
    points_ = data[:, 0:2]
    points__ = points_[0:50, :]
    points = points__.tolist()

    temp_index = 0
    for point in points:
        plt.scatter(point[0], point[1], marker='o', c='y')
        index_str = str(temp_index)
        plt.annotate(index_str, (point[0], point[1]))
        temp_index = temp_index + 1

    result = graham_scan(points)
    print(result)
    length = len(result)
    for i in range(0, length - 1):
        plt.plot([result[i][0], result[i + 1][0]], [result[i][1], result[i + 1][1]], c='r')
    plt.plot([result[0][0], result[length - 1][0]], [result[0][1], result[length - 1][1]], c='r')

    # for i in range(0, len(rank)):

    plt.show()


if __name__ == "__main__":
    test2()
