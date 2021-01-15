# 给定一个二维平面上的一个矩形区域，和k个中心点
# 将平面分割成k个区域，每个区域到其中一个中心点的距离最小
# 最后返回边界线
# 思路：任意两个点之间的边界是这两个点构成的线段的中垂线
# 这些中垂线上真正的边界线部分必然是到两个中心点的距离相等，并且到其他任意
# 一个中心点的距离都大于到这两个中心点的距离，而不是真正的边界线的部分
# 必然到某一个中心点的距离最小，而到其他所有中心点的距离都大于该距离。
# 但是实际计算中产生的浮点误差可能会干扰判断
# 划分出的子区域肯定都是凸的
# 两个点的中垂线上的线段只有可能是相关的这两个点的边界，不可能是其他点的边界
import numpy as np


def plane_split(rect, center_point):
    """

    :param rect: 要划分的矩形范围，包括四个顶点的坐标，依次是左上、左下、右上、右下四个顶点
    :param center_point: 中心点坐标，是一个np.array
    :return:
    """
    # 首先求出所有的中垂线集合
    (n, dim) = center_point.shape
    perpendicular = []  # 中垂线
    related_points = []  # 与中垂线相关的点

    for i in range(0, n-1):
        for j in range(i+1, n):
            if center_point[i, 0] == center_point[j, 0]:  # 斜率为无穷大
                perpendicular.append([float('inf'), center_point[i, 0]])
                related_points.append([i, j])
            else:
                k = (center_point[j, 1] - center_point[i, 1]) / (center_point[j, 0] - center_point[i, 0])
                b = (center_point[i, 1] + center_point[j, 1]) / 2 - k * (center_point[i, 0] + center_point[j, 0]) / 2
                perpendicular.append([k, b])
                related_points.append([i, j])

    # 需要判断一下有没有重叠的线
    # 计算每条中垂线与外边界矩形以及其他中垂线的交点，并按照从左到右、从上到下的顺序进行排序
    # 依次判断中垂线上的每一段是否是真正的边界线，也就是判断每一段是否到第三个中心点最近


def test():
    color = np.array([[112, 48, 160, 0.6*256],
                      [0, 32, 96, 0.6*256],
                      [0, 112, 192, 0.6*256],
                      [0, 176, 240, 0.6*256],
                      [0, 176, 80, 0.6*256],
                      [146, 208, 80, 0.6*256],
                      [255, 255, 0, 0.6*256],
                      [255, 192, 0, 0.6*256],
                      [255, 0, 0, 0.6*256],
                      [192, 0, 0, 0.6*256]])
    print(color/256)


if __name__ == '__main__':
    test()
