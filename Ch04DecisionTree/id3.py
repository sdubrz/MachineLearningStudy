import numpy as np
import matplotlib.pyplot as plt
import math


def ent(labels):
    """
    样本集合的信息熵
    :param labels: 样本集合中数据的类别标签
    :return:
    """
    label_name = []
    label_count = []

    for item in labels:
        if not(item in label_name):
            label_name.append(item)
            label_count.append(1)
        else:
            index = label_name.index(item)
            label_count[index] = label_count[index] + 1

    n = sum(label_count)
    entropy = 0.0
    for item in label_count:
        p = item / n
        entropy = entropy - p*math.log(p, 2)

    return entropy


def gain(attribute, labels):
    """
    计算信息增益
    :param attribute: 集合中样本该属性的值列表
    :param labels: 集合中样本的数据标签
    :return:
    """
    is_value = False  # 当前属性是离散的形容词还是连续的数值
    ent_d = ent(labels)

    if is_value:
        # 连续的实现方法
        print("应该使用连续的方法")
    else:
        # 属性值是离散的值
        value_name = []
        value_count = []
        for item in attribute:
            if not (item in attribute):
                


def test():
    label = [1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3]
    a = ent(label)
    print(a)
    print(math.log(3, 2))


if __name__ == '__main__':
    test()

