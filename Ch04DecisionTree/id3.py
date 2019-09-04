import numpy as np
import matplotlib.pyplot as plt
import math
from Ch04DecisionTree import Dataset


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


def gain(attribute, labels, is_value=False):
    """
    计算信息增益
    :param attribute: 集合中样本该属性的值列表
    :param labels: 集合中样本的数据标签
    :return:
    """
    # is_value = False  # 当前属性是离散的形容词还是连续的数值
    info_gain = ent(labels)
    n = len(labels)
    split_value = None  # 如果是连续值的话，也需要返回分隔界限的值

    if is_value:
        # 属性值是连续的数值，首先应该使用二分法寻找最佳分割点
        print("应该使用连续的方法")
        sorted_attribute = attribute.copy()
        sorted_attribute.sort()
        split = []  # 候选的分隔点
        for i in range(0, n-1):
            temp = (sorted_attribute[i] + sorted_attribute[i+1]) / 2
            split.append(temp)
        info_gain_list = []
        for temp_split in split:
            low_labels = []
            high_labels = []
            for i in range(0, n):
                if attribute[i] <= temp_split:
                    low_labels.append(labels[i])
                else:
                    high_labels.append(labels[i])
            temp_gain = info_gain - len(low_labels)/n*ent(low_labels) - len(high_labels)/n*ent(high_labels)
            info_gain_list.append(temp_gain)

        info_gain = max(info_gain_list)
        max_index = info_gain_list.index(info_gain)
        split_value = split[max_index]
    else:
        # 属性值是离散的值
        attribute_dict = {}
        label_dict = {}
        index = 0
        for item in attribute:
            if attribute_dict.__contains__(item):
                attribute_dict[item] = attribute_dict[item] + 1
                label_dict[item].append(labels[index])
            else:
                attribute_dict[item] = 1
                label_dict[item] = [labels[index]]
            index += 1

        for key, value in attribute_dict.items():
            info_gain = info_gain - value/n * ent(label_dict[key])

    return info_gain, split_value


def id3_tree(Data, title, label):
    """
    id3方法构造决策树，使用的标准是信息增益（信息熵）
    :param Data: 数据集，每个样本是一个 list
    :param title: 每个属性的名字，如 色泽、含糖率等
    :param label: 存储的是每个样本的类别
    :return:
    """


def test():
    a = [1, 3, 6, 7, 4, 8, 0, 7]
    b = a.copy()
    b.sort()
    print(b)
    print(a)


def run_test():
    watermelon, title, title_full = Dataset.watermelon3()
    color = []
    density = []
    labels = []
    for melon in watermelon:
        color.append(melon[0])
        density.append(melon[6])
        labels.append(melon[8])
    gain1, split1 = gain(color, labels, is_value=False)
    gain2, split2 = gain(density, labels, is_value=True)

    print(gain1)
    print(split1)
    print(gain2)
    print(split2)


if __name__ == '__main__':
    run_test()

