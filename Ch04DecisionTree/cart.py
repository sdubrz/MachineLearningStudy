# CART决策树，使用基尼指数（Gini index）来选择划分属性
# 分别实现预剪枝、后剪枝和不进行剪枝的实现

import math
from Ch04DecisionTree import TreeNode
from Ch04DecisionTree import Dataset


def is_number(s):
    """判断一个字符串是否为数字，如果是数字，我们认为这个属性的值是连续的"""
    try:
        float(s)
        return True
    except ValueError:
        pass
    return False


def gini(labels=[]):
    """
    计算数据集的基尼值
    :param labels: 数据集的类别标签
    :return:
    """
    data_count = {}
    for label in labels:
        if data_count.__contains__(label):
            data_count[label] += 1
        else:
            data_count[label] = 1

    n = len(labels)
    if n == 0:
        return 0

    gini_value = 1
    for key, value in data_count.items():
        gini_value = gini_value - (value/n)*(value/n)

    return gini_value


def gini_index_basic(n, attr_labels={}):
    gini_value = 0
    for attribute, labels in attr_labels.items():
        gini_value = gini_value + len(labels) / n * gini(labels)

    return gini_value


def gini_index(attributes=[], labels=[], is_value=False):
    """
    计算一个属性的基尼指数
    :param attributes: 当前数据在该属性上的属性值列表
    :param labels:
    :param is_value:
    :return:
    """
    n = len(labels)
    attr_labels = {}
    gini_value = 0  # 最终要返回的结果
    split = None  #

    if is_value:  # 属性值是连续的数值
        sorted_attributes = attributes.copy()
        sorted_attributes.sort()
        split_points = []
        for i in range(0, n-1):
            split_points.append((sorted_attributes[i+1]+sorted_attributes[i])/2)

        split_evaluation = []
        for current_split in split_points:
            low_labels = []
            up_labels = []
            for i in range(0, n):
                if attributes[i] <= current_split:
                    low_labels.append(labels[i])
                else:
                    up_labels.append(labels[i])
            attr_labels = {'small': low_labels, 'large': up_labels}
            split_evaluation.append(gini_index_basic(n, attr_labels=attr_labels))

        gini_value = min(split_evaluation)
        split = split_points[split_evaluation.index(gini_value)]

    else:  # 属性值是离散的词
        for i in range(0, n):
            if attr_labels.__contains__(attributes[i]):
                temp_list = attr_labels[attributes[i]]
                temp_list.append(labels[i])
            else:
                temp_list = []
                temp_list.append(labels[i])
                attr_labels[attributes[i]] = temp_list

        gini_value = gini_index_basic(n, attr_labels=attr_labels)

    return gini_value, split


def finish_node(current_node=TreeNode.TreeNode(), data=[], label=[]):
    """
    完成一个结点上的计算
    :param current_node: 当前计算的结点
    :param data: 数据集
    :param label: 数据集的 label
    :return:
    """
    n = len(label)

    # 判断当前结点中的数据是否属于同一类
    one_class = True
    this_data_index = current_node.data_index

    for i in this_data_index:
        for j in this_data_index:
            if label[i] != label[j]:
                one_class = False
                break
        if not one_class:
            break
    if one_class:
        current_node.judge = label[this_data_index[0]]
        return

    rest_title = current_node.rest_attribute  # 候选属性
    if len(rest_title) == 0:  # 如果候选属性为空，则是个叶子结点。需要选择最多的那个类作为该结点的类
        label_count = {}
        temp_data = current_node.data_index
        for index in temp_data:
            if label_count.__contains__(label[index]):
                label_count[label[index]] += 1
            else:
                label_count[label[index]] = 1
        final_label = max(label_count)
        current_node.judge = final_label
        return

    title_gini = {}  # 记录每个属性的基尼指数
    title_split_value = {}  # 记录每个属性的分隔值，如果是连续属性则为分隔值，如果是离散属性则为None
    for title in rest_title:
        attr_values = []
        current_label = []
        for index in current_node.data_index:
            this_data = data[index]
            attr_values.append(this_data[title])
            current_label.append(label[index])
        temp_data = data[0]
        this_gain, this_split_value = gini_index(attr_values, current_label, is_number(temp_data[title]))  # 如果属性值为数字，则认为是连续的
        title_gini[title] = this_gain
        title_split_value[title] = this_split_value

    best_attr = min(title_gini, key=title_gini.get)  # 基尼指数最小的属性名
    current_node.attribute_name = best_attr
    current_node.split = title_split_value[best_attr]
    rest_title.remove(best_attr)

    a_data = data[0]
    if is_number(a_data[best_attr]):  # 如果是该属性的值为连续数值
        split_value = title_split_value[best_attr]
        small_data = []
        large_data = []
        for index in current_node.data_index:
            this_data = data[index]
            if this_data[best_attr] <= split_value:
                small_data.append(index)
            else:
                large_data.append(index)
        small_str = '<=' + str(split_value)
        large_str = '>' + str(split_value)
        small_child = TreeNode.TreeNode(parent=current_node, data_index=small_data, attr_value=small_str,
                               rest_attribute=rest_title.copy())
        large_child = TreeNode.TreeNode(parent=current_node, data_index=large_data, attr_value=large_str,
                               rest_attribute=rest_title.copy())
        current_node.children = [small_child, large_child]

    else:  # 如果该属性的值是离散值
        best_titlevalue_dict = {}  # key是属性值的取值，value是个list记录所包含的样本序号
        for index in current_node.data_index:
            this_data = data[index]
            if best_titlevalue_dict.__contains__(this_data[best_attr]):
                temp_list = best_titlevalue_dict[this_data[best_attr]]
                temp_list.append(index)
            else:
                temp_list = [index]
                best_titlevalue_dict[this_data[best_attr]] = temp_list

        children_list = []
        for key, index_list in best_titlevalue_dict.items():
            a_child = TreeNode.TreeNode(parent=current_node, data_index=index_list, attr_value=key,
                               rest_attribute=rest_title.copy())
            children_list.append(a_child)
        current_node.children = children_list

    # print(current_node.to_string())
    for child in current_node.children:  # 递归
        finish_node(child, data, label)


def cart_tree(Data, title, label):
    """
    生成一颗 CART 决策树
    :param Data: 数据集，每个样本是一个 dict(属性名：属性值)，整个 Data 是个大的 list
    :param title:   每个属性的名字，如 色泽、含糖率等
    :param label: 存储的是每个样本的类别
    :return:
    """
    n = len(Data)
    rest_title = title.copy()
    root_data = []
    for i in range(0, n):
        root_data.append(i)

    root_node = TreeNode.TreeNode(data_index=root_data, rest_attribute=title.copy())
    finish_node(root_node, Data, label)

    return root_node


def print_tree(root=TreeNode.TreeNode()):
    """
    打印输出一颗树
    :param root: 根节点
    :return:
    """
    node_list = [root]
    while(len(node_list)>0):
        current_node = node_list[0]
        print('--------------------------------------------')
        print(current_node.to_string())
        print('--------------------------------------------')
        children_list = current_node.children
        if not (children_list is None):
            for child in children_list:
                node_list.append(child)
        node_list.remove(current_node)


def classify_data(decision_tree=TreeNode.TreeNode(), x={}):
    """
    使用决策树判断一个数据样本的类别标签
    :param decision_tree: 决策树的根节点
    :param x: 要进行判断的样本
    :return:
    """
    current_node = decision_tree
    while current_node.judge is None:
        if current_node.split is None:  # 离散属性
            can_judge = False  # 如果训练数据集不够大，测试数据集中可能会有在训练数据集中没有出现过的属性值
            for child in current_node.children:
                if child.attribute_value == x[current_node.attribute_name]:
                    current_node = child
                    can_judge = True
                    break
            if not can_judge:
                return None
        else:
            child_list = current_node.children
            if x[current_node.attribute_name] <= current_node.split:
                current_node = child_list[0]
            else:
                current_node = child_list[1]

    return current_node.judge


def run_test():
    train_watermelon, test_watermelon, title = Dataset.watermelon2()

    # 先处理数据
    train_data = []
    test_data = []
    train_label = []
    test_label = []
    for melon in train_watermelon:
        a_dict = {}
        dim = len(melon) - 1
        for i in range(0, dim):
            a_dict[title[i]] = melon[i]
        train_data.append(a_dict)
        train_label.append(melon[dim])
    for melon in test_watermelon:
        a_dict = {}
        dim = len(melon) - 1
        for i in range(0, dim):
            a_dict[title[i]] = melon[i]
        test_data.append(a_dict)
        test_label.append(melon[dim])

    decision_tree = cart_tree(train_data, title, train_label)
    print('训练的决策树是:')
    print_tree(decision_tree)
    print('\n')

    test_judge = []
    for melon in test_data:
        test_judge.append(classify_data(decision_tree, melon))
    print('决策树在测试数据集上的分类结果是：', test_judge)
    print('测试数据集的正确类别信息应该是：  ', test_label)

    accuracy = 0
    for i in range(0, len(test_label)):
        if test_label[i] == test_judge[i]:
            accuracy += 1
    accuracy /= len(test_label)
    print('决策树在测试数据集上的分类正确率为：'+str(accuracy*100)+"%")


if __name__ == '__main__':
    run_test()
