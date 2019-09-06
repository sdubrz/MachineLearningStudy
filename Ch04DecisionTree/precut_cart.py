# 预剪枝的CART决策树
from Ch04DecisionTree import TreeNode
from Ch04DecisionTree import Dataset
from Ch04DecisionTree import cart


def current_accuracy(tree_node=TreeNode.TreeNode(), test_data=[], test_label=[]):
    """
    计算当前决策树在训练数据集上的正确率
    :param tree_node: 要判断的决策树结点
    :param test_data: 测试数据集
    :param test_label: 测试数据集的label
    :return:
    """
    root_node = tree_node
    while not root_node.parent is None:
        root_node = root_node.parent

    accuracy = 0
    for i in range(0, len(test_label)):
        this_label = cart.classify_data(root_node, test_data[i])
        if this_label == test_label[i]:
            accuracy += 1
    # print(str(tree_node.index) + " 处，分对了"+str(accuracy))
    return accuracy / len(test_label)


def finish_node(current_node=TreeNode.TreeNode(), data=[], label=[], test_data=[], test_label=[]):
    """
    完成一个结点上的计算
    :param current_node: 当前计算的结点
    :param data: 数据集
    :param label: 数据集的 label
    :param test_data: 测试数据集
    :param test_label: 测试数据集的label
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

    # 先为当前结点添加一个临时判断，如果需要添加孩子结点，就再把它恢复成None
    data_count = {}
    for index in current_node.data_index:
        if data_count.__contains__(label[index]):
            data_count[label[index]] += 1
        else:
            data_count[label[index]] = 1
    before_judge = max(data_count, key=data_count.get)
    current_node.judge = before_judge
    before_accuracy = current_accuracy(current_node, test_data, test_label)

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
        this_gain, this_split_value = cart.gini_index(attr_values, current_label, cart.is_number(temp_data[title]))  # 如果属性值为数字，则认为是连续的
        title_gini[title] = this_gain
        title_split_value[title] = this_split_value

    best_attr = min(title_gini, key=title_gini.get)  # 基尼指数最小的属性名
    current_node.attribute_name = best_attr
    current_node.split = title_split_value[best_attr]
    rest_title.remove(best_attr)

    a_data = data[0]
    if cart.is_number(a_data[best_attr]):  # 如果是该属性的值为连续数值
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
        # 也需要先给子节点一个判断
        small_data_count = {}
        for index in small_child.data_index:
            if small_data_count.__contains__(label[index]):
                small_data_count[label[index]] += 1
            else:
                small_data_count[label[index]] = 1
        small_child_judge = max(small_data_count, key=small_data_count.get)
        small_child.judge = small_child_judge  # 临时添加的一个判断
        large_data_count = {}
        for index in large_child.data_index:
            if large_data_count.__contains__(label[index]):
                large_data_count[label[index]] += 1
            else:
                large_data_count[label[index]] = 1
        large_child_judge = max(large_data_count, key=large_data_count.get)
        large_child.judge = large_child_judge  # 临时添加的一个判断
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
            # 也需要先给子节点一个判断
            temp_data_count = {}
            for index in index_list:
                if temp_data_count.__contains__(label[index]):
                    temp_data_count[label[index]] += 1
                else:
                    temp_data_count[label[index]] = 1
            temp_child_judge = max(temp_data_count, key=temp_data_count.get)
            a_child.judge = temp_child_judge  # 临时添加的一个判断
            children_list.append(a_child)
        current_node.children = children_list

    current_node.judge = None
    later_accuracy = current_accuracy(current_node, test_data, test_label)
    # print(str(current_node.index)+"处，不剪枝的正确率是 "+str(later_accuracy) +"，剪枝的正确率是 "+str(before_accuracy))
    if before_accuracy > later_accuracy:
        current_node.children = None
        current_node.judge = before_judge
        # print(str(current_node.index)+"处进行剪枝")
        return
    else:
        # print(current_node.to_string())
        for child in current_node.children:  # 递归
            finish_node(child, data, label, test_data, test_label)


def precut_cart_tree(Data, title, label, test_data, test_label):
    """
    生成一颗预剪枝 CART 决策树
    :param Data: 训练数据集，每个样本是一个 dict(属性名：属性值)，整个 Data 是个大的 list
    :param title:   每个属性的名字，如 色泽、含糖率等
    :param label: 存储的是训练集每个样本的类别
    :param test_data: 测试数据集
    :param test_label: 测试数据集的标签
    :return:
    """
    n = len(Data)
    rest_title = title.copy()
    root_data = []
    for i in range(0, n):
        root_data.append(i)

    root_node = TreeNode.TreeNode(data_index=root_data, rest_attribute=title.copy())
    finish_node(root_node, Data, label, test_data, test_label)

    return root_node


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

    decision_tree = precut_cart_tree(train_data, title, train_label, test_data, test_label)
    print('训练的决策树是:')
    cart.print_tree(decision_tree)
    print('\n')

    test_judge = []
    for melon in test_data:
        test_judge.append(cart.classify_data(decision_tree, melon))
    print('决策树在测试数据集上的分类结果是：', test_judge)
    print('测试数据集的正确类别信息应该是：  ', test_label)

    accuracy = 0
    for i in range(0, len(test_label)):
        if test_label[i] == test_judge[i]:
            accuracy += 1
    accuracy /= len(test_label)
    print('决策树在测试数据集上的分类正确率为：' + str(accuracy * 100) + "%")


if __name__ == '__main__':
    run_test()
