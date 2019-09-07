# 后剪枝的CART决策树实现

from Ch04DecisionTree import TreeNode
from Ch04DecisionTree import cart
from Ch04DecisionTree import Dataset


def current_accuracy(root_node=TreeNode.TreeNode(), test_data=[], test_label=[]):
    """
    计算当前决策树在训练数据集上的正确率
    :param root_node: 决策树的根节点
    :param test_data: 测试数据集
    :param test_label: 测试数据集的label
    :return:
    """
    # root_node = tree_node
    # while not (root_node.parent is None):
    #     root_node = root_node.parent

    accuracy = 0
    for i in range(0, len(test_label)):
        this_label = cart.classify_data(root_node, test_data[i])
        if this_label == test_label[i]:
            accuracy += 1
    
    return accuracy / len(test_label)


def post_pruning(decision_tree=TreeNode.TreeNode(), test_data=[], test_label=[], train_label=[]):
    """
    对决策树进行后剪枝操作
    :param decision_tree: 决策树根节点
    :param test_data: 测试数据集
    :param test_label: 测试数据集的标签
    :param train_label: 训练数据集的标签
    :return:
    """
    leaf_father = []  # 所有的孩子都是叶结点的结点集合

    bianli_list = []
    bianli_list.append(decision_tree)
    while len(bianli_list) > 0:
        current_node = bianli_list[0]
        children = current_node.children
        wanted = True  # 判断当前结点是否满足所有的子结点都是叶子结点
        if not (children is None):
            for child in children:
                bianli_list.append(child)
                temp_bool = (child.children is None)
                wanted = (wanted and temp_bool)
        else:
            wanted = False

        if wanted:
            leaf_father.append(current_node)
        bianli_list.remove(current_node)

    while len(leaf_father) > 0:
        # 如果叶父结点为空，则剪枝完成。对于不需要进行剪枝操作的叶父结点，我们也之间将其从leaf_father中删除
        current_node = leaf_father.pop()
        # 不进行剪枝在测试集上的正确率
        before_accuracy = current_accuracy(root_node=decision_tree, test_data=test_data, test_label=test_label)

        data_index = current_node.data_index
        label_count = {}
        for index in data_index:
            if label_count.__contains__(index):
                label_count[train_label[index]] += 1
            else:
                label_count[train_label[index]] = 1
        current_node.judge = max(label_count, key=label_count.get)  # 如果进行剪枝当前结点应该做出的判断
        later_accuracy = current_accuracy(root_node=decision_tree, test_data=test_data, test_label=test_label)

        if before_accuracy > later_accuracy:  # 不进行剪枝
            current_node.judge = None
        else:  # 进行剪枝
            current_node.children = None
            # 还需要检查是否需要对它的父节点进行判断
            parent_node = current_node.parent
            if not (parent_node is None):
                children_list = parent_node.children
                temp_bool = True
                for child in children_list:
                    if not (child.children is None):
                        temp_bool = False
                        break
                if temp_bool:
                    leaf_father.append(parent_node)
    return decision_tree


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

    decision_tree = cart.cart_tree(train_data, title, train_label)
    decision_tree = post_pruning(decision_tree=decision_tree, test_data=test_data, test_label=test_label, train_label=train_label)

    print('剪枝之后的决策树是:')
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
    print('决策树在测试数据集上的分类正确率为：'+str(accuracy*100)+"%")


if __name__ == '__main__':
    run_test()
