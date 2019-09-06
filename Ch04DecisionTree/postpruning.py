# 后剪枝的CART决策树实现

from Ch04DecisionTree import TreeNode
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

    current_node = decision_tree
    bianli_list = []
    bianli_list.append(current_node)
    while len(bianli_list) > 0:
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



