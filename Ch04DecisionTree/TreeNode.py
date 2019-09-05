# 决策树的结点类实现


class TreeNode:
    """
    决策树结点类
    """
    current_index = 0

    def __init__(self, parent=None, attr_name=None, children=None, judge=None, split=None, data_index=None,
                 attr_value=None, rest_attribute=None):
        """
        决策树结点类初始化方法
        :param parent: 父节点
        """
        self.parent = parent  # 父节点，根节点的父节点为 None
        self.attribute_name = attr_name  # 本节点上进行划分的属性名
        self.attribute_value = attr_value  # 本节点上划分属性的值，是与父节点的划分属性名相对应的
        self.children = children  # 孩子结点列表
        self.judge = judge  # 如果是叶子结点，需要给出判断
        self.split = split  # 如果是使用连续属性进行划分，需要给出分割点
        self.data_index = data_index  # 对应训练数据集的训练索引号
        self.index = TreeNode.current_index  # 当前结点的索引号，方便输出时查看
        self.rest_attribute = rest_attribute  # 尚未使用的属性列表
        TreeNode.current_index += 1

    def to_string(self):
        """用一个字符串来描述当前结点信息"""
        this_string = 'current index : ' + str(self.index) + ";\n"
        if not (self.parent is None):
            parent_node = self.parent
            this_string = this_string + 'parent index : ' + str(parent_node.index) + ";\n"
            this_string = this_string + str(parent_node.attribute_name) + " : " + str(self.attribute_value) + ";\n"
        this_string = this_string + "data : " + str(self.data_index) + ";\n"
        if not(self.children is None):
            this_string = this_string + 'select attribute is : ' + str(self.attribute_name) + ";\n"
            child_list = []
            for child in self.children:
                child_list.append(child.index)
            this_string = this_string + 'children : ' + str(child_list)
        if not (self.judge is None):
            this_string = this_string + 'label : ' + self.judge
        return this_string
