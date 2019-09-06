# 对一些不太熟悉的数据结构语法进行测试


class Node:
    def __init__(self, value):
        self.value = value
        self.next = None


def test():
    root = Node(1)
    a = Node(2)
    b = Node(3)
    a.next = b
    root.next = a
    print('第一遍遍历')
    temp_node = root
    while not temp_node is None:
        print(temp_node.value)
        temp_node = temp_node.next
    print('第一遍遍历结束')
    print(root.value)


if __name__ == '__main__':
    test()

