# coding=utf-8
from documents import InvertedIndex
POSSIBLE_OPERATORS = '|&'


class Tree:

    def __init__(self, parent=None):
        self.operator = None
        self.childrens = []
        self.parent = parent

    def execute(self, inv_index):
        #  execute all leaves
        for i in range(len(self.childrens)):
            print(self.childrens[i])
            if isinstance(self.childrens[i], Tree):
                print("Execute", self.childrens[i])
                self.childrens[i].execute(inv_index)
            elif isinstance(self.childrens[i], InvertedIndex):
                pass
            else:
                self.childrens[i] = inv_index.filter(r"{}".format(self.childrens[i]))

    @staticmethod
    def parse(tree, query_string):
        if len(query_string) == 0:
            return
        query_string = query_string.replace(' ', '')
        if query_string[0] not in '()~' + POSSIBLE_OPERATORS:
            i = 1
            while i < len(query_string):
                if query_string[i] in POSSIBLE_OPERATORS + ')':
                    break
                i += 1
            if len(tree.childrens) > 0:
                if i < len(query_string):  # I have an operator
                    if query_string[i] in POSSIBLE_OPERATORS:
                        node = Tree(tree)
                        node.childrens.append(tree.childrens[0])
                        node.operator = tree.operator
                        node.childrens.append(query_string[:i])
                        tree.operator = query_string[i]
                        tree.childrens[0] = node
                        Tree.parse(tree, query_string[i + 1:])
                    elif query_string[i] == ')':
                        tree.childrens.append(query_string[:i])
                        Tree.parse(tree.parent, query_string[i + 1:])
                else:
                    tree.childrens.append(query_string)
                    return
            else:
                tree.operator = query_string[i]
                tree.childrens.append(query_string[:i])
                Tree.parse(tree, query_string[i + 1:])

        if query_string[0] in POSSIBLE_OPERATORS:
            tree.operator = query_string[0]
            Tree.parse(tree, query_string[1:])

        if query_string[0] == '(':
            node = Tree(tree)
            tree.childrens.append(node)
            Tree.parse(node, query_string[1:])

        if query_string[0] == ')':
            Tree.parse(node.parent, query_string[1:])

        if query_string[0] == '~':
            node = Tree(tree)
            if len(tree.childrens) > 0:
                tree.childrens[1] = node
            else:
                tree.childrens[0] = node
            node.operator = '~'
            node_child = Tree(node)
            node.childrens.append(node_child)
            Tree.parse(node_child, query_string[1:])
