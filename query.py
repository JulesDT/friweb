# coding=utf-8
POSSIBLE_OPERATORS = '|&'


class Tree:

    def __init__(self, parent=None):
        self.operator = None
        self.childrens = []
        self.parent = parent

    def normalize(self, tokenizer, normalizer):
        for i in range(len(self.childrens)):
            if isinstance(self.childrens[i], Tree):
                self.childrens[i].normalize(tokenizer, normalizer)
            # This happens before anything. So isintance(self.childrens[i], set) should never happen
            else:
                try:
                    self.childrens[i] = next(tokenizer.tokenize(self.childrens[i], normalizer))
                except StopIteration:
                    for j in range(len(self.parent.childrens)):
                        if self.parent.childrens[j] == self:
                            break
                    self.parent.childrens[j] = self.childrens[i ^ 1] # 1 ^ 1 = 0, 0 ^ 1 = 1 :)
                    # Small hack, but this prevent from having a non-normalized token make it to the layer above without being normalized
                    self.parent.normalize(tokenizer, normalizer)

    def prepare(self, inv_index):
        for i in range(len(self.childrens)):
            if isinstance(self.childrens[i], Tree):
                self.childrens[i].prepare(inv_index)
            elif isinstance(self.childrens[i], set):
                pass
            else:
                self.childrens[i] = set(inv_index.inverted_index.get(self.childrens[i], {}).keys())

    def execute(self, inv_index):
        for i in range(len(self.childrens)):
            if isinstance(self.childrens[i], Tree):
                self.childrens[i] = self.childrens[i].execute(inv_index)
        if self.operator == '&':
            return self.childrens[0] & self.childrens[1]
        if self.operator == '|':
            return self.childrens[0] | (self.childrens[1])
        if self.operator == '~':
            return inv_index - self.childrens[0]
        if self.operator is None and len(self.childrens) == 1:
            return self.childrens[0]

    def query(self, inv_index, tokenizer, normalizer):
        #  execute all leaves
        self.normalize(tokenizer, normalizer)
        self.prepare(inv_index)
        return self.execute(inv_index)

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
                        if isinstance(tree.childrens[0], Tree):
                            tree.childrens[0].parent = node
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
                if i < len(query_string):
                    tree.operator = query_string[i]
                    tree.childrens.append(query_string[:i])
                    Tree.parse(tree, query_string[i + 1:])
                else:
                    tree.childrens.append(query_string)

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
            tree.childrens.append(node)
            node.operator = '~'
            Tree.parse(node, query_string[1:])
