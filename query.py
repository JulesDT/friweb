# coding=utf-8
import re
possible_operators = '|&'


class Tree:

    def __init__(self, parent=None):
        self.operator = None
        self.childrens = []
        self.parent = parent


def parse(tree, query_string):
    # regex_word = re.compile(r"/^\w+/")
    # regex_parenthesis = re.compile(r"/^\(/")
    # regex_parenthesis_end = re.compile(r"/^\)/")

    # if query_string.match(regex_word):
    print(query_string)
    if len(query_string) == 0:
        return
    query_string = query_string.replace(' ', '')
    if query_string[0] not in '()~' + possible_operators:
        i = 1
        while i < len(query_string):
            if query_string[i] in possible_operators + ')':
                break
            i += 1
        if len(tree.childrens) > 0:
            if i < len(query_string):  # I have an operator
                if query_string[i] in possible_operators:
                    node = Tree(tree)
                    node.childrens.append(tree.childrens[0])
                    node.operator = tree.operator
                    node.childrens.append(query_string[:i])
                    tree.operator = query_string[i]
                    tree.childrens[0] = node
                    parse(tree, query_string[i + 1:])
                elif query_string[i] == ')':
                    tree.childrens.append(query_string[:i])
                    parse(tree.parent, query_string[i + 1:])
            else:
                tree.childrens.append(query_string)
                return
        else:
            tree.operator = query_string[i]
            tree.childrens.append(query_string[:i])
            parse(tree, query_string[i + 1:])

    if query_string[0] in possible_operators:
        tree.operator = query_string[0]
        parse(tree, query_string[1:])

        # find next operator
        # if len(children) > 0:
        #   si j'ai un operator dans la string
        #       je cree un nouveau node
        #       node.children[0] = tree.children[0]
        #       node.operator = tree.operator
        #       node.children[1] = mot trouvé
        #       node.parent = tree
        #       tree.operator = operator trouvé
        #       parse(tree, reste de la string)
        #   sinon
        #       je met le mot dans children[1] et basta
        # else:
        #   tree.operator = operator found
        #   node.children[0] = mot trouve
        #   parse(tree, reste_string)

    if query_string[0] == '(':
        node = Tree(tree)
        tree.childrens.append(node)
        parse(node, query_string[1:])
        # create new Node
        # if len(tree.children) > 0:
        #   tree.children[1] = node
        # else
        #   tree.children[0] = node
        # parse(node, reste_de_la_string)

    if query_string[0] == ')':
        parse(node.parent, query_string[1:])
        # parse(node.parent, reste_de_la_string)

    if query_string[0] == '~':
        node = Tree(tree)
        if len(tree.childrens) > 0:
            tree.childrens[1] = node
        else:
            tree.childrens[0] = node
        node.operator = '~'
        node_child = Tree(node)
        node.childrens.append(node_child)
        parse(node_child, query_string[1:])

        # new node
        # if len(tree.children) > 0:
        #   tree.children[1] = node
        # else
        #   tree.children[0] = node
        # node.operator = not
        # new node2
        # node.children[0] = node2
        # parse(node2, reste_de_la_string)


#     def parse(tree, str):
# if(str.match(/^\w+/)):
#     //soit ta string est en mode mot & mat | mut
#     tree.operator !=
#     tree.children[1] = tree
#     tree.parent = new Node()

#     tree.operator = &
#     tree.children[0] = mot
#     parse(tree, le reste de la string)
#     //soit ta string est en mode mot
#     ok
# if(str.match(/^\(/))
#     node = new Node()
#     tree.children[0] = node
#     parse(node, reste)

# if(str.match( ')'))
#     parse(tree.parent, reste)

tree = Tree()

query = "(A | D) & (B | C & E)"
parse(tree, query)