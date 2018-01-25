# coding=utf-8
from documents import *
from query import Tree


stop_list = StopList('common_words')
tokenizer = DocumentTokenizer(stop_list)
normalizer = DocumentNormalizer()
invindex_list = []
cs_block = CS276Block('./pa1-data/*')
# cs_block = CASMBlock('cacm.all')
for block in cs_block.get_next_block():
    invIndex = InvertedIndex()
    invindex_list.append(invIndex)
    for document in block:
        document.tokenize(tokenizer, normalizer, invIndex)

for inv_index in invindex_list[1:]:
    invindex_list[0].merge(inv_index)
# print(invindex_list[0].filter(r"inter"))
inv_index = invindex_list[0]
# inv_index.build_base_vector()

from IPython import embed
embed()

# tree = Tree(parent=None)
# Tree.parse(tree, '(interfax | fax) & ~fax')
# tree.query(inv_index)