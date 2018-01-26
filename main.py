# coding=utf-8
from documents import *
from query import Tree
from search_models import VectorModel, BooleanModel

doc_retreival = {}

stop_list = StopList('common_words')
tokenizer = DocumentTokenizer(stop_list)
normalizer = DocumentNormalizer()
invindex_list = []
# cs_block = CS276Block('./pa1-data/*')
cs_block = CASMBlock('cacm.all')
for block in cs_block.get_next_block():
    invIndex = InvertedIndex()
    invindex_list.append(invIndex)
    for document in block:
        document.tokenize(tokenizer, normalizer, invIndex)
        doc_retreival[document.id] = document.entry_string()

for inv_index in invindex_list[1:]:
    invindex_list[0].merge(inv_index)
# print(invindex_list[0].filter(r"inter"))

inv_index = invindex_list[0]
# inv_index.build_base_vector()

# tree = Tree(parent=None)
# Tree.parse(tree, '(inter & the) | politics')
# tree.execute(inv_index)

inv_index.build_tf_idf()

vector_model = VectorModel()
boolean_model = BooleanModel()

doc_ids = inv_index.search("Boolean Functions Decimal", boolean_model)
# doc_ids = inv_index.search("Boolean Decimal", vector_model)

print(doc_ids)
def pad(str):
    return re.sub( '^',' '*4, str ,flags=re.MULTILINE )

for i, doc_id in enumerate(doc_ids[0:min(len(doc_ids), 10)]):
    print("Result", str(i), "Document #", doc_id)
    print(pad(doc_retreival[doc_id]))
