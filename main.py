# coding=utf-8
from documents import DocumentNormalizer, DocumentTokenizer, StopList, InvertedIndex, CASMBlock, CS276Block
from query import Tree
from search_models import VectorModel, BooleanModel


stop_list = StopList('common_words')
tokenizer = DocumentTokenizer(stop_list)
normalizer = DocumentNormalizer()
invindex_list = []
retrieval_list = []
# cs_block = CS276Block('./pa1-data/*')
cs_block = CASMBlock('cacm.all')
for block in cs_block.get_next_block():
    invIndex = InvertedIndex()
    invindex_list.append(invIndex)
    doc_retrieval_block = {}
    for document in block:
        document.tokenize(tokenizer, normalizer, invIndex)
        doc_retrieval_block[document.id] = document.entry_string()
    retrieval_list.append(doc_retrieval_block)

doc_retrieval = retrieval_list[0]
for doc_retrieval_block in retrieval_list:
    doc_retrieval = {**doc_retrieval, **doc_retrieval_block}
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

doc_ids = inv_index.search("Boolean Functions Decimal", boolean_model, tokenizer, normalizer)
# doc_ids = inv_index.search("Boolean Decimal", vector_model)

print(doc_ids)
def pad(str):
    return re.sub( '^',' '*4, str ,flags=re.MULTILINE )

for i, doc_id in enumerate(doc_ids[0:min(len(doc_ids), 10)]):
    print("Result", str(i), "Document #", doc_id)
    print(pad(doc_retreival[doc_id]))
