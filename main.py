# coding=utf-8
import re
from documents import DocumentNormalizer, DocumentTokenizer, StopList, InvertedIndex, CASMBlock, CS276Block
from query import Tree
from search_models import VectorModel, BooleanModel


stop_list = StopList('common_words')
tokenizer = DocumentTokenizer(stop_list)
normalizer = DocumentNormalizer()
doc_retrieval = {}
document_data_store = {}
cs_block = CS276Block('./pa1-data/*')
# cs_block = CASMBlock('cacm.all')
cs_block = CASMBlock('cacm.all')
inv_index_list = []
retrieval_list = []
methods = []
''' 
Map
'''
for block in cs_block.get_next_block():
    doc_retrieval_block = {}
    for document in block:
        invIndex = InvertedIndex(methods)
        inv_index_list.append(invIndex)
        document.tokenize(tokenizer, normalizer, invIndex)
        doc_retrieval_block[document.id] = document.entry_string()
        invIndex.post_register_hook()
    retrieval_list.append(doc_retrieval_block)

'''
Shuffle
'''
shuffled_data = InvertedIndex.shuffle(inv_index_list)
'''
Reduce
'''
inv_index = InvertedIndex.reduce(shuffled_data, methods)
'''
EndMapReduce
'''

doc_retrieval = retrieval_list[0]
for doc_retrieval_block in retrieval_list:
    doc_retrieval = {**doc_retrieval, **doc_retrieval_block}


# vector_model = VectorModel()
boolean_model = BooleanModel()

doc_ids = inv_index.search("about & Boolean & Functions & Decimal", boolean_model, tokenizer, normalizer)
# doc_ids = inv_index.search("Boolean Decimal", vector_model)

print(doc_ids)
def pad(str):
    return re.sub( '^',' '*4, str ,flags=re.MULTILINE )

for i, doc_id in enumerate(doc_ids[0:min(len(doc_ids), 10)]):
    print("Result", str(i), "Document #", doc_id)
    print(pad(doc_retrieval[doc_id]))
