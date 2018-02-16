# coding=utf-8
import re
from documents import DocumentNormalizer, DocumentTokenizer, StopList, InvertedIndex, CASMBlock, CS276Block
from query import Tree
from search_models import VectorModel, BooleanModel
import collections
from functools import reduce


stop_list = StopList('common_words')
tokenizer = DocumentTokenizer(stop_list)
normalizer = DocumentNormalizer()
doc_retrieval = {}
document_data_store = {}
# cs_block = CS276Block('./pa1-data/*')
cs_block = CASMBlock('cacm.all')
inv_index_list = []
retrieval_list = []
methods = []
''' 
Map
'''
document_list = set()
for block in cs_block.get_next_block():
    document_list.update(block)
    doc_retrieval_block = {}
    for document in block:
        doc_retrieval_block[document.id] = document.entry_string()
    retrieval_list.append(doc_retrieval_block)
'''
Map
'''
# mapped_data = InvertedIndex.map(document_list, tokenizer, normalizer, methods)
def map_tokenize(doc):
    return [
                (word, doc.id, 1)
                    for field in doc.fields_to_tokenize
                        for word in tokenizer.tokenize(getattr(doc, field), normalizer)
            ]
mapped_data = map(map_tokenize, document_list)
mapped_data = [item for sublist in mapped_data for item in sublist]
'''
Shuffle
'''
shuffled_data = collections.defaultdict(list)
for word, doc_id, value in mapped_data:
    shuffled_data[word].append((doc_id, value))
'''
Reduce
'''
def reducer(reduced_data, new_entry):
    for entry in new_entry[1]:
        reduced_data[new_entry[0]][entry[0]] += 1
    return reduced_data
inverted_index = reduce(reducer, shuffled_data.items(), collections.defaultdict(lambda: collections.defaultdict(int)))
'''
EndMapReduce
'''
inv_index = InvertedIndex(methods)
inv_index.inverted_index = inverted_index

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

# a = 0
# for word in inv_index.inverted_index:
#     a += sum([inv_index.inverted_index[word][doc] for doc in inv_index.inverted_index[word]])

# from IPython import embed
# embed()