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
for block in cs_block.get_next_block():
    # MapReduce
    
    for document in block:
        document_data = document.map(tokenizer, normalizer)
        document_data = document.shuffle(document_data)
        document_data = document.reduce(document_data)
        document_data_store[document.id] = document_data
        doc_retrieval[document.id] = document.entry_string()
    
# Create invIndex_element
inv_index = InvertedIndex([])
for doc_id, document_data in document_data_store.items():
    for (word, amount) in document_data:
        inv_index.inverted_index[word][doc_id] = amount


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
