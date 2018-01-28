# coding=utf-8
import re
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
block_triplets_sets = []
for block in cs_block.get_next_block():
    invIndex = InvertedIndex()
    invindex_list.append(invIndex)
    doc_retrieval_block = {}
    for document in block:
        document.tokenize(tokenizer, normalizer, invIndex)
        doc_retrieval_block[document.id] = document.entry_string()
    retrieval_list.append(doc_retrieval_block)
    invIndex.post_register_hook()
    
    
    # Map
    # block_triplets = set()
    # block_triplets_sets.append(block_triplets)
    # doc_retrieval_block = {}
    # for document in block:
    #     for (doc_id, word, amount) in document.tokenize(tokenizer, normalizer, None, True):
    #         block_triplets.add((doc_id, word, amount))
    #     doc_retrieval_block[document.id] = document.entry_string()
    # retrieval_list.append(doc_retrieval_block)

# Reduce
invIndex = InvertedIndex([])
for block_set in block_triplets_sets:
    for (doc_id, word, amount) in block_set:
        if word in invIndex.inverted_index:
            if doc_id in invIndex.inverted_index[word]:
                invIndex.inverted_index[word][doc_id] += amount
            else:
                invIndex.inverted_index[word][doc_id] = amount
        else:
            invIndex.inverted_index[word] = {doc_id: amount}



# doc_retrieval = retrieval_list[0]
# for doc_retrieval_block in retrieval_list:
#     doc_retrieval = {**doc_retrieval, **doc_retrieval_block}
# for inv_index in invindex_list[1:]:
#     invindex_list[0].merge(inv_index)

# print(invindex_list[0].filter(r"inter"))

inv_index = invindex_list[0]


vector_model = VectorModel()
boolean_model = BooleanModel()

doc_ids = inv_index.search("about & Boolean & Functions & Decimal", boolean_model, tokenizer, normalizer)
# doc_ids = inv_index.search("Boolean Decimal", vector_model)

print(doc_ids)
def pad(str):
    return re.sub( '^',' '*4, str ,flags=re.MULTILINE )

for i, doc_id in enumerate(doc_ids[0:min(len(doc_ids), 10)]):
    print("Result", str(i), "Document #", doc_id)
    print(pad(doc_retrieval[doc_id]))
