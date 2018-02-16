# coding=utf-8
import re
from documents import DocumentNormalizer, DocumentTokenizer, StopList, InvertedIndex, CASMBlock, CS276Block
from query import Tree
from search_models import VectorModel, BooleanModel


stop_list = StopList('common_words')
tokenizer = DocumentTokenizer(stop_list)
normalizer = DocumentNormalizer()
# invindex_list = []
retrieval_list = []
cs_block = CS276Block('./pa1-data/*')
# cs_block = CASMBlock('cacm.all')
doc_list = []
for block in cs_block.get_next_block():
    doc_list += list(block)
invIndex = InvertedIndex([])
doc_retrieval_block = {}
for document in doc_list:
    document.tokenize(tokenizer, normalizer, invIndex)
    doc_retrieval_block[document.id] = document.entry_string()
invIndex.post_register_hook()

from IPython import embed
embed()