# coding=utf-8

import argparse
import pickle
import collections
from functools import reduce

from documents import DocumentNormalizer, DocumentTokenizer, StopList, InvertedIndex, CASMBlock, CS276Block
from query import Tree
from search_models import VectorModel, BooleanModel


parser = argparse.ArgumentParser(
    description='Builds inverted index given a method and a collection')

parser.add_argument(
    '-m',
    '--method',
    choices=['bsbi', 'map-reduce'],
    default='bsbi',
    help='the method to use to build the inverted index'
)

parser.add_argument(
    '-w',
    '--weights',
    choices=['tf-idf', 'tf-idf-norm', 'norm-freq', 'all', 'none'],
    default='all',
    help='the weights to use in vector model, use none to only generate the boolean model inv index')

parser.add_argument(
    '-c',
    '--collection',
    choices=['cacm', 'cs276'],
    default='cacm',
    help='the collection to build the inverted index of'
)

args = parser.parse_args()

indexOutputFile = './inv_index_' + args.collection + '_' + args.weights + '.pkl'
docRetreiveFile = './doc_retreive_' + args.collection + '_' + args.weights + '.pkl'

if args.weights == 'none' :
    args.weights = []
elif args.weights == 'all':
    args.weights = ['tf-idf', 'tf-idf-norm', 'norm-freq']
else:
    args.weights = [args.weights]

stop_list = StopList('common_words')
tokenizer = DocumentTokenizer(stop_list)
normalizer = DocumentNormalizer()

def bsbi():
    invindex_list = []
    retrieval_list = []

    if args.collection == 'cacm':
        cs_block = CASMBlock('cacm.all')
    elif args.collection == 'cs276':
        cs_block = CS276Block('./pa1-data/*')
    else:
        raise Exception('Collection ' + args.collection + ' not supported')

    for block in cs_block.get_next_block():
        invIndex = InvertedIndex(args.weights)
        invindex_list.append(invIndex)
        doc_retrieval_block = {}
        for document in block:
            document.tokenize(tokenizer, normalizer, invIndex)
            doc_retrieval_block[document.id] = document.entry_string()
        retrieval_list.append(doc_retrieval_block)
        invIndex.post_register_hook()

    doc_retrieval = retrieval_list[0]
    for doc_retrieval_block in retrieval_list:
        doc_retrieval = {**doc_retrieval, **doc_retrieval_block}
    for inv_index in invindex_list[1:]:
        invindex_list[0].merge(inv_index)

    inv_index = invindex_list[0]

    inv_index.save(indexOutputFile)
    print("inverted index saved to file " + indexOutputFile)
    with open(docRetreiveFile, 'wb') as f:
        pickle.dump(doc_retrieval, f, pickle.HIGHEST_PROTOCOL)
    print("retreival index saved to file " + docRetreiveFile)
def map_reduce():
    stop_list = StopList('common_words')
    tokenizer = DocumentTokenizer(stop_list)
    normalizer = DocumentNormalizer()
    doc_retrieval = {}
    document_data_store = {}
    retrieval_list = []
    if args.collection == 'cacm':
        cs_block = CASMBlock('cacm.all')
    elif args.collection == 'cs276':
        cs_block = CS276Block('./pa1-data/*')
    else:
        raise Exception('Collection ' + args.collection + ' not supported')
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
    inv_index = InvertedIndex(args.weights)
    inv_index.inverted_index = inverted_index
    inv_index.post_register_hook()


    doc_retrieval = retrieval_list[0]
    for doc_retrieval_block in retrieval_list:
        doc_retrieval = {**doc_retrieval, **doc_retrieval_block}

    inv_index.save(indexOutputFile)
    print("inverted index saved to file " + indexOutputFile)
    with open(docRetreiveFile, 'wb') as f:
        pickle.dump(doc_retrieval, f, pickle.HIGHEST_PROTOCOL)
    print("retreival index saved to file " + docRetreiveFile)

if(args.method) == 'bsbi':
    bsbi()
elif args.method == 'map-reduce':
    map_reduce()