import argparse
import pickle
import re

from documents import DocumentNormalizer, DocumentTokenizer, StopList, InvertedIndex, CASMBlock, CS276Block
from query import Tree
from search_models import VectorModel, BooleanModel


parser = argparse.ArgumentParser(
    description='Runs a search given a collection, a model, and weights')

parser.add_argument(
    '-m',
    '--model',
    choices=['vector', 'boolean'],
    default='vector',
    help='the model to use for search'
)

parser.add_argument(
    '-w',
    '--weights',
    choices=['tf-idf', 'tf-idf-norm', 'norm-freq'],
    default='tf-idf',
    help='the weights to use in vector model')

parser.add_argument(
    '-sw',
    '--sourceweights',
    choices=['tf-idf', 'tf-idf-norm', 'norm-freq', 'all', 'none'],
    default= 'all',
    help='which index file to read'
)

parser.add_argument(
    '-c',
    '--collection',
    choices=['cacm', 'cs276'],
    default='cacm',
    help='the collection to build the inverted index of'
)

parser.add_argument(
    '-i',
    '--input',
    help = 'a string to search'
)

args = parser.parse_args()

indexFile = './inv_index_' + args.collection + '_' + args.sourceweights + '.pkl'
docRetreiveFile = './doc_retreive_' + args.collection + '_' + args.sourceweights + '.pkl'

stop_list = StopList('common_words')
tokenizer = DocumentTokenizer(stop_list)
normalizer = DocumentNormalizer()


def pad(str):
    return re.sub( '^',' '*4, str ,flags=re.MULTILINE )


if args.model == 'vector':
    model = VectorModel(args.weights)
elif args.model == 'boolean':
    model = BooleanModel()
else:
    raise Exception('Model ' + args.model + ' does not exist')

with open(docRetreiveFile, 'rb') as f:
    retreive_dict = pickle.load(f)
    print("loaded retreive from " + docRetreiveFile)

    inv_index = InvertedIndex([])
    inv_index.load(indexFile)
    print("loaded inverted index from " + indexFile)

    while(True):
        user_input = args.input if args.input else input("> ")
        if(user_input == 'quit'):
            break

        doc_ids = inv_index.search(user_input, model, tokenizer, normalizer)
        for i, doc_id in enumerate(doc_ids[:10]):
            print('Document ' + str(i + 1) + ' : #' + str(doc_id))
            print(pad(retreive_dict[doc_id]))
        
        if args.input:
            break