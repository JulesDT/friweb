
import argparse
import pickle

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

args = parser.parse_args()

indexFile = './inv_index_' + args.collection + '_' + args.sourceweights + '.pkl'
docRetreiveFile = './doc_retreive_' + args.collection + '_' + args.sourceweights + '.pkl'

stop_list = StopList('common_words')
tokenizer = DocumentTokenizer(stop_list)
normalizer = DocumentNormalizer()


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

    doc_ids = inv_index.search('computer science', model, tokenizer, normalizer)
    for i, doc_id in enumerate(doc_ids[:10]):
        print('Document ' + str(i + 1) + ' :')
        print(retreive_dict[doc_id])