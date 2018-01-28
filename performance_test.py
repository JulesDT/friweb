import argparse
import re
import pickle
import collections
import math
import matplotlib.pyplot as plt

from documents import DocumentNormalizer, DocumentTokenizer, StopList, InvertedIndex, CASMBlock, CS276Block
from query import Tree
from search_models import VectorModel, BooleanModel

parser = argparse.ArgumentParser(
    description='Runs a performance test on cacm collection')

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

args = parser.parse_args()


indexFile = './inv_index_cacm_' + args.sourceweights + '.pkl'
docRetreiveFile = './doc_retreive_cacm_' + args.sourceweights + '.pkl'

stop_list = StopList('common_words')
tokenizer = DocumentTokenizer(stop_list)
normalizer = DocumentNormalizer()

if args.model == 'vector':
    model = VectorModel(args.weights)
elif args.model == 'boolean':
    model = BooleanModel()
else:
    raise Exception('Model ' + args.model + ' does not exist')

class QRels:

    def __init__(self, path):
        self.queries_results = collections.defaultdict(list)
        with open('./{}'.format(path), 'r') as f:
            for line in f.readlines():
                query, document_id, _, _ = line.split()
                query = int(query)
                document_id = int(document_id)
                self.queries_results[query] = self.queries_results.get(query, []) + [document_id]

class PerformanceQueries:

    def __init__(self, path):
        with open('./{}'.format(path), 'r') as f:
            document = f.read()
        self.queries = {}
        self.parse_from_string(document)
 
    def parse_from_string(self, document):
        queries_list = re.split('^\.I ', document, flags=re.MULTILINE)
        for query in queries_list:
            queries_part = re.split('^\.', query, flags=re.MULTILINE)
            if len(queries_part) > 0 and queries_part[0] != '':
                query_identifier = int(queries_part[0])
                for element in queries_part:
                    if element.startswith('W'):
                        self.queries[query_identifier] = ' '.join(element.split('\n')[1:])

with open(docRetreiveFile, 'rb') as f:
    retreive_dict = pickle.load(f)
    print("loaded retreive from " + docRetreiveFile)

    inv_index = InvertedIndex([])
    inv_index.load(indexFile)
    print("loaded inverted index from " + indexFile)

    qrels = QRels('./qrels.text')
    print("loaded relations from ./qrels.text")

    queries = PerformanceQueries('./query.text')
    print("loaded queries from ./query.text")

    avgRecallsAtRank = []
    avgPrecisionsAtRank = []

    for rank in range(1,100, 2):
        results = {qid:inv_index.search(query, model, tokenizer, normalizer)[:rank]
                    for qid,query in queries.queries.items()}
        
        right_results = collections.defaultdict(list)
        for qid, found_docs in results.items():
            for found_doc in found_docs:
                if qid in qrels.queries_results:
                    if found_doc in qrels.queries_results[qid]:
                        right_results[qid].append(found_doc)

        recall = {qid: len(right_results[qid]) / len(qrels.queries_results[qid]) for qid in qrels.queries_results.keys()}
        precision = {qid: len(right_results[qid]) / len(results[qid]) for qid in results.keys()}

        # avgRecall = recall[45]
        # avgPrecision = precision[45]
        avgRecall = sum(recall.values()) / len(recall)
        avgPrecision = sum(precision.values()) / len(precision)

        print("Average recall at rank " + str(rank) + ": " + str(avgRecall))
        print("Average precision at rank " + str(rank) + ": " + str(avgPrecision))

        avgRecallsAtRank.append(avgRecall)
        avgPrecisionsAtRank.append(avgPrecision)
    avgPrecisionsAtRank = [max(avgPrecisionsAtRank[rank:]) for rank in range(len(avgPrecisionsAtRank))]

    plt.plot(avgRecallsAtRank, avgPrecisionsAtRank)
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.savefig('./recallPrecision.png')