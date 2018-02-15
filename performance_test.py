import argparse
import re
import pickle
import collections
import math
import matplotlib.pyplot as plt
from timeit import timeit

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
    help='the model to use for search')

parser.add_argument(
    '-w',
    '--weights',
    choices=['tf-idf', 'tf-idf-norm', 'norm-freq', 'all'],
    default='all',
    help='the weights to use in vector model')

parser.add_argument(
    '-sw',
    '--sourceweights',
    choices=['tf-idf', 'tf-idf-norm', 'norm-freq', 'all', 'none'],
    default='all',
    help='which index file to read')

args = parser.parse_args()

indexFile = './inv_index_cacm_' + args.sourceweights + '.pkl'
docRetreiveFile = './doc_retreive_cacm_' + args.sourceweights + '.pkl'

stop_list = StopList('common_words')
tokenizer = DocumentTokenizer(stop_list)
normalizer = DocumentNormalizer()

if args.model == 'vector':
    if args.weights == 'all':
        models = [
            VectorModel('tf-idf'),
            VectorModel('tf-idf-norm'),
            VectorModel('norm-freq')
        ]
    else:
        models = [VectorModel(args.weights)]
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
                self.queries_results[query] = self.queries_results.get(
                    query, []) + [document_id]


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
                        self.queries[query_identifier] = ' '.join(
                            element.split('\n')[1:])


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

    # plot a curve for each model
    for model in models:
        N = 20

        # stores raw recalls and precision query_id -> list
        recalls = {}
        precisions = {}

        # stores interpolated recalls an precisions query_id -> list
        recalls_interpolated = {}
        precisions_interpolated = {}

        # run all queries first
        @timeit
        def run_queries():
            return {
                qid: inv_index.search(query, model, tokenizer, normalizer)
                for qid, query in queries.queries.items()
            }

        all_results, elapsed_time = run_queries()

        print(
            str(len(all_results)) + " queries ran in " + str(elapsed_time) +
            "ms using weights " + model.method)

        # build raw recall and precision
        for qid, ar in all_results.items():
            if len(qrels.queries_results[qid]) > 0:
                recalls[qid] = []
                precisions[qid] = []
                # limit the rank to max 200
                for rank in range(1, min(len(ar), 200)):
                    results = ar[:rank]
                    right_results = [
                        result for result in results
                        if result in qrels.queries_results[qid]
                    ]

                    recall = len(right_results) / len(
                        qrels.queries_results[qid])
                    precision = len(right_results) / len(results)

                    recalls[qid].append(recall)
                    precisions[qid].append(precision)

        # keep maximum precision for lower rank
        for qid, qprecisions in precisions.items():
            maximum = float("-inf")
            qprecisions_interpolated = []
            qprecisions.reverse()
            for val in qprecisions:
                maximum = max(maximum, val)
                qprecisions_interpolated.append(maximum)
            qprecisions_interpolated.reverse()
            precisions[qid] = qprecisions_interpolated

        # n points interpolate
        for qid in precisions.keys():

            def n_points_interpolate(qrecalls, qprecisions, n):
                qprecisions = [max(qprecisions)] + qprecisions + [
                    len(qprecisions) / len(qrels.queries_results[qid])
                ]
                qrecalls = [0] + qrecalls + [1]
                qprecisions_res = [qprecisions[0]]
                qrecalls_res = [qrecalls[0]]
                thresholds = [(i + 1) / n for i in range(n)]
                thresholdIter = 0
                for i, recall in enumerate(qrecalls):
                    if thresholdIter >= len(thresholds):
                        break

                    # iterate until we reach the required threshold
                    if recall < thresholds[thresholdIter]:
                        continue
                    # once it is reached fill thresholds until the recall value is not overshot
                    while thresholdIter < len(thresholds) \
                        and not thresholds[thresholdIter] > qrecalls[i]:               

                        # can handle affine interpolation
                        pt = qprecisions[i - 1] - (
                            qprecisions[i - 1] - qprecisions[i]
                        ) * (qrecalls[i - 1] - thresholds[thresholdIter]) / (
                            qrecalls[i - 1] - qrecalls[i]
                        ) if qrecalls[i - 1] - qrecalls[i] > 0 else qprecisions[
                            i - 1]
                        qprecisions_res.append(pt)
                        qrecalls_res.append(thresholds[thresholdIter])
                        
                        thresholdIter += 1



                return qrecalls_res, qprecisions_res

            qrecalls_res, qprecisions_res = n_points_interpolate(
                recalls[qid], precisions[qid], N)
            recalls_interpolated[qid] = qrecalls_res
            precisions_interpolated[qid] = qprecisions_res

        # average interpolated precisions
        avg_precisions_interpolated = []
        avg_recalls_interpolated = [i/(N) for i in range(N + 1)]
        for i in range(N+1):
            avg_precisions_interpolated.append(sum([precs[i] for qid, precs in precisions_interpolated.items()]) / len(precisions_interpolated))

        print(avg_recalls_interpolated)
        print(avg_precisions_interpolated)

        plt.plot(avg_recalls_interpolated, avg_precisions_interpolated, label=model.method)

    plt.legend()
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.savefig('./recallPrecision.png')
    print('Recall precision curves drawn in ./recallPrecision.png')