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


def n_points_interpolate(qrecalls, qprecisions, n):
    # will hold result
    qprecisions_res = [qprecisions[0]]
    qrecalls_res = [qrecalls[0]]

    # the wanted interpolation x values
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
            pt = qprecisions[i - 1] - (qprecisions[i - 1] - qprecisions[i]) * (
                qrecalls[i - 1] - thresholds[thresholdIter]) / (
                    qrecalls[i - 1] - qrecalls[i]
                ) if qrecalls[i - 1] - qrecalls[i] > 0 else qprecisions[i - 1]
            qprecisions_res.append(pt)
            qrecalls_res.append(thresholds[thresholdIter])

            thresholdIter += 1

    return qrecalls_res, qprecisions_res


def build_precision_recall(inv_index, queries, qrels, N):
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

                    # recall and precision definitions
                    recall = len(right_results) / len(
                        qrels.queries_results[qid])
                    precision = len(right_results) / len(results)

                    recalls[qid].append(recall)
                    precisions[qid].append(precision)

        # keep maximum precision for lower rank (interpolation)
        for qid, qprecisions in precisions.items():
            maximum = float("-inf")
            qprecisions_interpolated = []
            qprecisions.reverse()
            for val in qprecisions:
                maximum = max(maximum, val)
                qprecisions_interpolated.append(maximum)
            qprecisions_interpolated.reverse()
            precisions[qid] = qprecisions_interpolated

        # n points interpolation
        for qid in precisions.keys():

            qrecalls = recalls[qid]
            qprecisions = precisions[qid]

            qprecisions = [max(qprecisions)] + qprecisions + [
                len(qprecisions) / len(qrels.queries_results[qid])
            ]
            qrecalls = [0] + qrecalls + [1]

            qrecalls_res, qprecisions_res = n_points_interpolate(
                qrecalls, qprecisions, N)

            recalls_interpolated[qid] = qrecalls_res
            precisions_interpolated[qid] = qprecisions_res

        # average interpolated precisions
        avg_precisions_interpolated = []
        avg_recalls_interpolated = [i / (N) for i in range(N + 1)]
        for i in range(N + 1):
            avg_precisions_interpolated.append(
                sum([
                    precs[i] for qid, precs in precisions_interpolated.items()
                ]) / len(precisions_interpolated))

        yield model, avg_recalls_interpolated, avg_precisions_interpolated


def build_f_measure(recalls, precisions, beta):
    f_measures = []
    for i in range(len(recalls)):
        p = precisions[i]
        r = recalls[i]
        f_measures.append((1 + beta**2) * (p * r) / ((beta**2) * (p + r)))
    return recalls, f_measures


def build_e_measure(recalls, precisions, beta):
    recalls, f_measures = build_f_measure(recalls, precisions, beta)
    e_measures = list(map(lambda x: 1 - x, f_measures))
    return recalls, e_measures


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

    recall_precision_curves = [(model, recall, precision) for (
        model, recall,
        precision) in build_precision_recall(inv_index, queries, qrels, 20)]

    plt.figure(0)
    for model, avg_recalls_interpolated, avg_precisions_interpolated in recall_precision_curves:
        plt.plot(
            avg_recalls_interpolated,
            avg_precisions_interpolated,
            label=model.method)
    plt.title("recall precision")
    plt.legend()
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.savefig('./recallPrecision.png')

    print('Recall precision curves drawn in ./recall-precision.png')

    plt.figure(1)
    for model, avg_recalls_interpolated, avg_precisions_interpolated in recall_precision_curves:
        avg_recalls_interpolated, avg_f_measure_interpolated = build_f_measure(
            avg_recalls_interpolated, avg_precisions_interpolated, 1)
        plt.plot(
            avg_recalls_interpolated,
            avg_f_measure_interpolated,
            label=model.method)
    plt.title("recall f-measure")
    plt.legend()
    plt.xlabel('recall')
    plt.ylabel('f-measure')
    plt.savefig('./f-measure.png')

    print('F measure curve drawn in ./f-measure.png')

    plt.figure(2)
    for model, avg_recalls_interpolated, avg_precisions_interpolated in recall_precision_curves:
        avg_recalls_interpolated, avg_e_measure_interpolated = build_e_measure(
            avg_recalls_interpolated, avg_precisions_interpolated, 1)
        plt.plot(
            avg_recalls_interpolated,
            avg_e_measure_interpolated,
            label=model.method)
    plt.title("recall e-measure")
    plt.legend()
    plt.xlabel('recall')
    plt.ylabel('e-measure')
    plt.savefig('./e-measure.png')

    print('E measure curve drawn in ./e-measure.png')