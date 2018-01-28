import argparse
import re


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


class QRels:

    def __init__(self, path):
        self.queries_results = {}
        with open('./{}'.format(path), 'r') as f:
            for line in f.readlines():
                query, document_id, _, _ = line.split()
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
                        self.queries[query_identifier] = element.split('\n')[1]

with open(docRetreiveFile, 'rb') as f:
    retreive_dict = pickle.load(f)
    print("loaded retreive from " + docRetreiveFile)

    inv_index = InvertedIndex([])
    inv_index.load(indexFile)
    print("loaded inverted index from " + indexFile)

    # TODO