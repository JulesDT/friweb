import re
import glob
import math
import collections

class VectorModel:
    def __init__(self):
        pass

    def search(self, str, inv_index):
        str.replace(r"[\n\s]+", " ")
        tokens = str.split(' ')

        print(tokens)

        query_vector = SparseWordVector()
        counter = collections.Counter(tokens)
        for token, amount in counter.items():
            if token in inv_index.inverted_index:
                idf = math.log10(len(inv_index.inverted_index) / len(inv_index.inverted_index[token]))
                query_vector[token] = (1 + math.log10(amount)) * idf

        filtered_tf_idf = {
            token: inv_index.tf_idf[token]
            for token in tokens
            if token in inv_index.tf_idf
        }

        doc_vectors = collections.defaultdict(SparseWordVector)

        for term, posting in filtered_tf_idf.items():
            for doc_id, tf_idf in posting.items():
                doc_vectors[doc_id][term] = tf_idf

        similarities = {doc_id: doc_vector.cosSilimarity(query_vector) for doc_id, doc_vector in doc_vectors.items()}
        sorted_doc_ids = sorted(similarities, key=lambda k:similarities[k], reverse=True)

        return sorted_doc_ids

class SparseWordVector:
    def __init__(self, v = {}):
        self.v = v
        self.resetCache()

    def __setitem__(self, k, v):
        self.v[k] = v

    def resetCache(self):
        self._norm = -1

    def norm(self):
        if self._norm == -1:
            self._norm = sum([v * v for k, v in self.v.items()])
        return self._norm

    def cosSilimarity(self, other):
        v1 = self.v
        v2 = other.v
        v1_dims = set(v1.keys())
        v2_dims = set(v2.keys())
        common_dims = v1_dims.intersection(v2_dims)
        num = sum([v1[dim] * v2[dim] for dim in common_dims])
        return num / (self.norm() * other.norm())


class DocumentTokenizer:
    def __init__(self, stop_list):
        self.stop_list = stop_list

    def tokenize(self, s, normalizer):
        reg = re.compile(r"[a-zA-Z]+")
        for token in reg.findall(s):
            normalized_token = token
            if self.stop_list.valid(normalized_token):
                yield normalized_token


class DocumentNormalizer:
    @staticmethod
    def normalize(token):
        return token.lower()


class InvertedIndex:
    def __init__(self):
        self.inverted_index = collections.defaultdict(lambda: collections.defaultdict(int))
        self.base_dict = {}
        self.doc_ids = set()
        self.doc_vectors = {}
        self.tf_idf = {}

    def __str__(self):
        res = ""
        for (key, val) in self.inverted_index.items():
            res += key + str(val) + "\n"
        return res

    def filter(self, pattern, strict=False):
        copy = InvertedIndex()
        if strict:
            copy.inverted_index = {key: val for (key, val) in self.inverted_index.items() if pattern in key}
        else:
            copy.inverted_index = {pattern: self.inverted_index.get(pattern, set([]))}
        return copy

    def register(self, token, documentId):
        self.inverted_index[token][documentId] += 1

    def merge(self, inv_index):
        for token in inv_index.inverted_index.keys():
            if token in self.inverted_index:
                self.inverted_index[token].update(inv_index.inverted_index[token])
            else:
                self.inverted_index[token] = inv_index.inverted_index[token]

    def build_tf_idf(self):
        for (term, termPostings) in self.inverted_index.items():
            idf = math.log10(len(self.inverted_index) / len(termPostings))
            self.tf_idf[term] = {
                docId: (1 + math.log10(amount))*idf
                for (docId, amount) in termPostings.items()
            }

    def build_base_vector(self):
        self.base_dict = {k: v for v, k in enumerate(self.inverted_index.keys())}
        print(self.base_dict)

    def intersect(self, second_inv_index):
        copy = InvertedIndex()
        copy.inverted_index = {key: self.inverted_index[key]
                               for key in (set(self.inverted_index.keys()) & set(second_inv_index.inverted_index.keys()))}
        return copy

    def union(self, second_inv_index):
        copy = InvertedIndex()
        copy.inverted_index = {**self.inverted_index, **second_inv_index.inverted_index}
        return copy

    def not_operator(self, global_inv_index):
        copy = InvertedIndex()
        copy.inverted_index = {key: global_inv_index.inverted_index[key]
                               for key in (set(global_inv_index.inverted_index.keys()) - set(self.inverted_index.keys()))}
        return copy

    def search(self, string, model):
        return model.search(string, self)

class Document:
    def __init__(self):
        self.fields_to_tokenize = []
        self.id = ""

    def tokenize(self, tokenizer, normalizer, inverted_index):
        for field in self.fields_to_tokenize:
            setattr(self, field + '_tokens', [word for word in tokenizer.tokenize(getattr(self, field), normalizer)])
            for token in getattr(self, field + '_tokens'):
                inverted_index.register(token, self.id)


class CASMBlock:
    def __init__(self, path):
        self.path = path

    def get_next_block(self):
        doc_list = set()
        with open(self.path) as f:
            full_document = f.read()
            document_list = re.split('^\.I ', full_document, flags=re.MULTILINE)
            for document in document_list:
                doc = CACMDocument.from_string(document)
                doc_list.add(doc)
            yield doc_list


class CS276Block:
    def __init__(self, path):
        self.path = path

    def get_next_block(self):
        i = 0
        for filename in glob.glob(self.path):
            doc_list = set()
            print('Reading ' + filename)
            for documentFileName in glob.glob(filename + '/*'):
                with open(documentFileName) as f:
                    document = f.read()
                    doc = CS276Document(document, i)
                    doc_list.add(doc)
                    i += 1
            yield doc_list


class CACMDocument(Document):
    def __init__(self, i, t, w, k):
        Document.__init__(self)
        self.id = i
        self.title = t
        self.summary = w
        self.keywords = k

        self.fields_to_tokenize = ["title", "summary", "keywords"]

    def entry_string(self):
        return self.title + '\n\n' + self.summary[:500] 

    @classmethod
    def from_string(self, document):
        doc_parts = re.split('^\.', document, flags=re.MULTILINE)
        summary = ""
        keywords = ""
        title = ""
        identifier = 0
        if len(doc_parts) > 0 and doc_parts[0] != '':
            identifier = int(doc_parts[0])
            for element in doc_parts:
                if element.startswith('T'):
                    title = element.split('\n')[1]
                elif element.startswith('W'):
                    summary = element.split('\n')[1]
                elif element.startswith('K'):
                    keywords = element.split('\n')[1]
        return CACMDocument(identifier, title, summary, keywords)


class CS276Document(Document):
    def __init__(self, content, id):
        Document.__init__(self)
        self.content = content
        self.id = id

        self.fields_to_tokenize = ["content"]

    def entry_string(self):
        return self.content[:500] 

class StopList():
    def __init__(self, path):
        with open('./{}'.format(path), 'r') as f:
            self.stop_list = set(f.read().split('\n'))

    def valid(self, word):
        return word not in self.stop_list
