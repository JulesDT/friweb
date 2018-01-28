import re
import glob
import collections
import math
import pickle

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
            normalized_token = normalizer.normalize(token)
            if self.stop_list.valid(normalized_token):
                yield normalized_token


class DocumentNormalizer:
    @staticmethod
    def normalize(token):
        return token.lower()


class InvertedIndex:
    def __init__(self, methods):
        self.methods = methods
        self.inverted_index = collections.defaultdict(lambda: collections.defaultdict(int))
        self.doc_lengths = collections.defaultdict(int)
        self.tf_idf = collections.defaultdict(lambda: collections.defaultdict(int))
        self.tf_idf_norm = collections.defaultdict(lambda: collections.defaultdict(int))
        self.norm_freq = collections.defaultdict(lambda: collections.defaultdict(int))

    def __str__(self):
        res = ""
        for (key, val) in self.inverted_index.items():
            res += key + str(val) + "\n"
        return res

    def filter(self, pattern, strict=False):
        copy = InvertedIndex()
        if strict:
            copy.inverted_index = {pattern: self.inverted_index.get(pattern, {})}
        else:
            copy.inverted_index = {key: val for (key, val) in self.inverted_index.items() if pattern in key}
        return copy

    def register(self, token, documentId):
        self.inverted_index[token][documentId] += 1
        self.doc_lengths[documentId] += 1

    def post_register_hook(self, token, documentId):
        for method in self.methods:
            if method == 'tf-idf':
                self.build_tf_idf()
            elif method == 'tf-idf-norm':
                self.build_tf_idf_norm()
            elif method == 'norm-freq':
                self.build_norm_freq()

    def merge(self, inv_index):
        for token in inv_index.inverted_index.keys():
            self.inverted_index[token].update(inv_index.inverted_index[token])
        for method in self.methods:
            if method == 'tf_idf':
                for token in inv_index.tf_idf.keys():
                    self.tf_idf[token].update(inv_index.tf_idf[token])
            elif method == 'tf-idf-norm':
                for token in inv_index.tf_idf_norm.keys():
                    self.tf_idf_norm[token].update(inv_index.tf_idf_norm[token])
            elif method == 'norm-freq':
                for token in inv_index.norm_freq.keys():
                    self.norm_freq[token].update(inv_index.norm_freq[token])

    def build_tf_idf(self):
        for (term, termPostings) in self.inverted_index.items():
            idf = math.log10(len(self.inverted_index) / len(termPostings))
            self.tf_idf[term] = {
                docId: (1 + math.log10(amount))*idf
                for (docId, amount) in termPostings.items()
            }

    def build_tf_idf_norm(self):
        for (term, termPostings) in self.inverted_index.items():
            idf = math.log10(len(self.inverted_index) / len(termPostings))
            self.tf_idf_norm[term] = {
                docId: (1 + math.log10(amount))*idf
                for (docId, amount) in termPostings.items()
            }

    def build_norm_freq(self):
        # let us basically invert the inverted index ><
        doc_to_word_idx = collections.defaultdict(lambda: collections.defaultdict(int))
        for term, postings in self.inverted_index.items():
            for doc_id, amt in postings.items():
                doc_to_word_idx[doc_id][term] += amt
        # and map this to a dict getting the most frequent term for every document

        most_frequent = {}
        for doc_id, words in doc_to_word_idx.items():
            most_frequent[doc_id] = max(words.values())

        for (term, termPostings) in self.inverted_index.items():
            self.norm_freq[term] = {
                docId: amt / most_frequent[docId]
                for (docId, amt) in termPostings.items()
            }

    def search(self, string, model, tokenizer, normalizer):
        return model.search(string, self, tokenizer, normalizer)

    def save(self, path):
        with open(path, 'wb') as f:
            toDump = {
                "inverted_index" : self.inverted_index,
                "tf_idf" : self.tf_idf
            }
            pickle.dump(toDump, f, pickle.HIGHEST_PROTOCOL)

    def load(self, path):
        with open(path, 'rb') as f:
            loaded = pickle.load(f)
            self.tf_idf = loaded["tf_idf"]
            self.inverted_index = loaded["inverted_index"]

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
