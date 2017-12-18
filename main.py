# coding=utf-8
import re


class DocumentTokenizer:

    def __init__(self, stop_list):
        self.stop_list = stop_list

    def tokenize(self, s, normalizer):
        for token in re.findall(r"[a-zA-Z]+", s):
            normalized_token = normalizer.normalize(token)
            if self.stop_list.valid(normalized_token):
                yield normalized_token


class DocumentNormalizer:
    @staticmethod
    def normalize(token):
        return token.lower()


class InvertedIndex:
    def __init__(self):
        self.inverted_index = {}

    def __str__(self):
        res = ""
        for (key, val) in self.inverted_index.items():
            res += key + str(val) + "\n"
        return res

    def filter(self, pattern):
        copy = InvertedIndex()
        copy.inverted_index = {key: val for (key, val) in self.inverted_index.items() if re.match(pattern, key)}
        return copy

    def register(self, token, documentId):
        self.inverted_index[token] = \
            [documentId] if token not in self.inverted_index else self.inverted_index[token] + [documentId]


class Document:
    def __init__(self):
        self.fields_to_tokenize = []
        self.id = ""

    def tokenize(self, tokenizer, normalizer, inverted_index):
        for field in self.fields_to_tokenize:
            setattr(self, field + '_tokens', [word for word in tokenizer.tokenize(getattr(self, field), normalizer)])
            for token in getattr(self, field + '_tokens'):
                inverted_index.register(token, self.id)


class CACMDocument(Document):

    def __init__(self, i, t, w, k):
        Document.__init__(self)
        self.id = i
        self.title = t
        self.summary = w
        self.keywords = k

        self.fields_to_tokenize = ["title", "summary", "keywords"]

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


class StopList():

    def __init__(self, path):
        with open('./{}'.format(path), 'r') as f:
            self.stop_list = set(f.read().split('\n'))

    def valid(self, word):
        return word not in self.stop_list


with open('cacm.all') as f:
    invIndex = InvertedIndex()
    stop_list = StopList('common_words')
    tokenizer = DocumentTokenizer(stop_list)
    normalizer = DocumentNormalizer()
    full_document = f.read()
    document_list = re.split('^\.I ', full_document, flags=re.MULTILINE)
    for document in document_list:
        doc = CACMDocument.from_string(document)
        doc.tokenize(tokenizer, normalizer, invIndex)
    print(invIndex.filter(r"the"))
