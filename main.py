# coding=utf-8
import re
import glob


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
        if token in self.inverted_index:
            self.inverted_index[token].add(documentId)
        else:
            self.inverted_index[token] = set([documentId])

    def merge(self, inv_index):
        for token in inv_index.inverted_index.keys():
            if token in self.inverted_index:
                self.inverted_index.update(inv_index.inverted_index[token])
            else:
                self.inverted_index = inv_index.inverted_index[token]


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


class CS276Document(Document):
    def __init__(self, content, id):
        Document.__init__(self)
        self.content = content
        self.id = id

        self.fields_to_tokenize = ["content"]


class StopList():
    def __init__(self, path):
        with open('./{}'.format(path), 'r') as f:
            self.stop_list = set(f.read().split('\n'))

    def valid(self, word):
        return word not in self.stop_list


# with open('cacm.all') as f:
#     invIndex = InvertedIndex()
#     stop_list = StopList('common_words')
#     tokenizer = DocumentTokenizer(stop_list)
#     normalizer = DocumentNormalizer()
#     full_document = f.read()
#     document_list = re.split('^\.I ', full_document, flags=re.MULTILINE)
#     for document in document_list:
#         doc = CACMDocument.from_string(document)
#         doc.tokenize(tokenizer, normalizer, invIndex)
#     print(invIndex.filter(r"the"))

stop_list = StopList('common_words')
tokenizer = DocumentTokenizer(stop_list)
normalizer = DocumentNormalizer()
document_list = []
i = 0
invindex_list = []
for filename in glob.glob('./pa1-data/*'):
    invIndex = InvertedIndex()
    invindex_list.append(invIndex)
    print('Reading ' + filename)
    for documentFileName in glob.glob(filename + '/*'):
        with open(documentFileName) as f:
            document = f.read()
            doc = CS276Document(document, i)
            doc.tokenize(tokenizer, normalizer, invIndex)
            i += 1
for inv_index in invindex_list[1:]:
    invindex_list[0].merge(inv_index)
print(invIndex.filter(r"inter"))