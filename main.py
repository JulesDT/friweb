import re


class DocumentTokenizer:
    @staticmethod
    def tokenize(str):
        return re.findall(r"\w+", str)


class InvertedIndex:
    def __init__(self):
        self.invertedIndex = {}

    def incr(self, token):
        self.invertedIndex[token] = 1 if token not in self.invertedIndex else self.invertedIndex[token] + 1


class Document:

    def __init__(self):
        self.fieldsToTokenize = []
        self.id = ""

    def tokenize(self, tokenizer, invertedIndex):
        for field in self.fieldsToTokenize:
            setattr(self, field + '_tokens', tokenizer.tokenize(getattr(self, field)))
            inve

class CACMDocument(Document):

    # TODO find a clean way to do so

    def __init__(self, i, t, w, k):
        self.id = i
        self.title = t
        self.summary = w
        self.keywords = k

        self.fieldsToTokenize = ["title", "summary", "keywords"]


    def __init__(self, document):
        self.summary = ''
        self.keywords = ''
        self.title = ''
        self.id = 0
        doc_parts = re.split('^\.', document, flags=re.MULTILINE)
        if len(doc_parts) > 0 and doc_parts[0] != '':
            self.id = int(doc_parts[0])
            for element in doc_parts:
                if element.startswith('T'):
                    self.title = element.split('\n')[1]
                elif element.startswith('W'):
                    self.summary = element.split('\n')[1]
                elif element.startswith('K'):
                    self.keywords = element.split('\n')[1]


with open('cacm.all') as f:
    full_document = f.read()
    document_list = re.split('^\.I ', full_document, flags=re.MULTILINE)
    for document in document_list:
        doc = Document(document)
