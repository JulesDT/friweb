import re


class DocumentTokenizer:
    @staticmethod
    def tokenize(str):
        return re.findall(r"\w+", str)


class InvertedIndex:
    def __init__(self):
        self.inverted_index = {}

    def register(self, token, documentId):
        self.inverted_index[token] = \
            [documentId] if token not in self.inverted_index else self.inverted_index[token] + [documentId]


class Document:

    def __init__(self):
        self.fields_to_tokenize = []
        self.id = ""

    def tokenize(self, tokenizer, inverted_index):
        for field in self.fields_to_tokenize:
            setattr(self, field + '_tokens', tokenizer.tokenize(getattr(self, field)))
            for token in getattr(self, field + '_tokens'):
                inverted_index.register(token, self.id)

class CACMDocument(Document):

    # TODO find a clean way to do so

    def __init__(self, i, t, w, k):
        self.id = i
        self.title = t
        self.summary = w
        self.keywords = k

        self.fields_to_tokenize = ["title", "summary", "keywords"]

    @classmethod
    def from_string(self, document):
        doc_parts = re.split('^\.', document, flags=re.MULTILINE)
        if len(doc_parts) > 0 and doc_parts[0] != '':
            identifier = int(doc_parts[0])
            for element in doc_parts:
                if element.startswith('T'):
                    title = element.split('\n')[1]
                elif element.startswith('W'):
                    summary = element.split('\n')[1]
                elif element.startswith('K'):
                    keywords = element.split('\n')[1]
        return Document(identifier, title, summary, keywords)


with open('cacm.all') as f:
    full_document = f.read()
    document_list = re.split('^\.I ', full_document, flags=re.MULTILINE)
    for document in document_list:
        doc = Document(document)
