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


with open('cacm.all') as f:
    invIndex = InvertedIndex()
    tokenizer = DocumentTokenizer()
    full_document = f.read()
    document_list = re.split('^\.I ', full_document, flags=re.MULTILINE)
    for document in document_list:
        doc = CACMDocument.from_string(document)
        doc.tokenize(tokenizer, invIndex)
    print(invIndex)

