import re


class Document:
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

    def __init__(self, i, t, w, k):
        self.id = i
        self.title = t
        self.summary = w
        self.keywords = k

with open('cacm.all') as f:
    full_document = f.read()
    document_list = re.split('^\.I ', full_document, flags=re.MULTILINE)
    for document in document_list:
        doc = Document(document)
