import re

class Document:
    def __init__(self, str):
        # TODO jules
        # parser le string et creer l objet
        pass

    def __init__(self, i, t, w, k):
        self.id = i
        self.title = t
        self.summary = w
        self.keywords = k

with open('cacm.all') as f:
    full_document = f.read()
    document_list = re.split('^\.I ', full_document, flags=re.MULTILINE)
    for document in document_list[1:2]:
        doc_parts = re.split('^\.', document, flags=re.MULTILINE)
        print(doc_parts[3])
