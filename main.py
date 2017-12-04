import re

with open('cacm.all') as f:
    full_document = f.read()
    document_list = re.split('^\.I ', full_document, flags=re.MULTILINE)
    for document in document_list[1:2]:
        doc_parts = re.split('^\.', document, flags=re.MULTILINE)
        print(doc_parts[3])
