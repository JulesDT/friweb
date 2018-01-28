class QRels:

    def __init__(self, path):
        self.queries_results = {}
        with open('./{}'.format(path), 'r') as f:
            for line in f.readlines():
                query, document_id, _, _ = line.split()
                self.queries_results[query] = self.queries_results.get(query, []) + [document_id]
