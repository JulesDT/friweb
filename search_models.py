import math
import collections
from documents import SparseWordVector
from query import Tree

class VectorModel:
    def __init__(self):
        pass

    def search(self, str, inv_index):
        str.replace(r"[\n\s]+", " ")
        tokens = str.split(' ')

        print(tokens)

        query_vector = SparseWordVector()
        counter = collections.Counter(tokens)
        for token, amount in counter.items():
            if token in inv_index.inverted_index:
                idf = math.log10(len(inv_index.inverted_index) / len(inv_index.inverted_index[token]))
                query_vector[token] = (1 + math.log10(amount)) * idf

        filtered_tf_idf = {
            token: inv_index.tf_idf[token]
            for token in tokens
            if token in inv_index.tf_idf
        }

        doc_vectors = collections.defaultdict(SparseWordVector)

        for term, posting in filtered_tf_idf.items():
            for doc_id, tf_idf in posting.items():
                doc_vectors[doc_id][term] = tf_idf

        similarities = {doc_id: doc_vector.cosSilimarity(query_vector) for doc_id, doc_vector in doc_vectors.items()}
        sorted_doc_ids = sorted(similarities, key=lambda k:similarities[k], reverse=True)
        print(sorted_doc_ids)
        return sorted_doc_ids

class BooleanModel:
    def __init__(self):
        pass

    def search(self, str, inv_index):
        str.replace(r"[\n\s]+", " ")
        tokens = str.split(' ')

        print(" & ".join(tokens))

        tree = Tree(parent=None)
        Tree.parse(tree, " & ".join(tokens))
        result = tree.query(inv_index)
        # from IPython import embed
        # embed()
        sorted_doc_ids = sorted(result)
        return sorted_doc_ids