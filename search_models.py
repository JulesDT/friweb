import math
import collections
import re
from documents import SparseWordVector
from query import Tree

class VectorModel:
    def __init__(self, method):
        self.method = method

    def search(self, str, inv_index, tokenizer, normalizer):
        # let us first define which w_d,t to use depending on method
        if self.method == 'tf-idf':
            wdt = inv_index.tf_idf
        elif self.method == 'tf-idf-norm':
            wdt = inv_index.td_idf_norm
        elif self.method == 'norm-freq':
            wdt = inv_index.norm_freq
        else:
            raise Exception("VectorModel search does not handle `" + inv_index.method + "` method")

        if len(wdt) == 0:
            raise Exception("Can not use method " + self.method + " as it is not present in input file")

        # let us build the query vector
        gen = tokenizer.tokenize(str, normalizer)
        tokens = [token for token in gen]

        query_vector = SparseWordVector()
        counter = collections.Counter(tokens)
        for token, amount in counter.items():
            if token in wdt:
                idf = math.log10(len(wdt) / len(wdt[token]))
                query_vector[token] = (1 + math.log10(amount)) * idf

        # then build the document vectors
        # as we use cosine similarity, we dont have to build up the whole document vector
        # just build the doc vector on the word dimensions of the query

        # so we filter out the right part of the wdt

        filtered_wdt = {
            token: wdt[token]
            for token in tokens
            if token in wdt
        }

        # and we build up the document vectors
        doc_vectors = collections.defaultdict(SparseWordVector)

        for term, posting in filtered_wdt.items():
            for doc_id, tf_idf in posting.items():
                doc_vectors[doc_id][term] = tf_idf

        # then let us build a cos similarity result and order it by maximum similarity

        similarities = {doc_id: doc_vector.cosSilimarity(query_vector) for doc_id, doc_vector in doc_vectors.items()}
        sorted_doc_ids = sorted(similarities, key=lambda k:similarities[k], reverse=True)

        return sorted_doc_ids

class BooleanModel:
    def __init__(self):
        pass

    def search(self, search_string, inv_index, tokenizer, normalizer):
        search_string.replace(r"[\n\s]+", " ")
        tree = Tree(parent=None)
        Tree.parse(tree, search_string)
        result = tree.query(inv_index, tokenizer, normalizer)
        # from IPython import embed
        # embed()
        sorted_doc_ids = sorted(result)
        return sorted_doc_ids