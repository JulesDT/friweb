import math
import collections
import re
from documents import SparseWordVector
from query import Tree

class VectorModel:
    def __init__(self, method):
        self.method = method

    def search(self, input, inv_index, tokenizer, normalizer):

        if self.method == 'tf-idf':
            doc_norms = inv_index.doc_norms_tf_idf
        elif self.method == 'tf-idf-norm':
            doc_norms = inv_index.doc_norms_tf_idf_norm
        elif self.method == 'norm-freq':            
            doc_norms = inv_index.doc_norms_norm_freq
            doc_most_frequent = inv_index.doc_most_frequent
        else:
            raise Exception("VectorModel search does not handle `" + inv_index.method + "` method")

        if len(doc_norms) == 0:
            raise Exception("Can not use method " + self.method + " as it is not present in input file")

        # let us build the query vector
        gen = tokenizer.tokenize(input, normalizer)
        tokens = [token for token in gen]

        query_vector = SparseWordVector()
        counter = collections.Counter(tokens)
        for token, amount in counter.items():
            if token in inv_index.inverted_index:
                idf = math.log10(len(inv_index.inverted_index) / len(inv_index.inverted_index[token]))
                if self.method == 'tf-idf':
                    query_vector.v[token] = amount * idf
                elif self.method == 'tf-idf-norm':
                    query_vector.v[token] = (1 + math.log10(amount)) * idf
                elif self.method == 'norm-freq':
                    query_vector.v[token] = amount / max(counter.values())

        # then build the document vectors
        # as we use cosine similarity, we dont have to build up the whole document vector
        # just build the doc vector on the word dimensions of the query and manually set its norm

        # so we filter out the right part of the wdt

        document_vectors = collections.defaultdict(SparseWordVector)
        for term in query_vector.v.keys():
            if term in inv_index.inverted_index:
                postings = inv_index.inverted_index[term]
                idf = math.log10(len(inv_index.inverted_index) / len(postings))
                for doc_id, raw_tf in postings.items():
                    if self.method == 'tf-idf':
                        document_vectors[doc_id].v[term] = raw_tf * idf
                    elif self.method == 'tf-idf-norm':
                        document_vectors[doc_id].v[term] = (1 + math.log10(raw_tf)) * idf
                    elif self.method == 'norm-freq':
                        document_vectors[doc_id].v[term] = raw_tf / doc_most_frequent[doc_id]

        for doc_id, document_vector in document_vectors.items():
            document_vector.setCustomNorm(math.sqrt(doc_norms[doc_id]))

        # then let us build a cos similarity result and order it by maximum similarity

        similarities = {doc_id: doc_vector.cosSimilarityCallerDims(query_vector) for doc_id, doc_vector in document_vectors.items()}
        sorted_doc_ids = sorted(similarities, key=lambda k:similarities[k], reverse=True)

        # for id in sorted_doc_ids[:10]:
        #     print("#######")
        #     print("id: " + str(id))
        #     print("sim: " + str(similarities[id]))
        #     cmon = set(query_vector.v.keys()).intersection(set(document_vectors[id].v.keys()))
        #     print("cmon: " + str(cmon))
        #     print("query: " + str(query_vector.v.keys()))
        #     print("document: " + str(document_vectors[id].v.keys()))

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