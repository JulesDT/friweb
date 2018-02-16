import math, collections
from functools import reduce
from documents import DocumentNormalizer, DocumentTokenizer, StopList, InvertedIndex, CASMBlock, CS276Block

def generate_indexes(data, split_ratio=1):
    stop_list = StopList('common_words')
    tokenizer = DocumentTokenizer(stop_list)
    normalizer = DocumentNormalizer()
    retrieval_list = []
    if data == 'cacm':
        cs_block = CASMBlock('cacm.all')
    elif data == 'cs276':
        cs_block = CS276Block('./pa1-data/*')
    else:
        raise Exception('Collection ' + data + ' not supported')
    ''' 
    Map
    '''
    document_list = set()
    for block in cs_block.get_next_block():
        document_list.update(block)
        doc_retrieval_block = {}
        for document in block:
            doc_retrieval_block[document.id] = document.entry_string()
        retrieval_list.append(doc_retrieval_block)


    # Half it !
    document_list = set(list(document_list)[:int(len(document_list)/split_ratio)])
    '''
    Map
    '''
    def map_tokenize(doc):
        return [
                    (word, doc.id, 1)
                        for field in doc.fields_to_tokenize
                            for word in tokenizer.tokenize(getattr(doc, field), normalizer)
                ]
    mapped_data = map(map_tokenize, document_list)
    mapped_data = [item for sublist in mapped_data for item in sublist]
    '''
    Shuffle
    '''
    shuffled_data = collections.defaultdict(list)
    for word, doc_id, value in mapped_data:
        shuffled_data[word].append((doc_id, value))
    '''
    Reduce
    '''
    def reducer(reduced_data, new_entry):
        for entry in new_entry[1]:
            reduced_data[new_entry[0]][entry[0]] += 1
        return reduced_data
    inverted_index = reduce(reducer, shuffled_data.items(), collections.defaultdict(lambda: collections.defaultdict(int)))
    inv_index = InvertedIndex([])
    inv_index.inverted_index = inverted_index

    return inv_index

def heap_law(inverted_index, half_inverted_index):
    T1 = sum([inverted_index[word][doc_id] for word in inverted_index for doc_id in inverted_index[word]])
    T2 = sum([half_inverted_index[word][doc_id] for word in half_inverted_index for doc_id in half_inverted_index[word]])

    M1 = len(inverted_index)
    M2 = len(half_inverted_index)

    b = math.log(M2/M1)/math.log(T2/T1)
    k = (M2 - M1)/(T2**b - T1**b)
    
    return b, k

def frequency_rank(inverted_index):
    numbers = {
        word: sum([inverted_index[word][doc_id] for doc_id in inverted_index[word]])
            for word in inverted_index
    }
    total_number_of_tokens = sum(numbers.values())
    sorted_values = sorted(numbers.items(), key=lambda x: x[1], reverse=True)
    return [(word, value/total_number_of_tokens) for word, value in sorted_values]


print("CACM")
inv_index = generate_indexes('cacm')
half_inv_index = generate_indexes('cacm', split_ratio=2)
b, k = heap_law(inv_index.inverted_index, half_inv_index.inverted_index)
print("Tokens: Full -> {}, 1/2 -> {}".format(
    sum([inv_index.inverted_index[word][doc_id] for word in inv_index.inverted_index for doc_id in inv_index.inverted_index[word]]),
    sum([half_inv_index.inverted_index[word][doc_id] for word in half_inv_index.inverted_index for doc_id in half_inv_index.inverted_index[word]])
))
print("Vocabulary: Full -> {}, 1/2 -> {}".format(
    len(inv_index.inverted_index),
    len(half_inv_index.inverted_index)
))
print("b: {}, k: {}".format(b, k))

print("cs276")
inv_index = generate_indexes('cs276')
half_inv_index = generate_indexes('cs276', split_ratio=2)
b, k = heap_law(inv_index.inverted_index, half_inv_index.inverted_index)
print("Tokens: Full -> {}, 1/2 -> {}".format(
    sum([inv_index.inverted_index[word][doc_id] for word in inv_index.inverted_index for doc_id in inv_index.inverted_index[word]]),
    sum([half_inv_index.inverted_index[word][doc_id] for word in half_inv_index.inverted_index for doc_id in half_inv_index.inverted_index[word]])
))
print("Vocabulary: Full -> {}, 1/2 -> {}".format(
    len(inv_index.inverted_index),
    len(half_inv_index.inverted_index)
))
print("b: {}, k: {}".format(b, k))