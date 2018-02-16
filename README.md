# friweb

## Description

This project is built over two main modules
 - An indexing module, enabling to index CACM or CS276 collection, using BSBI or Map-Reduce indexing methods
 - A search module, enabling to search string in CACM or CS276 collections in interactive mode or as command parameter

Different search methods available :
 - Boolean Search
 - Vectorial Search
   - Using `tf-idf` weights (tf-idf)
   - Using `tf-idf-norm` weights (normalized tf-idf)
   - Using `norm-freq` weights (normalized frequencies)

You will find more detailed results and statistics in [this document](./rapport.md)

The different services communicate each other with files containing inverted indexes and document retreival informations stored in pickle dump files.

> Naming conventions : inv_index_<collection>_<weights>.pkl
>                      doc_retreive_<collection>_<weights>.pkl

## Design choices

We have chosen to implement :
 - A generic document class implementing a tokenizing function so that extending this system to any other collection is easy
 - Generic tokenizer and normalizer classes so that they can easily be customized, independently to the collection
 - Cosine similarity as the vectorial search similarity definition : we just need to store the document vector norms, and we can build the similarities on the fly.
 - A way to easily merge, save and load inverted indexes with arbitrary stored data. The function `post_register_hook` enables to customize easily which information from built document vectors are kept and stored.
 - To store document and query vectors as sparse vectors, enabling fast dot products and search.
 - A full boolean expression parser
 
## File descriptions

 - [index.py](./index.py) : allows to index collections, more is usage section
 - [search.py](./search.py) : allows to search in built indexes (stored in files)
 - [document.py](./document.py) : contains inverted index, sparse word vectors and document definitions
 - [performance_test.py](./performance_test.py) : used to test model performances
 - [rapport.md](./rapport.md) : contains performance results
 - [query.py](./query.py) : contains boolean expression parser
 - [search_models.py](./search_models.py) : contains search models (boolean and vectorial) implementations
## Usage

To **build** the index

> python3 index.py

```
usage: index.py [-h] [-m {bsbi,map-reduce}]
                [-w {tf-idf,tf-idf-norm,norm-freq,all,none}] [-c {cacm,cs276}]

Builds inverted index given a method and a collection

optional arguments:
  -h, --help            show this help message and exit
  -m {bsbi,map-reduce}, --method {bsbi,map-reduce}
                        the method to use to build the inverted index
  -w {tf-idf,tf-idf-norm,norm-freq,all,none}, --weights {tf-idf,tf-idf-norm,norm-freq,all,none}
                        the weights to use in vector model, use none to only
                        generate the boolean model inv index
  -c {cacm,cs276}, --collection {cacm,cs276}
                        the collection to build the inverted index of
```

To **search**

> python3 search.py

```
usage: search.py [-h] [-m {vector,boolean}]
                 [-w {tf-idf,tf-idf-norm,norm-freq}]
                 [-sw {tf-idf,tf-idf-norm,norm-freq,all,none}]
                 [-c {cacm,cs276}] [-i INPUT]

Runs a search given a collection, a model, and weights

optional arguments:
  -h, --help            show this help message and exit
  -m {vector,boolean}, --model {vector,boolean}
                        the model to use for search
  -w {tf-idf,tf-idf-norm,norm-freq}, --weights {tf-idf,tf-idf-norm,norm-freq}
                        the weights to use in vector model
  -sw {tf-idf,tf-idf-norm,norm-freq,all,none}, --sourceweights {tf-idf,tf-idf-norm,norm-freq,all,none}
                        which index file to read
  -c {cacm,cs276}, --collection {cacm,cs276}
                        the collection to build the inverted index of
  -i INPUT, --input INPUT
                        a string to search
```

To **test** performances (E-F-R-measures, recall precision and MAP)

> python3 performance_test.py
