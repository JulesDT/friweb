import argparse

parser = argparse.ArgumentParser(
    description='Builds inverted index given a method and a collection')

parser.add_argument(
    '-m',
    '--method',
    choices=['tf-idf', 'tf-idf-norm', ''],
    default='tf-idf',
    help='the method to use to build the inverted index')

args = parser.parse_args()

print(args)

