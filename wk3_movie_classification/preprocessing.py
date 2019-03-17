import os
import pandas as pd
import itertools
import gluonnlp as nlp
from sklearn.model_selection import train_test_split
import pickle
from data_load import movie_data
from konlpy.tag import Mecab

mecab = Mecab()

# load dataset
dataset = movie_data('ratings.txt')
data = dataset.data[['document', 'label']] # document 컬럼과 label 컬럼만 남김
data = data[~data.document.isna()] # NA 제거

documents = list(data.document)
labels = list(data.label)

# tokenize
documents_tokenized = [ mecab.morphs(d) for d in documents]

# build vocab
counter = nlp.data.count_tokens(itertools.chain.from_iterable(documents_tokenized))
vocab = nlp.Vocab(counter=counter, min_freq=10)

# token to index
print(vocab.token_to_idx)

