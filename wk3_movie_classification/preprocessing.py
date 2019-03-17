import os
import pandas as pd
import itertools
import gluonnlp as nlp
from sklearn.model_selection import train_test_split
import pickle
from model.data_load import movie_data
from konlpy.tag import Mecab

mecab = Mecab()

# load dataset
dataset = movie_data('ratings.txt')
data = dataset.data[['document', 'label']] # document 컬럼과 label 컬럼만 남김
data = data[~data.document.isna()] # NA 제거

# train / test split
tr_data, tst_data = train_test_split(data , test_size=0.2)

### 아래부터는 training dataset만 사용함
tr_documents = list(tr_data.document)
tr_labels = list(tr_data.label)

# tokenize
tr_data_tokenized = [(mecab.morphs(d),l) for d,l in zip(tr_documents, tr_labels)]

# build vocab
counter = nlp.data.count_tokens(itertools.chain.from_iterable([d for d,l in tr_data_tokenized]))
vocab = nlp.Vocab(counter=counter, min_freq=10)

# connecting embedding to vocab
ptr_embedding = nlp.embedding.create('fasttext',source='wiki.ko')
vocab.set_embedding(ptr_embedding)

# saving vocab
with open('../data/vocab.pkl', mode = 'wb') as io:
    pickle.dump(vocab, io)

# saving training / test dataset to txt
tr_data.to_csv('../data/movie/tr_ratings.txt', index=False, sep='\t')
tst_data.to_csv('../data/movie/tst_ratings.txt', index=False, sep='\t')
