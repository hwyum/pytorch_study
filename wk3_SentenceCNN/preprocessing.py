import numpy as np
import pandas as pd
import itertools
import gluonnlp as nlp
from sklearn.model_selection import train_test_split
import pickle
from model.data import movie_data
from konlpy.tag import Mecab
from model.utils import Vocab
from collections import Counter

mecab = Mecab()

# load dataset
data = pd.read_table('./data/ratings.txt')
data = data[['document', 'label']] # document 컬럼과 label 컬럼만 남김
data = data[~data.document.isna()] # NA 제거

# train / dev / test split
tr_data, tst_data = train_test_split(data , test_size=0.2)
tr_data, dev_data = train_test_split(tr_data, test_size=0.2)

### 아래부터는 training dataset만 사용함
tr_documents = list(tr_data.document)
tr_labels = list(tr_data.label)

# tokenize
tr_data_tokenized = [(mecab.morphs(d),l) for d,l in zip(tr_documents, tr_labels)]

# build vocab
counter = Counter(itertools.chain.from_iterable([d for d,l in tr_data_tokenized]))
vocab = Vocab(counter=counter, min_freq=10, bos_token=None, eos_token=None)

# connecting embedding to vocab
ptr_embedding = nlp.embedding.create('fasttext',source='wiki.ko')
embeddings = []
for key, idx in vocab.token_to_idx.items():
    if key in ['<pad>', '<unk>']:
        embeddings.append(np.zeros(300))
        continue
    ind = ptr_embedding.token_to_idx[key]
    embedding = ptr_embedding.idx_to_vec[ind]
    embeddings.append(embedding.asnumpy())
vocab.embedding = np.array(embeddings)

# saving vocab
with open('./data/vocab.pkl', mode = 'wb') as io:
    pickle.dump(vocab, io)

# saving training / test dataset to txt
tr_data.to_csv('./data/tr_ratings.txt', index=False, sep='\t')
dev_data.to_csv('./data/dev_ratings.txt', index=False, sep='\t')
tst_data.to_csv('./data/tst_ratings.txt', index=False, sep='\t')
