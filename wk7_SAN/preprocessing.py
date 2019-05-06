import os
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
from konlpy.tag import Mecab
import gluonnlp as nlp
import pickle
import itertools

# data load
data_root = './data/'
data_path = Path(data_root) / 'kor_pair_train.csv'
data_df = pd.read_csv(data_path)[['question1','question2','is_duplicate']]
data_df.is_duplicate = data_df.is_duplicate.map(lambda x: 1-x)  # is_duplicate 레이블 변경 : 중복이면 1, 중복 아니면 0으로 (현재 노테이션이 헷갈려서...)

# train-validation split
train_df, val_df = train_test_split(data_df, test_size=0.2)

# save train / validation data
train_df.to_csv(Path(data_root) / 'tr_pairs.txt', index=False, sep='\t')
val_df.to_csv(os.path.join(data_root, 'val_pairs.txt'), index=False, sep='\t')

# build vocab
tokenizer = Mecab()
tr_tokenized_q1 = [tokenizer.morphs(q) for q in train_df.question1]
tr_tokenized_q2 = [tokenizer.morphs(q) for q in train_df.question2]

counter = nlp.data.count_tokens(itertools.chain.from_iterable([tokens for tokens in tr_tokenized_q1+tr_tokenized_q2]))
vocab = nlp.Vocab(counter=counter, min_freq=10)

# save vocab
vocab_path = Path(data_root) / 'vocab.pkl'
with open(vocab_path, mode='wb') as io:
    pickle.dump(vocab, io)
