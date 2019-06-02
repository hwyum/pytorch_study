import sys
sys.path.append( '/Users/haewonyum/Google 드라이브/Colab Notebooks/Pytorch_study/wk8_BiLSTM-CRF')

import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
import time
import mxnet
import gluonnlp as nlp
import itertools
import pickle


data_root = './data'
data = pd.read_csv('./data/train_data.txt', sep=' |\t', header=None)
data.columns = ['idx', 'word', 'tag']

# 문장단위 데이터로 정리
sentence_lst = []
tags_lst = []
sentence = []
tags = []

start = time.time()
for row, idx in enumerate(data.idx):
    idx = int(idx)
    if idx == 1:
        if row > 0:
            sentence_lst.append(' '.join(sentence))
            tags_lst.append(' '.join(tags))
            sentence = []
            tags = []
        sentence.append(data.iloc[row].word)
        tags.append(data.iloc[row].tag)
    else:
        sentence.append(data.iloc[row].word)
        tags.append(data.iloc[row].tag)

    if (row+1) % 1000 == 0:
        print("\r{} / {} data processed".format(row+1, len(data)))

    # if row > 10000: break

sentence_lst.append(' '.join(sentence))
tags_lst.append(' '.join(tags))
data_new = pd.DataFrame.from_dict({'sentence':sentence_lst, 'tags':tags_lst})

# train-test split
train_df, tst_df = train_test_split(data_new, test_size=0.2)

# train-validation split
train_df, val_df = train_test_split(train_df, test_size=0.2)

# save train / validation / test data
train_df.to_csv(Path(data_root) / 'tr_data.txt', index=False, sep='\t')
val_df.to_csv(Path(data_root) / 'val_data.txt' , index=False, sep='\t')
tst_df.to_csv(Path(data_root) / 'tst_data.txt' , index=False, sep='\t')

end = time.time()
print("train_df: {}".format(len(train_df)))
print("val_df: {}".format(len(val_df)))
print("tst_df: {}".format(len(tst_df)))
print("consumed {} seconds".format(end-start))

# Build Vocab with training data
counter = nlp.data.count_tokens(itertools.chain.from_iterable(
            [sentence.split() for sentence in train_df.sentence]))
vocab = nlp.Vocab(counter=counter, bos_token=None, eos_token=None)
counter_tag = nlp.data.count_tokens(itertools.chain.from_iterable(
            [tags.split() for tags in train_df.tags]))
vocab_tag = nlp.Vocab(counter=counter_tag, bos_token=None, eos_token=None)

# connecting SISG embedding with vocab
ptr_embedding = nlp.embedding.create('fasttext', source='wiki.ko')
vocab.set_embedding(ptr_embedding)

# saving vocab
with open(Path(data_root) / 'vocab.pkl', mode='wb') as io:
    pickle.dump(vocab, io)

with open(Path(data_root) / 'vocab_tag.pkl', mode='wb') as io:
    pickle.dump(vocab_tag, io)