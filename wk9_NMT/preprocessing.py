import pandas as pd
from pathlib import Path
import re
import gluonnlp as nlp
import itertools
import pickle

# Path
data_root = './data'
tr_ko_path = 'korean-english-park.train.ko'
tr_en_path = 'korean-english-park.train.en'
dev_ko_path = 'korean-english-park.dev.ko'
dev_en_path = 'korean-english-park.dev.en'
tst_ko_path = 'korean-english-park.test.ko'
tst_en_path = 'korean-english-park.test.en'

data_tr_ko, data_tr_en = [], []
data_dev_ko, data_dev_en = [], []
data_tst_ko, data_tst_en = [], []

paths = [tr_ko_path, tr_en_path, dev_ko_path, dev_en_path, tst_ko_path, tst_en_path]
lists = [data_tr_ko, data_tr_en, data_dev_ko, data_dev_en, data_tst_ko, data_tst_en]

# Load data as a list of sentences
for i, path in enumerate(paths):
    with open(Path(data_root) / path, mode='r', newline='\n') as io:
        while (True):
            line = io.readline()
            if not line: break
            lists[i].append(line.replace('\n',''))


# 문장 부호(.!?)의 경우 별도의 토큰으로 분리하고, 이외의 기호는 모두 제거하는 전처리 함수 정의
def normalizeString(s):
#     s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^가-힣a-zA-Z.!?]+", r" ", s)
    return s.strip()


# 위 함수를 이용해 전처리
def dataPreprocess(sentenceList):
    dataset = list(map(normalizeString, sentenceList))
    return dataset

data_tr_ko = dataPreprocess(data_tr_ko)
data_tr_en = dataPreprocess(data_tr_en)
data_dev_ko = dataPreprocess(data_dev_ko)
data_dev_en = dataPreprocess(data_dev_en)
data_tst_ko = dataPreprocess(data_tst_ko)
data_tst_en = dataPreprocess(data_tst_en)


# Save paired datasets
tr_data_koen = pd.DataFrame(list(zip(data_tr_ko, data_tr_en)), columns=['ko','en'])
dev_data_koen = pd.DataFrame(list(zip(data_dev_ko, data_dev_en)), columns=['ko','en'])
tst_data_koen = pd.DataFrame(list(zip(data_tst_ko, data_tst_en)), columns=['ko','en'])

tr_data_koen.to_csv(Path(data_root) / 'tr_data_koen.txt', sep='\t', index=False)
dev_data_koen.to_csv(Path(data_root) / 'dev_data_koen.txt', sep='\t', index=False)
tst_data_koen.to_csv(Path(data_root) / 'tst_data_koen.txt', sep='\t', index=False)


# Build Vocab
# data_tr_koen = data_tr_ko + data_tr_en
counter_ko = nlp.data.count_tokens(itertools.chain.from_iterable([sentence.split() for sentence in data_tr_ko]))
counter_en = nlp.data.count_tokens(itertools.chain.from_iterable([sentence.split() for sentence in data_tr_en]))
vocab_ko = nlp.Vocab(counter_ko, min_freq=10)
vocab_en = nlp.Vocab(counter_en, min_freq=10)

# Save Vocab
with open(Path(data_root) / 'vocab_ko.pkl', mode='wb') as io:
    pickle.dump(vocab_ko, io)

with open(Path(data_root) / 'vocab_en.pkl', mode='wb') as io:
    pickle.dump(vocab_en, io)
