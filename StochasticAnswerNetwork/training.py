import argparse
from path import Path
from utils import Config
from model.data import SentencePair, collate_fn
# from model.network import MaLSTM
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from konlpy.tag import Mecab
from tqdm import tqdm, tqdm_notebook
from tokenizer import JamoTokenizer
import gluonnlp as nlp

data_dir = './data'
model_dir = './experiments/base_model'
data_dir = Path(data_dir)
model_dir = Path(model_dir)

data_config = Config(data_dir / 'config.json')
model_config = Config(model_dir / 'config.json')

# Vocab and Tokenizer
with open(data_config.word_vocab_path, mode='rb') as io:
    word_vocab = pickle.load(io)
word_tokenizer = Mecab().morphs

with open(data_config.char_vocab_path, mode='rb') as io:
    char_vocab = pickle.load(io)
char_tokenizer = JamoTokenizer().tokenize
char_padder = nlp.data.PadSequence(length=model_config.char_max_len)

# Model
# model = MaLSTM(vocab, model_config.embedding_dim, model_config.hidden_size)

# DataLoader
tr_ds = SentencePair(data_config.tr_path, word_vocab, char_vocab, word_tokenizer, char_tokenizer, char_padder)
val_ds = SentencePair(data_config.val_path, word_vocab, char_vocab, word_tokenizer, char_tokenizer, char_padder)

tr_dl = DataLoader(tr_ds, batch_size=2, collate_fn=collate_fn)