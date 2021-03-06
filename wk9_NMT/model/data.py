import torch
import torch.nn as nn
from torch.utils.data import Dataset
import gluonnlp as nlp
import pickle
import pandas as pd
from typing import AnyStr, Callable, List, Tuple
from torch.nn.utils.rnn import pad_sequence



class NMTdataset(Dataset):
    """ Dataset class for dealing with Ko-En NMT paired data """
    def __init__(self, path:AnyStr, vocab_src:nlp.Vocab, vocab_tgt:nlp.Vocab, split_fn:Callable) -> None:
        """
        class initialization
        :param path: file path for paired dataset
        """
        self._corpus = pd.read_csv(path, sep='\t' )
        self._corpus = self._corpus[(~self._corpus.ko.isna()) & (~self._corpus.en.isna())]

        # 50자로 길이 제한
        self._corpus = self._corpus[(self._corpus.ko.map(len)<51) & (self._corpus.en.map(len)<51)]

        self._tokenizer_src = Tokenizer(vocab_src, split_fn)
        self._tokenizer_tgt = Tokenizer(vocab_tgt, split_fn)

    def __len__(self) -> int:
        return len(self._corpus)

    def __getitem__(self, idx):

        src = self._tokenizer_src(self._corpus.ko.iloc[idx])
        src = self._tokenizer_src.transform(src)

        tgt_input = self._tokenizer_tgt(self._corpus.en.iloc[idx])
        tgt_input = ['<bos>'] + tgt_input
        tgt_input = self._tokenizer_tgt.transform(tgt_input)

        tgt_output = self._tokenizer_tgt(self._corpus.en.iloc[idx])
        tgt_output = tgt_output + ['<eos>']
        tgt_output = self._tokenizer_tgt.transform(tgt_output)

        return (torch.tensor(src), torch.tensor(tgt_input), torch.tensor(tgt_output))


class Tokenizer():
    """ Class for tokenizing and transfoming to indices given vocab and split function """
    def __init__(self, vocab:nlp.Vocab, split_fn):
        self.vocab = vocab
        self.split_fn = split_fn

    def __call__(self, sent:str):
        """ return tokens generated by split function """
        return self.split_fn(sent)

    def transform(self, tokens:List):
        """ return indices for tokens generated by split function """
        assert type(tokens) == list
        indicies = [self.vocab.token_to_idx[token] for token in tokens]
        return indicies

    def tokenize_and_transform(self, sent:str):
        """ return indices given sentence (first transfrom to tokens and then to indices) """
        tokens = self(sent)
        indicies = [self.vocab.token_to_idx[token] for token in tokens]

        return indicies




#
#
# from torch.utils.data import DataLoader
# with open('./data/vocab.pkl',mode='rb') as io:
#     vocab = pickle.load(io)
#
# split_fn = lambda x: x.split()
# tr_ds = NMTdataset('./data/tr_data_koen.txt', vocab, split_fn)
# tr_dl = DataLoader(tr_ds, batch_size=2, collate_fn=collate_fn)