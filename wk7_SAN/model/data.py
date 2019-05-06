from torch.utils.data import Dataset
import torch
import pandas as pd
import gluonnlp as nlp
from konlpy.tag import Mecab
from typing import Tuple

class QuestionPair():
    """ Korean Question Pair Dataset """
    def __init__(self, datapath:str, vocab:nlp.Vocab, tokenizer: Mecab) -> None:
        self.df = pd.read_csv(datapath, sep='\t')
        self.vocab = vocab
        self.tokenizer = tokenizer

    def __len__(self) -> int :
        """ return dataset length """
        return len(self.df)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q1 = self.df.iloc[idx].question1
        q2 = self.df.iloc[idx].question2
        label = self.df.iloc[idx].is_duplicate

        q1_tokenized_toidx = [self.vocab.token_to_idx[token] for token in self.tokenizer.morphs(q1)]
        q2_tokenized_toidx = [self.vocab.token_to_idx[token] for token in self.tokenizer.morphs(q2)]

        sample = (torch.tensor(q1_tokenized_toidx), torch.tensor(q2_tokenized_toidx), label)

        return sample
