import random
import math
import torch
import pandas as pd
import gluonnlp as nlp
from konlpy.tag import Mecab
from typing import Tuple, List
from torch.utils.data import Sampler, Dataset
from torch.utils.data.sampler import BatchSampler, RandomSampler
from torch.nn.utils.rnn import pad_sequence
from typing import Callable, Iterable


class SentencePair(Dataset):
    """ Korean Question Pair Dataset """
    def __init__(self, datapath:str, word_vocab:nlp.Vocab, char_vocab:nlp.Vocab, word_tokenizer, char_tokenizer, char_padder) -> None:
        """
        Args:
            datapath:
            vocab:
            word_tokenizer:
            char_tokenizer:
            char_padder:
        """
        self.corpus = pd.read_csv(datapath)
        self.word_vocab = word_vocab
        self.char_vocab = char_vocab
        self.word_tokenizer = word_tokenizer
        self.char_tokenizer = char_tokenizer
        self.char_padder = char_padder


    def __len__(self) -> int :
        """ return dataset length """
        return len(self.corpus)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q1 = self.corpus.iloc[idx].question1
        q2 = self.corpus.iloc[idx].question2
        label = self.corpus.iloc[idx].is_duplicate

        q1_word_tokenized_toidx = [self.word_vocab.token_to_idx[token] for token in self.word_tokenizer(q1)]
        q2_word_tokenized_toidx = [self.word_vocab.token_to_idx[token] for token in self.word_tokenizer(q2)]

        q1_char_tokenized_toidx = [self.char_padder([self.char_vocab.token_to_idx[char] for char in self.char_tokenizer(token)]) \
                                   for token in self.word_tokenizer(q1)]
        q2_char_tokenized_toidx = [self.char_padder([self.char_vocab.token_to_idx[char] for char in self.char_tokenizer(token)]) \
                                   for token in self.word_tokenizer(q2)]

        sample = (torch.tensor(q1_word_tokenized_toidx), torch.tensor(q1_char_tokenized_toidx),
                  torch.tensor(q2_word_tokenized_toidx), torch.tensor(q2_char_tokenized_toidx), torch.tensor(label))

        return sample


def collate_fn(inputs: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]) \
        -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

    q1_word, q1_char, q2_word, q2_char, label = list(zip(*inputs))

    q1_word = pad_sequence(q1_word, batch_first=True, padding_value=1)
    q2_word = pad_sequence(q2_word, batch_first=True, padding_value=1)

    q1_char = pad_sequence(q1_char, batch_first=False, padding_value=1).permute(1, 0, 2)
    q2_char = pad_sequence(q2_char, batch_first=False, padding_value=1).permute(1, 0, 2)


    return q1_word, q1_char, q2_word, q2_char, torch.tensor(label)