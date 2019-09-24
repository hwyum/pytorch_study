from torch.utils.data import Dataset
import torch
import pandas as pd
import gluonnlp as nlp
from konlpy.tag import Mecab
from typing import Tuple, List
from torch.utils.data import Sampler
from torch.nn.utils.rnn import pad_sequence


class QuestionPair(Dataset):
    """ Korean Question Pair Dataset """
    def __init__(self, datapath:str, vocab:nlp.Vocab, tokenizer) -> None:
        self.df = pd.read_csv(datapath)
        self.vocab = vocab
        self.tokenizer = tokenizer

        self.question1_idx = self.df.question1.map(lambda x: [vocab.token_to_idx[token] for token in tokenizer(x)])
        self.question2_idx = self.df.question2.map(lambda x: [vocab.token_to_idx[token] for token in tokenizer(x)])

    def __len__(self) -> int :
        """ return dataset length """
        return len(self.df)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q1 = self.df.iloc[idx].question1
        q2 = self.df.iloc[idx].question2
        label = self.df.iloc[idx].is_duplicate

        q1_tokenized_toidx = [self.vocab.token_to_idx[token] for token in self.tokenizer(q1)]
        q2_tokenized_toidx = [self.vocab.token_to_idx[token] for token in self.tokenizer(q2)]

        sample = (torch.tensor(q1_tokenized_toidx), torch.tensor(q2_tokenized_toidx), torch.tensor(label))

        return sample


class BucketedSampler(Sampler):
    """ Sampler for bucketing """
    def __init__(self, data_source, bucketing_tgt=1):
        self.data_source = data_source
        if bucketing_tgt not in [1,2]:
            raise ValueError("bucketing target should be 1 or 2 (sentence1 or sentence2"
                             ", but got bucketing target={}".format(bucketing_tgt))
        self.bucketing_tgt = bucketing_tgt

    def __iter__(self):
        if self.bucketing_tgt == 1:
            sentence_length = self.data_source.question1_idx.map(len) # Series
        else:
            sentence_length = self.data_source.question2_idx.map(len)  # Series
        sorted_idx = sentence_length.sort_values(ascending=True).index.tolist()
        return iter(sorted_idx)


def collate_fn(inputs: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    question1, question2, label = list(zip(*inputs))
    question1 = pad_sequence(question1, batch_first=True, padding_value=1)
    question2 = pad_sequence(question2, batch_first=True, padding_value=1)

    return question1, question2, label

