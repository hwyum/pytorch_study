import torch
from torch.utils.data import Dataset
import pandas as pd
import mxnet
import gluonnlp as nlp
from typing import Tuple


class NER_data(Dataset):
    """ Dataset class for dealing with Korean NER dataset """
    def __init__(self, filepath, vocab:nlp.Vocab, tag_to_ix):
        self.data = pd.read_csv(filepath, sep='\t')
        self.vocab = vocab
        self.tag_to_ix = tag_to_ix

    def __len__(self) -> int:
        """ return dataset length """
        return len(self.data)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        """ return preprocessed idx_th data in form of (setence, tags) """
        sentence = self.data.iloc[idx].sentence.split()
        tags = self.data.iloc[idx].tags.split()
        sentence_to_idx = [self.vocab[word] for word in sentence]
        tags_to_idx = [self.tag_to_ix[tag] for tag in tags]

        return torch.tensor(sentence_to_idx), torch.tensor(tags_to_idx)



