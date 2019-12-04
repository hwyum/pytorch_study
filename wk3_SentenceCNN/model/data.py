from torch.utils.data import Dataset, DataLoader, ConcatDataset
import pandas as pd
import os
import torch
from pathlib import Path
from model.utils import Vocab, Tokenizer, PadSequence
from typing import Callable

class Corpus(Dataset):
    """ general corpus class """
    def __init__(self, filepath:str, vocab:Vocab, tokenizer:Tokenizer, padder:PadSequence=None, sep=',',
                 doc_col:str='document', label_col:str='label'):
        """ Instantiating movie_dataset class
        Args:
            filepath(str): Data file path.
                           Data file should be in comma separated or tab separated format. (default: comma seperated)
                           Data should have two columns containing document and label.
            vocab: pre-defined vocab which is instance of model.utils.Vocab
            tokenizer: instance of model.utils.Tokenizer
            padder: instance of model.utils.PadSequence
            sep: separator to be used to load data
            doc_col: column name for document or sentence (Default: 'document')
            label_col: column name for label (Default: 'label')
        """
        self.data = pd.read_csv(filepath, sep=sep)
        self.vocab = vocab
        self.tokenizer = tokenizer
        self.padder = padder
        self._doc_col = doc_col
        self._label_col = label_col

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tokenized_indices = self.tokenizer.tokenize_and_transform(self.data.iloc[idx][self._doc_col])
        if self.padder:
            tokenized_indices = self.padder(tokenized_indices)
        label = self.data.iloc[idx][self._label_col]

        sample = (torch.tensor(tokenized_indices), torch.tensor(label))
        return sample