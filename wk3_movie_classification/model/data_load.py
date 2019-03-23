from torch.utils.data import Dataset, DataLoader, ConcatDataset
import pandas as pd
import os
import torch

class movie_data(Dataset):
    """naver data dataset class"""

    def __init__(self, filepath, vocab, tokenizer, padder, transform=None):
        """ Instantiating movie_dataset class

        Args:
            filepath (string): data file path
            vocab (gluonnlp.Vocab): instance of gluonnlp.Vocab
            tokenizer
            padder
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data = pd.read_table(filepath)
        self.vocab = vocab
        self.tokenizer = tokenizer
        self.padder = padder
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        # tokenize & padding
        tokenized = self.tokenizer.morphs(self.data.iloc[idx]['document'])
        tokenized = self.padder(tokenized)

        # token to index
        token_to_idx = self.vocab[tokenized]
        label = self.data.iloc[idx]['label']

        sample = (torch.tensor(self.padder(token_to_idx)), torch.tensor(label))

        return sample