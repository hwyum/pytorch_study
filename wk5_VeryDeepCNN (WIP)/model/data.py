from torch.utils.data import Dataset
import torch
import pandas as pd
import numpy as np


class MovieDataJaso(Dataset):
    """
    jaso level movie dataset class
    """

    def __init__(self, filepath, tokenizer, padder) -> None:
        """
        initializing the class
        :param filepath: dataset file path
        :param tokenizer: tokenizer used to tokenize into jaso level
        :param padder: padder used to pad when the length of the sequence is less than max_len
        """

        self.data = pd.read_table(filepath)
        self.tokenizer = tokenizer
        self.padder = padder

    def __len__(self) -> int:
        """return dataset length"""
        return len(self.data)

    def __getitem__(self, idx) -> torch.Tensor:
        """ return specific index of the data in model input type """

        document = self.data.iloc[idx].document
        label = self.data.iloc[idx].label
        document_tokenized = self.padder(self.tokenizer.tokenize_and_transform(document))

        # print(type(document_tokenized), type(torch.tensor(np.asarray(label))))

        sample = (torch.tensor(document_tokenized), torch.tensor(np.asarray(label))) # transform into torch tensor

        return sample