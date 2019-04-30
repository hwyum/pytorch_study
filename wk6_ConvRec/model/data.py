
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
        :param tokenizer: tokenizer used to tokenize into Jaso(자소) level
        :param padder: padder used to pad when the length of the sequence is less than max_len (max_len: hyper parameter)
        """

        # read_table is deprecated, use read_csv instead, passing sep = '\t'.
        data = pd.read_csv(filepath, sep='\t')

        # 토큰 길이 20 이상인 데이터만 활용
        self.data = data[data.document.map(lambda x: len(tokenizer.tokenize(x))) >= 20]
        self.tokenizer = tokenizer
        self.padder = padder

    def __len__(self) -> int:
        """return dataset length"""
        return len(self.data)

    def __getitem__(self, idx) -> torch.Tensor:
        """ return specific index of the data in model input type """

        document = self.data.iloc[idx].document
        label = self.data.iloc[idx].label
        document_tokenized = self.tokenizer.tokenize_and_transform(document)
        length = len(document_tokenized)
        document_tokenized_and_padded = self.padder(document_tokenized)

        # print(type(document_tokenized), type(torch.tensor(np.asarray(label))))

        # transform into torch tensor
        sample = (torch.tensor(document_tokenized_and_padded), torch.tensor(np.asarray(label)), torch.tensor(np.asarray(length)))

        return sample