from torch.utils.data import Dataset
import torch
import pandas as pd

class movie_data_jaso(Dataset):
    ''' jaso level movie dataset class '''

    def __init__(self, filepath, tokenizer, padder) -> None:
        ''' initializing the class
        Args:
            filepath : dataset file path
            tokenizer : tokenizer to be tokenized into jaso level
            padder : padder to be padded
        '''

        self.data = pd.read_table(filepath)
        self.tokenizer = tokenizer
        self.padder = padder

    def __len__(self) -> int:
        ''' return dataset length'''
        return len(self.data)

    def __getitem__(self, idx) -> torch.Tensor:
        ''' return specific index of the data in model input type '''

        document = self.data.iloc[idx].document
        label = self.data.iloc[idx].label
        document_tokenized = self.tokenizer.tokenize_and_transform(document)
        sample = (torch.tensor(document_tokenized), torch.tensor(label))

        return sample



