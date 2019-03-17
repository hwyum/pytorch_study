from torch.utils.data import Dataset, DataLoader, ConcatDataset
import pandas as pd
import os

class movie_data(Dataset):
    """naver movie dataset class"""

    def __init__(self, filename, transform=None):
        """ Instantiating movie_dataset class

        Args:
            filename (string): data file name (train+test: ratings.txt)
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        path = '../data/movie/'
        path = os.path.join(path, filename)
        # path = os.path.join(path, batch_name)
        # self.batch = unpickle(path)
        self.data = pd.read_table(path)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        document = self.data['document'].iloc[idx]
        label = self.data['label'].iloc[idx]
        sample = (document, label)

        return sample