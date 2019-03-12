from torch.utils.data import Dataset, DataLoader, ConcatDataset
import pandas as pd

class movie_data(Dataset):
    """naver movie dataset class"""

    def __init__(self, batch_name, transform=None):
        """
        Args:
            batch_name (string): batch file name
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        # path = './data/cifar-10-batches-py/'
        #
        # path = os.path.join(path, batch_name)
        # self.batch = unpickle(path)
        # self.transform = transform

    def __len__(self):
        # return len(self.batch[b'data'])

    def __getitem__(self, idx):
        # image = self.batch[b'data'].reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("uint8")[idx]
        # label = np.array(self.batch[b'labels'])[idx]
        # sample = (image, label)
        #
        # if self.transform:
        #     sample = self.transform(sample)
        #
        # return sample