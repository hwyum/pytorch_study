
import torch
import torch.nn as nn
import torch.functional as F

class characterCNN(nn.Module):
    ''' Character level CNN implementation'''
    def __init__(self, num_enbedding, embedding_dim, class_num) -> None:
        '''
        Args:
            class_num: the number of output classes
        '''

        super(characterCNN).__init__()

        self._embedding = nn.Embedding(num_embeddings=num_enbedding, embedding_dim=embedding_dim, padding_idx=0)
        self._conv_1 = nn.Conv1d(in_channels=embedding_dim, out_channels=1024, kernel_size=7)
        self._conv_2 = nn.Conv1d(in_channels=1024, out_channels=1024, kernel_size=7)
        self._conv_3 = nn.Conv1d(in_channels=1024, out_channels=1024, kernel_size=3)
        self._maxpool = nn.MaxPool1d(3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return

    ## initialization에 대해서는 좀 더 찾아볼것!!
    def __init_weights(self, layer) -> None:
        if isinstance(layer, nn.Conv1d):
            nn.init.kaiming_uniform_(layer.weight)
        elif isinstance(layer, nn.Linear):
            nn.init.xavier_normal_(layer.weight)