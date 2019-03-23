
import torch
import torch.nn as nn
import torch.functional as F

class characterCNN(nn.Module):
    ''' Character level CNN implementation'''
    def __init__(self, num_embedding, embedding_dim, max_len=300, class_num=2) -> None:
        '''
        Args:
            num_embedding: length of the token2index dictionary (size of the character set)
            embedding_dim: embedding dimension
            max_len: maximum length of sequences, default = 300
            class_num: the number of output classes, default = 2
        '''

        super(characterCNN).__init__()

        self._embedding = nn.Embedding(num_embeddings=num_embedding, embedding_dim=embedding_dim, padding_idx=0)
        self._conv_1 = nn.Conv1d(in_channels=embedding_dim, out_channels=1024, kernel_size=7)
        self._conv_2 = nn.Conv1d(in_channels=1024, out_channels=1024, kernel_size=7)
        self._conv_3 = nn.Conv1d(in_channels=1024, out_channels=1024, kernel_size=3)
        self._maxpool = nn.MaxPool1d(3)

        self._fc1 = nn.Linear(in_features=1024*7, out_features=2048)
        self._fc2 = nn.Linear(in_features=2048, out_features=2048)
        self._fc3 = nn.Linear(in_features=2048, out_features=class_num)

        # initialization
        self.apply(self.__init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._embedding(x) # Output: (*, embedding_dim), where * is the input shape -> (N, L, Embedding_dim)
        x = x.permute(0, 2, 1)

        # Conv Layers #1~6 (Input Shape: N x C x L)
        x = self._maxpool(self._conv_1(x)) # N x 1024 x 294 -> N x 1024 x 98
        x = self._maxpool(self._conv_2(x)) # N x 1024 x 92 -> N x 1024 x 30
        x = self._conv_3(x) # N x 1024 x 28
        x = self._conv_3(x) # N x 1024 x 26
        x = self._conv_3(x) # N x 1024 x 24
        conv_output = self._maxpool(self._conv_3(x)) # N x 1024 x 22 -> N x 1024 x 7

        # FC Layers #7~9
        fc_output = self._fc1(conv_output)
        fc_output = self._fc2(fc_output)
        fc_output = self._fc3(fc_output)

        return fc_output

    ## initialization에 대해서는 좀 더 찾아볼것!!
    def __init_weights(self, layer) -> None:
        nn.init.normal_(layer.weight, mean=0, std=0.02) # 논문 설정
        # if isinstance(layer, nn.Conv1d):
        #     nn.init.kaiming_uniform_(layer.weight)
        # elif isinstance(layer, nn.Linear):
        #     nn.init.xavier_normal_(layer.weight)