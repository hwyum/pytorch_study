import torch
import torch.nn as nn
import torch.nn.functional as F

class Embedding(nn.Module):
    """ class for Embedding """
    def __init__(self, num_embedding, embedding_dim) -> None:
        """
        initialization of Embedding class
        :param num_embedding: size of vocab
        :param embedding_dim: output embedding dimension
        """
        super(Embedding, self).__init__()
        self._embedding = nn.Embedding(num_embeddings=num_embedding, embedding_dim=embedding_dim, padding_idx=0)

    def forward(self, input:torch.tensor) -> torch.tensor:
        return self._embedding(input)

class ConvLayer(nn.Module):
    """ class for Convolusional layer with max pooling """
    def __init__(self, in_channels:int, out_channels:int, kernel_size:int, pooling_size:int) -> None:
        """
        initialization of ConvLayer class
        :param in_channels: number of input channels
        :param out_channels: number of output channels
        :param kernel_size: kernel size
        :param pooling_size: pooling size for max pooling
        """
        super(ConvLayer, self).__init__()
        self._conv = nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size)
        self._maxpool = nn.MaxPool1d(pooling_size)

    def forward(self, input:torch.tensor) -> torch.tensor:
        output = self._maxpool(self._conv(input))
        output = F.relu(output)
        # print('output shape for ConvLayer: {}'.format(output.size()))
        return output

class RecLayer(nn.Module):
    """ class for Bi-LSTM layer """
    def __init__(self, input_size, hidden_size):
        """
        initialization of RecLayer class
        :param input_size: input dimension (of one timestep)
        :param hidden_size: hidden dimension for hidden units
        """
        super(RecLayer,self).__init__()
        self._bilstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True, bidirectional=True)
        self._dropout = nn.Dropout(0.5)

    def forward(self, input:torch.tensor) -> torch.tensor:
        """
        input: (batch, seq, feature) when batch_first = True
        """
        input = input.permute(0, 2, 1)  # batch x channel x length -> batch x length x channel
        # print('input shape for RecLayer: {}'.format(input.size()))

        output, (hn, cn) = self._bilstm(input)  # output shape: Batch x Seq_len x (Hidden_dim * 2)
        # print('output shape for Bi-LSTM: {}'.format(output.size()))

        hn_cat = torch.cat([hn[0], hn[1]], dim=1)
        return hn_cat  # output shape: Batch x (Hidden_dim * 2)



class Flatten(nn.Module):
    """ flattening conv output to feed into FC layers"""

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, input: torch.tensor) -> torch.tensor:
        return torch.flatten(input, start_dim=1)


class Permute(nn.Module):
    """ puermutation """
    def __init__(self):
        super(Permute, self).__init__()

    def forward(self, input: torch.tensor) -> torch.tensor:
        return input.permute(0, 2, 1)

### Test
# import numpy as np
# test = torch.tensor(np.array([1,2,3,4,5]), dtype=torch.float32)
# test = test.view(1,5,1)
#
# padded_indices_wbf = padded_indices_wbf.view(5,13,1)
#
# bilstm = nn.LSTM(input_size=1, hidden_size=10, batch_first=True, bidirectional=True)
# hn, cn = bilstm(padded_indices_wbf)[1]
# hn[0].size()
# torch.cat([hn[0],hn[1]], dim=1).size()
