
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, pad_packed_sequence, pack_sequence, pack_padded_sequence, PackedSequence

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
        # Embedding output : N x L x C

    def forward(self, inputs: (torch.Tensor, torch.Tensor)) -> (torch.Tensor, torch.Tensor):
        # print("[Embedding] inputs: {}".format(inputs))
        input, length = inputs
        emb_out = self._embedding(input)

        # print("\nInput length: {}".format(length))
        return emb_out, length


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

        self._in_channels = in_channels
        self._out_channels = out_channels
        self._kernel_size = kernel_size
        self._pooling_size = pooling_size

        self._conv = nn.Conv1d(in_channels=self._in_channels, out_channels=self._out_channels, kernel_size=self._kernel_size)
        self._maxpool = nn.MaxPool1d(self._pooling_size)

    def forward(self, inputs: (torch.Tensor, torch.Tensor)) -> (torch.Tensor, torch.Tensor):
        input, length = inputs
        output = self._maxpool(self._conv(input))
        output = F.relu(output)

        # length tracking
        length = length - (self._kernel_size - 1)
        length = (length - (self._pooling_size - 1) - 1) / (self._pooling_size) + 1
        # print("\nConv length: {}".format(length))
        return output, length


class Pack_padded_seq(nn.Module):
    """ class for buiding pack_padded_sequence """
    def __init__(self) -> None:
        super(Pack_padded_seq,self).__init__()

    def forward(self, inputs: (torch.Tensor, torch.Tensor)) -> PackedSequence:
        input, length = inputs  # input: batch x length x hidden
        
        pad_seq = pad_sequence(input, batch_first=True)
        sorted_idx = torch.argsort(length, descending=True)
#         input = input[sorted_idx]
        length = length[sorted_idx]        
        padded_seq = pad_seq[sorted_idx]
        pack_padded_seq = pack_padded_sequence(padded_seq,length, batch_first=True)
        # print("input shape:{}, length shape:{}".format(input.size(), length.size()))
        # print("Rec Input length: {}".format(length))
        return pack_padded_seq


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

    def forward(self, input: PackedSequence) -> torch.Tensor:
        """
        input: (batch, seq, feature) when batch_first = True
        """
        packed_output, (hn, cn) = self._bilstm(input)  # output shape: Batch x Seq_len x (Hidden_dim * 2)
        # print('output shape for Bi-LSTM: {}'.format(output.size()))

        hn_cat = torch.cat([hn[0], hn[1]], dim=1)
        return hn_cat  # output shape: Batch x (Hidden_dim * 2)


class Flatten(nn.Module):
    """ flattening conv output to feed into FC layers"""
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return torch.flatten(input, start_dim=1)


class Permute(nn.Module):
    """ puermutation """
    def __init__(self):
        super(Permute, self).__init__()

    def forward(self, inputs: (torch.Tensor, torch.Tensor)) -> (torch.Tensor, torch.Tensor):
        input, length = inputs
        return input.permute(0, 2, 1), length


class Dropout(nn.Module):
    """ dropout with keeping length """
    def __init__(self, prob=0.5):
        super(Dropout, self).__init__()
        self._dropout = nn.Dropout(prob)

    def forward(self, inputs: (torch.Tensor, torch.Tensor)) -> (torch.Tensor, torch.Tensor):
        input, length = inputs
        return self._dropout(input), length
    