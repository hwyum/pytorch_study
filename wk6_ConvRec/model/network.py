import torch
import torch.nn as nn
from torch.nn import ModuleList
import torch.nn.functional as F
from model.modules import Embedding, Permute, ConvLayer, RecLayer, Flatten
from typing import List


class ConvRec(nn.Module):
    """ Implementation of Convolution-Recurrent Network """
    def __init__(self, num_embedding:int, embedding_dim:int, conv_in_channels:List, conv_out_channels:List,
                 conv_kernel_size:List, conv_pooling_size:List, hidden_size, class_num:int) -> None:
        """
        Initialization of CovRec Class
        :param num_embedding: input vocab size
        :param embedding_dim: output embedding dimension (hyper-parameter)
        :param
        :param
        """
        super(ConvRec, self).__init__()

        self._convLayers = ModuleList([ConvLayer(i, o, k, p) for i, o, k, p in zip(conv_in_channels, conv_out_channels,
                                                                                   conv_kernel_size, conv_pooling_size)])

        self._recLayer = RecLayer(input_size=conv_out_channels[-1], hidden_size=hidden_size)

        self._ops = nn.Sequential(Embedding(num_embedding, embedding_dim),  # output_shape : NLC
                                  Permute(),   # output_shape : NCL
                                  *self._convLayers,  # output_shape : NCL
                                  nn.Dropout(0.5),
                                  self._recLayer,    # output_shape : NLC
                                  nn.Dropout(0.5),
                                  Flatten(),
                                  nn.Linear(hidden_size * 2, class_num))

    def forward(self, input: torch.tensor) -> torch.tensor:
        output = self._ops(input)
        return output