import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Union

class Embedding(nn.Module):
    """ class for Embedding """
    def __init__(self, num_embedding, embedding_dim, is_pair:bool=True) -> None:
        """
        initialization of Embedding class
        :param num_embedding: size of vocab
        :param embedding_dim: output embedding dimension
        :param is_pair: whether input data is a set of paired sentences or single sentences
        """
        super(Embedding, self).__init__()
        self.is_pair = is_pair
        self._embedding = nn.Embedding(num_embeddings=num_embedding, embedding_dim=embedding_dim, padding_idx=0)

        # Embedding output : N x L x C

    def forward(self, inputs: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # print("[Embedding] inputs: {}".format(inputs))

        if self.is_pair:
            q1, q2= inputs
            emb_out_q1 = self._embedding(q1)
            emb_out_q2 = self._embedding(q2)
            return emb_out_q1, emb_out_q2

        else:
            return self._embedding(inputs)

class BiLSTM(nn.Module):
    """ class for bidirectional LSTM """
    def __init__(self, input_size, hidden_size, is_pair:bool) -> None:
        """ initialization of BiLSTM class """
        super(BiLSTM, self).__init__()
        self.is_pair = is_pair
        self._bilstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True, bidirectional=True)

    def forward(self, inputs: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if self.is_pair:
            q1, q2 = inputs
            output1, (hn1, cn1) = self._bilstm(q1)
            output2, (hn2, cn2) = self._bilstm(q2)

            hn_cat1 = torch.cat([hn1[0], hn1[1]], dim=1)
            hn_cat2 = torch.cat([hn2[0], hn2[1]], dim=1)
            return hn_cat1, hn_cat2

        else:
            output, (hn, cn) = self._inputs     # hn : hidden state of the last time step
            hn_cat = torch.cat([hn[0], hn[1]], dim=1)
            return hn_cat  # output shape: Batch x (Hidden_dim * 2)

