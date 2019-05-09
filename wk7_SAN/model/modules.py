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
        self._is_pair = is_pair
        self._embedding = nn.Embedding(num_embeddings=num_embedding, embedding_dim=embedding_dim, padding_idx=0)

        # Embedding output : N x L x C

    def forward(self, inputs: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # print("[Embedding] inputs: {}".format(inputs))

        if self._is_pair:
            q1, q2= inputs
            emb_out_q1 = self._embedding(q1)
            emb_out_q2 = self._embedding(q2)
            return emb_out_q1, emb_out_q2

        else:
            return self._embedding(inputs)

class BiLSTM(nn.Module):
    """ class for bidirectional LSTM """
    def __init__(self, input_size, hidden_size, is_pair:bool=True) -> None:
        """ initialization of BiLSTM class """
        super(BiLSTM, self).__init__()
        self._hidden_size = hidden_size
        self._is_pair = is_pair
        self._bilstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True, bidirectional=True)

    def forward(self, inputs: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if self._is_pair:
            q1, q2 = inputs
            outputs1, _ = self._bilstm(q1)
            outputs2, _ = self._bilstm(q2)

            hn_cat1 = torch.cat((outputs1[:, :, :self._hidden_size], outputs1[:, :, self._hidden_size:]), dim=2)
            hn_cat2 = torch.cat((outputs2[:, :, :self._hidden_size], outputs2[:, :, self._hidden_size:]), dim=2)
            return hn_cat1, hn_cat2

        else:
            outputs, _ = self._inputs     # outputs : batch, seq_len, num_directions * hidden_size)
            hn_cat = torch.cat((outputs[:,:,:self._hidden_size], outputs[:,:,self._hidden_size:]), dim=2)
            return hn_cat  # output shape: Batch x seq_len x (Hidden_dim * 2)

class SelfAttention(nn.Module):
    """ class for self attention """
    def __init__(self, in_features, hidden_units, hops, is_pair:bool=True) -> None:
        super(SelfAttention, self).__init__()
        self._is_pair = is_pair
        self._linear_1 = nn.Linear(in_features=in_features, out_features=hidden_units)
        self._linear_2 = nn.Linear(in_features=hidden_units, out_features=hops)

    def forward(self, inputs: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if self._is_pair:
            q1_hn, q2_hn = inputs  # batch x 2u x n
            q1_weights = F.softmax(self._linear_2(F.tanh(self._linear_1(q1_hn))), dim=-1)  # batch x r x n (seq_len)
            q2_weights = F.softmax(self._linear_2(F.tanh(self._linear_1(q2_hn))), dim=-1)
            q1_representations = torch.matmul(q1_weights, q1_hn.permute(0,2,1))     # batch x r x 2u
            q2_representations = torch.matmul(q2_weights, q2_hn.permute(0,2,1))

            return q1_representations, q2_representations

        else:
            weights = F.softmax(self._linear_2(F.tanh(self._linear_1(inputs))), dim=-1)
            representations = torch.matmul(weights, inputs.permute(0,2,1))

            return representations


class BatchMM(nn.Module):
    """ class for calculating Batch Dot Product
    to implement F_h = batcheddot(M_h, W_fh)
    """
    def __init__(self) -> None:
        super(BatchMM, self).__init__()

        self._W_fh = nn.Linear()


    def forward(self, *input):



class Permute(nn.Module):
    """ puermutation """
    def __init__(self) -> None:
        super(Permute, self).__init__()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input.permute(0, 2, 1), length