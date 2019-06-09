import torch
import torch.nn as nn
from typing import Union, Tuple
import gluonnlp as nlp

class Embedding(nn.Module):
    """ class for Embedding """
    def __init__(self, num_embedding:int, embedding_dim:int, is_pair:bool=False, is_pretrained:bool=False, idx_to_vec=None) -> None:
        """
        initialization of Embedding class
        :param num_embedding: size of vocab
        :param embedding_dim: output embedding dimension
        :param is_pair: whether input data is a set of paired sentences or single sentences
        :param is_pretrained: whether creating embedding vector from pretrained weights
        :param idx_to_vec: (only of is_pretrained=True) idx_to_vec
        """
        super(Embedding, self).__init__()
        self._is_pair = is_pair
        self._embedding = nn.Embedding(num_embeddings=num_embedding, embedding_dim=embedding_dim, padding_idx=0)
        self._is_pretrained = is_pretrained
        self._idx_to_vec = idx_to_vec

        if self._is_pretrained:
            assert self._idx_to_vec is not None
            self._embedding_from_pretrained = nn.Embedding.from_pretrained(torch.from_numpy(self._idx_to_vec.asnumpy()), freeze=True)

        # Embedding output : N x L x C

    def forward(self, inputs: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if self._is_pretrained:
            return  self._embedding_from_pretrained(inputs)

        if self._is_pair:
            q1, q2= inputs
            emb_out_q1 = self._embedding(q1)
            emb_out_q2 = self._embedding(q2)
            return emb_out_q1, emb_out_q2

        else:
            return self._embedding(inputs)


class BiLSTM(nn.Module):
    """ class for bidirectional LSTM """
    def __init__(self, input_size, hidden_size, is_pair:bool=False) -> None:
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
            return outputs1, outputs2

        else:
            outputs, hidden = self._bilstm(inputs)     # outputs : batch, seq_len, num_directions * hidden_size)
            return outputs, hidden  # output shape: Batch x seq_len x (Hidden_dim * 2)