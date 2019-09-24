import torch
import torch.nn as nn
from gluonnlp import Vocab
from typing import Union, Tuple


class Embedding(nn.Module):
    """ class for Embedding """
    def __init__(self, num_embeddings, embedding_dim, padding_idx, is_pretrained:bool=False, idx_to_vec=None,
                 freeze:bool=False, is_paired_input:bool = False):
        """ initialization of Embedding class
        Args:
        Returns:
        """
        super(Embedding,self).__init__()
        self._is_pretrained = is_pretrained
        self._idx_to_vec = idx_to_vec
        self._freeze = freeze
        self._ispair = is_paired_input
        self._embedding = nn.Embedding(num_embeddings, embedding_dim, padding_idx)

        if self._is_pretrained:
            assert self._idx_to_vec is not None
            self._embedding_from_pretrained = nn.Embedding.from_pretrained(self._idx_to_vec,freeze=self._freeze)
        # Embedding output : N x L x C

    def forward(self, inputs: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]) -> Union[
        torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:

        if self._ispair:
            q1, q2 = inputs
            if self._is_pretrained:
                emb_out_q1 = self._embedding_from_pretrained(q1)
                emb_out_q2 = self._embedding_from_pretrained(q2)
            else:
                emb_out_q1 = self._embedding(q1)
                emb_out_q2 = self._embedding(q2)
            return emb_out_q1, emb_out_q2

        else:
            if self._is_pretrained:
                return self._embedding_from_pretrained(inputs)
            else:
                return self._embedding(inputs)


class LSTM(nn.Module):
    """ class for LSTM """
    def __init__(self, input_size, hidden_size, is_paired_input:bool = False, is_bidrectional:bool = False) -> None:
        super(LSTM, self).__init__()
        self._hidden_size = hidden_size
        self._ispair = is_paired_input
        self._lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, batch_first=True, bidirectional=is_bidrectional)

    def forward(self, inputs: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]) -> Union[
        torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if self._ispair:
            q1, q2 = inputs
            outputs1, hidden1 = self._lstm(q1)
            outputs2, hidden2 = self._lstm(q2)

            sen_rep1 = hidden1[0].view(-1, self._hidden_size)
            sen_rep2 = hidden2[0].view(-1, self._hidden_size)
            return sen_rep1, sen_rep2

        else:
            outputs, hidden = self._lstm(inputs)  # outputs : (batch, seq_len, num_directions * hidden_size)
            sen_rep = hidden[0].view(-1, self._hidden_size)
            return sen_rep # return last hidden state as sentence representation


class Similarity(nn.Module):
    """ class for calculating similarity between two sentence representations """
    def __init__(self):
        super(Similarity, self).__init__()

    def forward(self, inputs: Tuple[torch.Tensor, torch.Tensor]):
        ha, hb = inputs
        similarity = torch.exp(-torch.abs(ha-hb))
        return similarity


class FeatureExtractor(nn.Module):
    """ class for extracting features to be fed into classifier """
    def __init__(self):
        """ initialization of FeatureExtractor class """
        super(FeatureExtractor, self).__init__()

    def forward(self, inputs: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            inputs: tuple of sentence representations (Last hidden states of two LSTMs)
        Returns:
            feature1: element-wise (absolute) differences; |h(a)-h(b)|
            feature2: element-wise products; h(a)*h(b)
        """
        rep1, rep2 = inputs
        feature1 = torch.abs(rep1-rep2)
        feature2 = rep1 * rep2

        return feature1, feature2, torch.cat((feature1, feature2), 1)
