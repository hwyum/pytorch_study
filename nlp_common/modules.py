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