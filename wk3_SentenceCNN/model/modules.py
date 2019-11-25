import torch
import torch.nn as nn
import torch.nn.functional as F

class Embedding(nn.Module):
    """ class for Embedding """
    def __init__(self, num_embedding:int, embedding_dim:int,
                 is_pretrained:bool=False, idx_to_vec=None, freeze:bool=True) -> None:
        """
        initialization of Embedding class
        :param num_embedding: size of vocab
        :param embedding_dim: output embedding dimension
        :param is_pretrained: whether creating embedding vector from pretrained weights
        :param idx_to_vec: (only for is_pretrained=True) idx_to_vec
        :param freeze: (only used when is_pretrained=True) whether pre-trained weight is trainable or not
        """
        super(Embedding, self).__init__()
        self._embedding = nn.Embedding(num_embeddings=num_embedding, embedding_dim=embedding_dim, padding_idx=0)
        self._is_pretrained = is_pretrained
        self._idx_to_vec = idx_to_vec
        self._freeze = freeze

        if self._is_pretrained:
            assert self._idx_to_vec is not None
            self._embedding_from_pretrained = nn.Embedding.from_pretrained(self._idx_to_vec, freeze=self._freeze)

        # Embedding output : N x L x C

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self._is_pretrained:
            return  self._embedding_from_pretrained(inputs)

        return self._embedding(inputs)