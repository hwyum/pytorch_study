import torch
import torch.nn as nn
import torch.nn.functional as F
from model.modules import Embedding, BiLSTM, SelfAttention

class SAN(nn.Module):
    def __init__(self, num_embedding, embedding_dim, hidden_dim) -> None:
        super(SAN, self).__init__()


        self._ops = nn.Sequential(Embedding(num_embedding, embedding_dim),
                                  BiLSTM(embedding_dim, hidden_dim),
                                  SelfAttention(2*hidden_dim, 150, 30)
                                  )


    def forward(self, *input):