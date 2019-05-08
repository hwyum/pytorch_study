import torch
import torch.nn as nn
import torch.nn.functional as F
from model.modules import Embedding, BiLSTM

class SAN(nn.Module):
    def __init__(self, num_embedding, embedding_dim, hidden_dim) -> None:
        super(SAN, self).__init__()


        self._ops = nn.Sequential(Embedding(num_embedding, embedding_dim),
                                  BiLSTM(embedding_dim, hidden_dim),
                                  )


    def forward(self, *input):