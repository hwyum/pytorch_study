import torch
import torch.nn as nn
import torch.nn.functional as F
from model.modules import Embedding

class SAN(nn.Module):
    def __init__(self, num_embedding, embedding_dim) -> None:
        super(SAN, self).__init__()

        self._ops = nn.Sequential(Embedding(num_embedding=num_embedding, embedding_dim=embedding_dim))


    def forward(self, *input):