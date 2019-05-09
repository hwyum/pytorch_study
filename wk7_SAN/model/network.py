import torch
import torch.nn as nn
import torch.nn.functional as F
from model.modules import Embedding, BiLSTM, SelfAttention, WeightedHidden

class SAN(nn.Module):
    def __init__(self, num_embedding, embedding_dim, lstm_hidden, attn_hidden, attn_hops) -> None:
        super(SAN, self).__init__()


        self._ops = nn.Sequential(Embedding(num_embedding, embedding_dim),
                                  BiLSTM(embedding_dim, lstm_hidden),
                                  SelfAttention(2*lstm_hidden, attn_hidden, attn_hops),
                                  WeightedHidden()
                                  )


    def forward(self, inputs):
        scores = self._ops(inputs)
        return scores
