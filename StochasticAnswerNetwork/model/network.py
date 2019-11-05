import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from model.modules import LexiconEncoding, ContextualEncoding, MemoryModule, AnswerModule


class StochasticAnswerNetwork(nn.Module):
    """ class for Stochastic Answer Network """
    def __init__(self, word_vocab, char_vocab, word_embedding_dim, char_embedding_dim, embedding_output_dim, hidden_size):
        super(StochasticAnswerNetwork, self).__init__()
        self.lexicon_encoder = LexiconEncoding(word_vocab, char_vocab, word_embedding_dim, char_embedding_dim)
        self.contextual_encoder = ContextualEncoding(embedding_output_dim, hidden_size)
        self.memory = MemoryModule(hidden_size)
        self.answer = AnswerModule(hidden_size, step=5, num_class=2, prediction_dropout=0.2)

    def forward(self, inputs:Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]):
        e_ph = self.lexicon_encoder(inputs)
        c_ph = self.contextual_encoder(e_ph)
        m_ph = self.memory(c_ph)
        p_r = self.answer(m_ph)

        return p_r