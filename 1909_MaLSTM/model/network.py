import torch
import torch.nn as nn
import torch.nn.functional as F
import gluonnlp as nlp
from model.modules import Embedding, LSTM, Similarity, FeatureExtractor


class MaLSTM(nn.Module):
    """ class for MaLSTM model"""
    def __init__(self, vocab:nlp.Vocab, embedding_dim, hidden_size):
        """ initialization of MaLSTM class
        Args:
            vocab: vocabulary built from gluonnlp

        Returns:
            score: (batch, num_classes)
        """
        super(MaLSTM, self).__init__()

        ptr_weight = torch.from_numpy(vocab.embedding.idx_to_vec.asnumpy())

        self._pad_idx = vocab._token_to_idx[vocab._padding_token]
        self._embedding = Embedding(len(vocab.token_to_idx), embedding_dim, self._pad_idx, is_pretrained=True, idx_to_vec=ptr_weight,
                                    freeze=False, is_paired_input=True)
        self._lstm = LSTM(embedding_dim, hidden_size, is_paired_input=True)
        self._similarity = Similarity()
        self._extractor = FeatureExtractor()
        self._classifier = nn.Linear(hidden_size*2,2)

    def forward(self, inputs):
        embedded = self._embedding(inputs)
        lstm_outputs = self._lstm(embedded)
        _,_,features = self._extractor(lstm_outputs)
        score = self._classifier(features)

        return score


