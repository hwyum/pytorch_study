import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Union, Any

class Embedding(nn.Module):
    """ class for Embedding """
    def __init__(self, num_embedding, embedding_dim, is_pair:bool=True) -> None:
        """
        initialization of Embedding class
        :param num_embedding: size of vocab
        :param embedding_dim: output embedding dimension
        :param is_pair: whether input data is a set of paired sentences or single sentences
        """
        super(Embedding, self).__init__()
        self._is_pair = is_pair
        self._embedding = nn.Embedding(num_embeddings=num_embedding, embedding_dim=embedding_dim, padding_idx=0)

        # Embedding output : N x L x C

    def forward(self, inputs: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if self._is_pair:
            q1, q2= inputs
            emb_out_q1 = self._embedding(q1)
            emb_out_q2 = self._embedding(q2)
            return emb_out_q1, emb_out_q2

        else:
            return self._embedding(inputs)

class BiLSTM(nn.Module):
    """ class for bidirectional LSTM """
    def __init__(self, input_size, hidden_size, is_pair:bool=True) -> None:
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
            outputs, _ = self._bilstm(inputs)     # outputs : batch, seq_len, num_directions * hidden_size)
            return outputs  # output shape: Batch x seq_len x (Hidden_dim * 2)


class SelfAttention(nn.Module):
    """ class for self attention """

    # SelfAttention(2*lstm_hidden, attn_hidden, attn_hops)
    def __init__(self, in_features, hidden_units, hops, is_pair:bool=True) -> None:
        super(SelfAttention, self).__init__()
        self._is_pair = is_pair
        self._linear_1 = nn.Linear(in_features=in_features, out_features=hidden_units)
        self._linear_2 = nn.Linear(in_features=hidden_units, out_features=hops)

    def forward(self, inputs: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[Any, Any]:
        if self._is_pair:
            q1_hn, q2_hn = inputs  # batch x n x 2u

            q1_weights = F.softmax(self._linear_2(torch.tanh(self._linear_1(q1_hn))), dim=1)  # batch x n x r
            q2_weights = F.softmax(self._linear_2(torch.tanh(self._linear_1(q2_hn))), dim=1)

            q1_weights = q1_weights.permute(0, 2, 1)  # batch x r x n
            q2_weights = q2_weights.permute(0, 2, 1)  # batch x r x n

            # print("Self Attention :")
            # print("\thn_size: {}".format(q1_hn.size()))
            # print("\tweights_size: {}".format(q1_weights.size()))
            # print("\t{}".format(sum(q1_weights[0,0,:])))

            return inputs, (q1_weights, q2_weights)

        else:
            weights = F.softmax(self._linear_2(F.tanh(self._linear_1(inputs))), dim=1)
            weights = weights.permute(0, 2, 1)  # batch x r x n

            return inputs, weights


class WeightedHidden(nn.Module):
    """ class for ___ """
    def __init__(self, is_pair=True) -> None:
        super(WeightedHidden, self).__init__()
        self._is_pair = is_pair

    def forward(self, inputs: Tuple[Any, Any]):
        if self._is_pair:
            q1_hn, q2_hn = inputs[0]    # batch x n x 2u
            q1_weights, q2_weights = inputs[1]  # batch x r x n (seq_len)

            # print("WeightedHidden :")
            # print("\tweights_size:{}".format(q1_weights.size()))
            # print("\thn_size:{}".format(q1_hn.size()))

            q1_embedding = torch.bmm(q1_weights, q1_hn)   # batch x r x 2u
            q2_embedding = torch.bmm(q2_weights, q2_hn)

            return q1_embedding, q2_embedding

        else:
            hn = inputs[0]
            weights = inputs[1]
            representations = torch.bmm(weights, hn)

            return representations


class GatedEncoder(nn.Module):
    """ class for Gated Encoder (only for paired dataset)
    input matrices  : (batch x r x 2u, batch x r x 2u)
        1) multiply each row in the matrix embedding by a different weight matrix.
            for each matrix(M_h or M_p), for each batch, (r x 2u) @ (2u x 2u) => (r x 2u)
            output matrices : F_h, F_p
        2) Element-wise product of F_h and F_p
    output : (batch x r x 2u)
    """
    def __init__(self, lstm_hidden) -> None:
        super(GatedEncoder, self).__init__()
        self._w1 = nn.Parameter(torch.randn((2 * lstm_hidden, 2 * lstm_hidden)), requires_grad=True)
        self._w2 = nn.Parameter(torch.randn((2 * lstm_hidden, 2 * lstm_hidden)), requires_grad=True)

    def forward(self, inputs: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        q1_embedding, q2_embedding = inputs  # batch x r x 2u
        f1 = q1_embedding @ self._w1    # batch x r x 2u
        f2 = q2_embedding @ self._w2    # batch x r x 2u
        fr = f1 * f2    # batch x r x 2u (element-wise product)
        return fr


class Classifier(nn.Module):
    """ calss for Classification """
    def __init__(self, lstm_hidden, hops, fc_hidden, class_num) -> None:
        super(Classifier, self).__init__()
        self._linear_1 = nn.Linear(in_features=hops * 2 * lstm_hidden, out_features=fc_hidden)
        self._linear_2 = nn.Linear(in_features=fc_hidden, out_features=class_num)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        fc_out_1 = F.relu(self._linear_1(input))
        fc_out_2 = self._linear_2(fc_out_1)
        return fc_out_2


class Flatten(nn.Module):
    """ flattening output to feed into FC layers"""
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return torch.flatten(input, start_dim=1)