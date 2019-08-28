import torch
import torch.nn as nn
import gluonnlp as nlp
import torch.nn.functional as F
from typing import Union, Tuple


class BiLSTM(nn.Module):
    """ class for bidirectional LSTM """
    def __init__(self, input_size, hidden_size, is_pair:bool=False) -> None:
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
            outputs, hidden = self._bilstm(inputs)     # outputs : batch, seq_len, num_directions * hidden_size)
            return outputs, hidden[0]


class StackingLSTM(nn.Module):
    """ stacking LSTM """
    def __init__(self, input_size, hidden_size, num_layers:int=2) ->  None:
        """ initialization of stackingLSTM class """
        super(StackingLSTM, self).__init__()
        self._hidden_size = hidden_size
        self._lstm = nn.LSTM(input_size, hidden_size, batch_first=True, num_layers=num_layers)

    def forward(self, inputs:torch.Tensor, initial_hidden:Tuple[torch.Tensor]=None) -> Tuple[torch.Tensor, torch.Tensor]:
        # output: (batch, seq_len, num_layers * hidden_size)
        # hidden: (num_layers, batch, hidden_size)
        outputs, hidden = self._lstm(inputs, initial_hidden)

        return outputs, hidden


class Encoder(nn.Module):
    """ encoder class """
    def __init__(self, vocab_ko:nlp.Vocab, embedding_dim:int, hidden_dim:int, padding_idx=1):
        super(Encoder,self).__init__()
        pad_idx = padding_idx

        self._embedding = nn.Embedding(len(vocab_ko.token_to_idx), embedding_dim, padding_idx=pad_idx) # NLC
        self._lstm = StackingLSTM(embedding_dim, hidden_dim)
        # self._dev = dev

    def forward(self, input_ko):
        embedded = self._embedding(input_ko)
        lstm_out, last_hidden = self._lstm(embedded)
        return lstm_out, last_hidden

    def init_hidden(self):
        return


class Decoder(nn.Module):
    """ Decoder class """
    def __init__(self, vocab_tgt:nlp.Vocab, embedding_dim:int, hidden_dim:int):
        super(Decoder,self).__init__()
        self._embedding = nn.Embedding(len(vocab_tgt.token_to_idx), embedding_dim, padding_idx=1)
        self._lstm = StackingLSTM(embedding_dim, hidden_dim)
        self._linear = nn.Linear(hidden_dim, len(vocab_tgt.token_to_idx))
        # self._softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden_state):
        """

        :param input: (batch,)
        :param hidden_state: input hidden state
        :return:
        """
        embedded = self._embedding(input) # (batch, 1, hidden)
        lstm_out, next_hidden = self._lstm(embedded, hidden_state)
        output = self._linear(lstm_out)

        # lstm_out: (batch, seq_len, hidden_size)
        # hidden: batch, num_layers * num_directions, hidden_size
        #
        return output, next_hidden


class Attention(nn.Module):
    """
    will use general score function
    * reference: https://machinetalk.org/2019/03/29/neural-machine-translation-with-attention-mechanism/
    """
    def __init__(self, hidden_dim:int, max_length:int):

        self._wa = nn.Linear(hidden_dim, hidden_dim)
        self._attn = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, decoder_inputs, decoder_hidden, encoder_outputs):
        # Dot score: h_t (dot) Wa (dot) h_s
        # encoder_output shape: (batch_size, max_len, hidden)
        # decoder_hidden shape: (batch_size, 1, hidden)
        # decoder_inputs shape: (batch_size, 1,

        # score will have shape: (batch_size, 1, max_len)
        score = torch.bmm(decoder_hidden, self._wa(encoder_outputs).permute(0,2,1))
        # alignment vector a_t
        alignment = F.softmax(score, dim=2)

        # context vector : (batch_size, 1, max_len) @ (batch_size, max_len, hidden) -> (batch, 1, hidden)
        # The context vector is what we use to compute the final output of the decoder.
        # It is the weighted average of the encoder’s output.
        context = torch.bmm(alignment, encoder_outputs)
        return context, alignment

