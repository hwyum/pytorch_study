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
        self.pad_idx = padding_idx

        self._embedding = nn.Embedding(len(vocab_ko.token_to_idx), embedding_dim, padding_idx=self.pad_idx) # NLC
        self._lstm = StackingLSTM(embedding_dim, hidden_dim)
        # self._dev = dev

    def forward(self, input_ko):
        embedded = self._embedding(input_ko)
        lstm_out, last_hidden = self._lstm(embedded) # last_hidden: (num_layers * num_directions, batch, hidden_size)
        return lstm_out, last_hidden

    def init_hidden(self):
        return


class Decoder(nn.Module):
    """ Decoder class """
    def __init__(self, vocab_tgt:nlp.Vocab, embedding_dim:int, hidden_dim:int):
        """ Initialization of Decoder """
        super(Decoder,self).__init__()
        self._embedding = nn.Embedding(len(vocab_tgt.token_to_idx), embedding_dim, padding_idx=1)
        self._lstm = StackingLSTM(embedding_dim, hidden_dim)
        self._wc = nn.Linear(hidden_dim * 2, hidden_dim)
        self._linear = nn.Linear(hidden_dim, len(vocab_tgt.token_to_idx))
        # self._softmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, hidden_state, context_vector=None):
        """
        Args:
            inputs: (batch,)
            hidden_state: decoder hidden state in current time step
            context_vector:

        Returns:
            output: decoder output for the current time step
            next_decoder_hidden: decoder hidden state for next time step
        """
        embedded = self._embedding(inputs) # (batch, 1, hidden)
        lstm_out, next_decoder_hidden = self._lstm(embedded, hidden_state)
        next_hidden, next_cell_state = next_decoder_hidden

        if context_vector is not None:
            context_hidden_concat = torch.cat((context_vector, next_hidden[-1].unsqueeze(1)), dim=2)
            attentional_hidden_state = torch.tanh(self._wc(context_hidden_concat))
            output = self._linear(attentional_hidden_state)
            output = F.softmax(output, dim=2)

        else:
            output = self._linear(lstm_out)
            output = F.softmax(output, dim=2)

        # lstm_out: (batch, 1, hidden_size)
        # hidden: batch, num_layers * num_directions, hidden_size
        #
        return output, next_decoder_hidden


class Attention(nn.Module):
    """
    will use general score function
    * reference: https://machinetalk.org/2019/03/29/neural-machine-translation-with-attention-mechanism/
    """
    def __init__(self, hidden_dim:int):
        super(Attention, self).__init__()
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


# Luong attention layer
# https://pytorch.org/tutorials/beginner/chatbot_tutorial.html#decoder
class Attn(nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()
        self.method = method
        if self.method not in ['dot', 'general', 'concat']:
            raise ValueError(self.method, "is not an appropriate attention method.")
        self.hidden_size = hidden_size
        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, hidden_size)
        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
            self.v = nn.Parameter(torch.FloatTensor(hidden_size))

    def dot_score(self, hidden, encoder_output):
        return torch.sum(hidden * encoder_output, dim=2)

    def general_score(self, hidden, encoder_output):
        energy = self.attn(encoder_output)
        return torch.sum(hidden * energy, dim=2)

    def concat_score(self, hidden, encoder_output):
        energy = self.attn(torch.cat((hidden.expand(encoder_output.size(0), -1, -1), encoder_output), 2)).tanh()
        return torch.sum(self.v * energy, dim=2)

    def forward(self, hidden, encoder_outputs):
        # Calculate the attention weights (energies) based on the given method
        if self.method == 'general':
            attn_energies = self.general_score(hidden, encoder_outputs)
        elif self.method == 'concat':
            attn_energies = self.concat_score(hidden, encoder_outputs)
        elif self.method == 'dot':
            attn_energies = self.dot_score(hidden, encoder_outputs)

        # Transpose max_length and batch_size dimensions
        attn_energies = attn_energies.t()

        # Return the softmax normalized probability scores (with added dimension)
        return F.softmax(attn_energies, dim=1).unsqueeze(1)

# https://pytorch.org/tutorials/beginner/chatbot_tutorial.html#decoder
class LuongAttnDecoderRNN(nn.Module):
    def __init__(self, attn_model, embedding, hidden_size, output_size, n_layers=1, dropout=0.1):
        super(LuongAttnDecoderRNN, self).__init__()

        # Keep for reference
        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout

        # Define layers
        self.embedding = embedding
        self.embedding_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=(0 if n_layers == 1 else dropout))
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

        self.attn = Attn(attn_model, hidden_size)

    def forward(self, input_step, last_hidden, encoder_outputs):
        # Note: we run this one step (word) at a time
        # Get embedding of current input word
        embedded = self.embedding(input_step)
        embedded = self.embedding_dropout(embedded)
        # Forward through unidirectional GRU
        rnn_output, hidden = self.gru(embedded, last_hidden)
        # Calculate attention weights from the current GRU output
        attn_weights = self.attn(rnn_output, encoder_outputs)
        # Multiply attention weights to encoder outputs to get new "weighted sum" context vector
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))
        # Concatenate weighted context vector and GRU output using Luong eq. 5
        rnn_output = rnn_output.squeeze(0)
        context = context.squeeze(1)
        concat_input = torch.cat((rnn_output, context), 1)
        concat_output = torch.tanh(self.concat(concat_input))
        # Predict next word using Luong eq. 6
        output = self.out(concat_output)
        output = F.softmax(output, dim=1)
        # Return output and final hidden state
        return output, hidden