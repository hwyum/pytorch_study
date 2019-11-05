import torch
import torch.nn as nn
import torch.nn.functional as F
import gluonnlp as nlp
from typing import Union, Tuple
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence, PackedSequence
import random


class Embedding(nn.Module):
    """ class for Embedding """
    def __init__(self, num_embeddings, embedding_dim, padding_idx, is_pretrained:bool=False, idx_to_vec=None,
                 freeze:bool=False, is_paired_input:bool = False):
        """ initialization of Embedding class
        Args:
        Returns:
        """
        super(Embedding,self).__init__()
        self._is_pretrained = is_pretrained
        self._idx_to_vec = idx_to_vec
        self._freeze = freeze
        self._ispair = is_paired_input
        self._embedding = nn.Embedding(num_embeddings, embedding_dim, padding_idx)

        if self._is_pretrained:
            assert self._idx_to_vec is not None
            self._embedding_from_pretrained = nn.Embedding.from_pretrained(self._idx_to_vec,freeze=self._freeze)
        # Embedding output : N x L x C

    def forward(self, inputs: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]) -> Union[
        torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:

        if self._ispair:
            q1, q2 = inputs
            if self._is_pretrained:
                emb_out_q1 = self._embedding_from_pretrained(q1)
                emb_out_q2 = self._embedding_from_pretrained(q2)
            else:
                emb_out_q1 = self._embedding(q1)
                emb_out_q2 = self._embedding(q2)
            return emb_out_q1, emb_out_q2

        else:
            if self._is_pretrained:
                return self._embedding_from_pretrained(inputs)
            else:
                return self._embedding(inputs)


class CharacterCNN(nn.Module):
    ''' Character level CNN implementation'''

    def __init__(self, embedding_dim, kernel_size=[1,3,5], conv_channel=[50,100,150], classifier=True, class_num=2) -> None:
        '''
        Args:
            num_embedding: length of the token2index dictionary (size of the character set)
            embedding_dim: embedding dimension
            max_len: maximum length of sequences, default = 300
            classifier: whether including fc layers for classification or not
            class_num: the number of output classes, default = 2
        '''

        super(CharacterCNN, self).__init__()

        # self._embedding = nn.Embedding(num_embeddings=num_embedding, embedding_dim=embedding_dim, padding_idx=0)
        self._conv_1 = nn.Conv1d(in_channels=embedding_dim, out_channels=conv_channel[0], kernel_size=kernel_size[0])
        self._conv_2 = nn.Conv1d(in_channels=embedding_dim, out_channels=conv_channel[1], kernel_size=kernel_size[1])
        self._conv_3 = nn.Conv1d(in_channels=embedding_dim, out_channels=conv_channel[2], kernel_size=kernel_size[2])

        self._kernel_size = kernel_size
        self._classifier = classifier

        # initialization
        self.apply(self.__init_weights)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: character embedding vector (Batch, Length_word, Length_char, Embedding_dim)
        return:
        """
        Batch, Length_word, Length_char, Embedding_dim = input.size()
        input = input.reshape((Batch * Length_word, Length_char, Embedding_dim))
        input = input.permute(0, 2, 1) # N x C x L

        # Conv Layers #1~3 (Input Shape: N x C x L)
        conv_output_1 = torch.max(self._conv_1(input),2)[0]
        conv_output_2 = torch.max(self._conv_2(input),2)[0]
        conv_output_3 = torch.max(self._conv_3(input),2)[0]
        output = torch.cat((conv_output_1,conv_output_2,conv_output_3), dim=1)

        return output

    ## initialization에 대해서는 좀 더 찾아볼것!!
    def __init_weights(self, layer) -> None:
        if isinstance(layer, nn.Conv1d) or isinstance(layer, nn.Linear):
            nn.init.normal_(layer.weight, mean=0, std=0.02)


class StackingLSTM(nn.Module):
    """ stacking LSTM """
    def __init__(self, input_size, hidden_size, bidirectional:bool=False, num_layers:int=2) ->  None:
        """ initialization of stackingLSTM class
        Args:
            input_size:
            hidden_size:
            bidirectional:
            num_layers:
        Returns:

        """
        super(StackingLSTM, self).__init__()
        self._hidden_size = hidden_size
        self._lstm = nn.LSTM(input_size, hidden_size, batch_first=True, num_layers=num_layers, bidirectional=bidirectional)

    def forward(self, inputs:torch.Tensor, initial_hidden:Tuple[torch.Tensor]=None) -> Tuple[torch.Tensor, torch.Tensor]:
        # output: (batch, seq_len, num_layers * hidden_size)
        # h_n: (num_layers * num_directions, batch, hidden_size)
        outputs, (h_n, c_n) = self._lstm(inputs, initial_hidden)


        return outputs, h_n


class Maxout(nn.Module):
    """ class for Maxout Layer """
    def __init__(self, input_size, hidden_size):
        super(Maxout, self).__init__()
        self._ops_1 = nn.Linear(input_size, hidden_size)
        self._ops_2 = nn.Linear(input_size, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feature_1 = self._ops_1(x)
        feature_2 = self._ops_2(x)
        return feature_1.max(feature_2)


class LexiconEncoding(nn.Module):
    #todo: pack padded sequence implementation
    """ class for Lexcion Encoding Layer """
    def __init__(self, word_vocab:nlp.Vocab, char_vocab:nlp.Vocab, word_embedding_dim, char_embedding_dim):
        super(LexiconEncoding, self).__init__()
        self._word_embedding = Embedding(len(word_vocab), embedding_dim=word_embedding_dim, padding_idx=word_vocab.token_to_idx[word_vocab.padding_token],
                                         is_pretrained=True, idx_to_vec=torch.from_numpy(word_vocab.embedding.idx_to_vec.asnumpy()), is_paired_input=True)
        self._char_embedding = Embedding(len(char_vocab), embedding_dim=char_embedding_dim, padding_idx=word_vocab.token_to_idx[word_vocab.padding_token], is_paired_input=True)
        self._char_cnn = CharacterCNN(char_embedding_dim, kernel_size=[1,3,5], conv_channel=[50,100,150])

    def forward(self, inputs):
        p_word, p_char, h_word, h_char, label = inputs

        # word embedding
        p_word_embedding, h_word_embedding = self._word_embedding((p_word, h_word))

        # character embedding
        p_char_embedding, h_char_embedding = self._char_embedding((p_char, h_char))
        p_char_embedding = self._char_cnn(p_char_embedding).reshape(p_word.size()[0], p_word.size()[1], -1)
        h_char_embedding = self._char_cnn(h_char_embedding).reshape(h_word.size()[0], h_word.size()[1], -1)

        e_p = torch.cat((p_word_embedding, p_char_embedding), dim=2) # (Batch, Length_word, Embedding_dim)
        e_h = torch.cat((h_word_embedding, h_char_embedding), dim=2)  # (Batch, Length_word, Embedding_dim)

        return e_p, e_h


class ContextualEncoding(nn.Module):
    def __init__(self, input_size:int, hidden_size:int, bidirectional:bool=True, num_layers:int=2):
        """
        Args:
            input_size(int): input size of input vectors (Here, output embedding size of lexicon encoding layer)
            hidden_size(int): hidden size for stacking LSTM
            bidirectional(bool): whether lstm layer is bidirectional or not (default = False)
            num_layers(int): number of layers (default = 2)
        """
        super(ContextualEncoding, self).__init__()
        self._bilstm_1 = nn.LSTM(input_size, hidden_size, batch_first=True, num_layers=num_layers, bidirectional=bidirectional)
        self._bilstm_2 = nn.LSTM(hidden_size * 2, hidden_size, batch_first=True, num_layers=num_layers, bidirectional=bidirectional)
        # self._lstm = StackingLSTM(input_size, hidden_size, bidirectional, num_layers)
        self._maxout = Maxout(hidden_size * 2, hidden_size)

    # todo: Implementation of concatenation of 2 LSTM outputs
    def forward(self, inputs: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        e_p, e_h = inputs
        lstm_out_1_p = self._bilstm_1(e_p)[0]
        lstm_out_1_h = self._bilstm_1(e_h)[0]

        lstm_out_2_p = self._bilstm_2(lstm_out_1_p)[0]
        lstm_out_2_h = self._bilstm_2(lstm_out_1_h)[0]

        c_p_1 = self._maxout(lstm_out_1_p)
        c_h_1 = self._maxout(lstm_out_1_h)

        c_p_2 = self._maxout(lstm_out_2_p)
        c_h_2 = self._maxout(lstm_out_2_h)

        c_p = torch.cat((c_p_1, c_p_2), dim=2)
        c_h = torch.cat((c_h_1, c_h_2), dim=2)

        return c_p, c_h


class MemoryModule(nn.Module):
    """
    class for memory(information gathering) with dot-product attention
    """
    # todo: Implementation
    def __init__(self, hidden_size):
        super(MemoryModule, self).__init__()
        self._transform = nn.Linear(hidden_size * 2, hidden_size * 2, bias=False)
        self._dropout = nn.Dropout()
        self._bilstm = nn.LSTM(hidden_size * 6, hidden_size, batch_first=True, bidirectional=True)

    # todo: Implementation
    def forward(self, inputs):
        c_p, c_h = inputs
        c_p_hat = F.relu(self._transform(c_p))
        c_h_hat = F.relu(self._transform(c_h))

        # attention matrix
        a = self._dropout(torch.bmm(c_p_hat, c_h_hat.permute(0, 2, 1)))

        u_p = torch.cat((c_p, torch.bmm(a, c_h)), dim=2) # 4d
        u_h = torch.cat((c_h, torch.bmm(a.permute(0, 2, 1), c_p)), dim=2)

        up_cp = torch.cat((u_p, c_p), dim=2) # 6d
        uh_ch = torch.cat((u_h, c_h), dim=2)
        m_p = self._bilstm(up_cp)[0]
        m_h = self._bilstm(uh_ch)[0]

        return m_p, m_h


#todo: Implementation
class AnswerModule(nn.Module):
    def __init__(self, hidden_size, step, num_class, prediction_dropout=0.2):
        super(AnswerModule, self).__init__()
        self._step = step
        self._num_class = num_class

        self._theta_2 = nn.Parameter(torch.randn(1, hidden_size * 2)) # (1, 2d)
        self._theta_3 = nn.Parameter(torch.randn(hidden_size * 2, hidden_size * 2)) # (2d, 2d)
        # self._theta_4 = nn.Parameter(torch.randn(hidden_size * 8, num_class)) # (8d, num_class)
        self._classifier = nn.Linear(hidden_size * 8, num_class)
        self._gru = nn.GRUCell(hidden_size * 2, hidden_size * 2)

        self._prediction_dropout = prediction_dropout

    def forward(self, inputs):
        m_p, m_h = inputs
        alpha = F.softmax(self._theta_2 @ m_h.permute(0, 2, 1), dim=2) # (batch, 1, len)
        s_0 = torch.bmm(alpha, m_h) # (batch, 1, 2d)

        s_t_all = {}
        x_t_all = {}

        s_t_1 = s_0.squeeze(1) # (batch, 2d)
        for i in range(self._step):

            # find beta
            tmp = s_t_1 @ self._theta_3  # (batch, 2d) (2d, 2d) => (batch, 2d)
            tmp = torch.bmm(tmp.unsqueeze(1), m_p.permute(0, 2, 1))  # (batch, 1, 2d) (batch, 2d, len) => (batch, 1, len)
            beta = F.softmax(tmp, dim=2)

            x_t = torch.bmm(beta, m_p).squeeze(1)  # (batch, 1, 2d)
            s_t = self._gru(x_t, s_t_1)
            s_t_1 = s_t

            s_t_all[i] = s_t
            x_t_all[i] = x_t

        else:
            p_t_all = []
            for j in range(self._step):
                s_t = s_t_all[j]
                x_t = x_t_all[j]
                features = torch.cat((s_t, x_t, torch.abs(s_t-x_t), s_t * x_t), dim=1)
                # p_t = F.softmax(features @ self._theta_4, dim=1)
                p_t = self._classifier(features)
                p_t_all.append(p_t)

            # dropout은 어떻게??
            included_step_num = self._step - int(self._step * self._prediction_dropout)
            included_step = random.sample(range(self._step), included_step_num)
            p_t_all = [p_t for i, p_t in enumerate(p_t_all) if i in included_step]

            p_r = torch.mean(torch.stack(p_t_all), dim=0)

        return p_r