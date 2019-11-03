import torch
import torch.nn as nn
import gluonnlp as nlp
from typing import Union, Tuple
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence, PackedSequence


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

# class MaxOut(nn.Module):
#     def __init__(self, input_size, hidden_size):
#         super(MaxOut, self).__init__()
#         self._ops_1 = nn.Linear(input_size, hidden_size)
#         self._ops_2 = nn.Linear(input_size, hidden_size)
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         feature_1 = self._ops_1(x)
#         feature_2 = self._ops_2(x)
#         return feature_1.max(feature_2)


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
    def __init__(self, input_size:int, hidden_size:int, bidirectional:bool, num_layers:int):
        """
        Args:
            input_size(int): input size of input vectors (Here, output embedding size of lexicon encoding layer)
            hidden_size(int): hidden size for stacking LSTM
            bidirectional(bool): whether lstm layer is bidirectional or not (default = False)
            num_layers(int): number of layers (default = 2)
        """
        self._lstm = StackingLSTM(input_size, hidden_size, bidirectional, num_layers)
        self._maxout = Maxout(hidden_size * 2, hidden_size)

    # todo: Implementation
    def forward(self, inputs: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        p_embedding, h_embedding = inputs
        p_lstm_output = self._lstm(p_embedding)
        h_lstm_output = self._lstm(h_embedding)

        c_p = self._maxout(p_lstm_output)
        c_h = self._maxout(h_lstm_output)

        return c_p, c_h


class InformationGathering(nn.Module):
   """
   class for information gathering with dot-product attention
   """
    # todo: Implementation
    def __init__(self, input_size):
        """
        Args:
            input_size:
        """
        _transform = nn.Linear(input_size, input_size)

    # todo: Implementation
    def forward(self, inputs):
        return