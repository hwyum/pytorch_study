import torch
import torch.nn as nn
import gluonnlp as nlp
from model.modules import Encoder, Decoder, Attention


class Seq2Seq(nn.Module):
    """ model class for seq2seq with attention """
    def __init__(self, vocab_src:nlp.Vocab, vocab_tgt:nlp.Vocab, embedding_dim:int, hidden_dim:int, num_layers:int=1,
                 bos_idx=2, eos_idx=3, use_teacher_forcing:bool=True, teacher_forcing_ratio=0.5, use_attention=True):
        """ initialization of the class """

        self._dev = torch.cuda.current_device()
        self._encoder = Encoder(vocab_src, embedding_dim, hidden_dim)
        self._decoder = Decoder(vocab_tgt, embedding_dim, hidden_dim)
        self._attn = Attention(hidden_dim)
        self._use_teacher_forcing = use_teacher_forcing
        self._teacher_forcing_ratio = teacher_forcing_ratio
        self._bos_idx = bos_idx
        self._use_attention = use_attention


    def forward(self, *inputs):
        src, tgt_in, tgt_out = inputs
        batch_size = src.size()[0]
        max_len = tgt_in.size()[1]

        encoder_out, encoder_hidden = self._encoder(src) # encoder_out : (batch, max_len, hidden_dim * 2) (BiLSTM)
        decoder_input = torch.full((batch_size, 1), self._bos_idx).long().to(self._dev) # float32 -> int64
        decoder_hidden = encoder_hidden

        if self._use_attention:
            # todo : implementation
            context_vector = self._attn()
            pass
        else:
            context_vector = encoder_hidden

        if self._use_teacher_forcing: # Teacher forcing: Feed the target as the next input --> 트레이닝 코드로.
            for di in range(max_len):
                decoder_output, next_hidden = self._decoder(decoder_input, decoder_hidden)
                # if decoder_input[:,0]



    def to(self, dev):
        super(Seq2Seq, self).to(dev)
        self._dev = dev

        return


#torch.cat([*encoder_hidden[0]], dim=1).unsqueeze(1)
