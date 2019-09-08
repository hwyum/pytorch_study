import torch
import torch.nn as nn
import torch.nn.functional as F
import gluonnlp as nlp
from model.modules import Encoder, Decoder, Attention
import random

class Seq2Seq(nn.Module):
    """ model class for seq2seq with attention """
    def __init__(self, vocab_src:nlp.Vocab, vocab_tgt:nlp.Vocab, embedding_dim:int, hidden_dim:int, dev,
                 num_layers:int=1, bos_idx=2, eos_idx=3, use_attention=True):
        """ initialization of the class """
        super(Seq2Seq, self).__init__()

        self._encoder = Encoder(vocab_src, embedding_dim, hidden_dim)
        self._decoder = Decoder(vocab_tgt, embedding_dim, hidden_dim)

        self._dev = dev
        self._hidden_dim = hidden_dim
        self._bos_idx = bos_idx
        self._eos_idx = eos_idx
        self._pad_idx = self._encoder.pad_idx
        self._mask = None

        # global attention related
        self._use_attention = use_attention
        self._attn = Attention(self._hidden_dim) if self._use_attention else None

        # teacher forcing related
        self._use_teacher_forcing = None
        self._teacher_forcing_ratio = None

    def forward(self, inputs, use_teacher_forcing=True, teacher_forcing_ratio=0.5):
        src, tgt_in, tgt_out = inputs
        batch_size = src.size()[0]
        max_len = tgt_in.size()[1]
        mask = (tgt_out != self._pad_idx)

        # teacher forcing
        self._use_teacher_forcing = use_teacher_forcing
        self._teacher_forcing_ratio = teacher_forcing_ratio

        encoder_output, encoder_hidden = self._encoder(src) # encoder_out : (batch, max_len, hidden_dim * 2) (BiLSTM)
        decoder_input = torch.full((batch_size, 1), self._bos_idx).long().to(self._dev) # float32 -> int64
        decoder_hidden = encoder_hidden # initialize decoder's hidden state with encoder's last hidden state

        loss = 0
        nTotals = 0
        for di in range(max_len):
            if self._use_attention:
                decoder_hidden_top = decoder_hidden[0][0].unsqueeze(1)  # top layer's hidden state (batch_size, 1, hidden)
                context_vector = self._attn(decoder_input, decoder_hidden_top, encoder_output)[0]  # (batch, 1, hidden)
                decoder_output, next_decoder_hidden = self._decoder(decoder_input, decoder_hidden, context_vector)
            else:
                decoder_output, next_decoder_hidden = self._decoder(decoder_input, decoder_hidden)

            decoded_label = decoder_output.topk(1)[1]

            # calculate and accumulate loss
            mask_loss, nTotal = self.maskNLLLoss(decoder_output, tgt_out[:,di], mask[:,di], self._dev)
            loss += mask_loss
            nTotals += nTotal

            # Teacher forcing: Feed the target as the next input
            if self._use_teacher_forcing:
                if random.random() < self._teacher_forcing_ratio:
                    decoder_input = tgt_out[:,di].unsqueeze(-1)
                else:
                    decoder_input = decoded_label.squeeze(2)
            else:
                decoder_input = decoded_label.squeeze(2)
            decoder_hidden = next_decoder_hidden

        return loss/max_len, nTotals

    def maskNLLLoss(self, decoder_output, target, mask, dev):
        """
        Calculate average Negative Log Likelihood Loss for mini batch in one time step
        Args:
            decoder_output: (batch, 1, tgt_vocab_size)
            target: (batch, )
            mask: (batch, )
            dev: current device (cpu or gpu)
        return:
            loss:

        """
        mask = mask.unsqueeze(-1)
        nTotal = mask.sum()
        crossEntropy = -torch.log(torch.gather(decoder_output.squeeze(1), 1, target.unsqueeze(-1)).squeeze(1))
        loss = crossEntropy.masked_select(mask).mean()
        loss = loss.to(dev)
        return loss, nTotal.item()

    def to(self, dev):
        super(Seq2Seq, self).to(dev)
        self._dev = dev
        return
