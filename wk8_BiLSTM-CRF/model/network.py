import torch
import torch.nn as nn
from utils import argmax, log_sum_exp
import mxnet
import gluonnlp as nlp
from typing import Dict


START_TAG = "<START>"
STOP_TAG = "<STOP>"
BATCH_SIZE = 128

class BiLSTM_CRF(nn.Module):

    def __init__(self, vocab:nlp.Vocab, tag_to_ix:Dict, embedding_dim:int, hidden_dim:int):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = len(vocab.token_to_idx)
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)

        # nn.Embedding.from_pretrained(torch.from_numpy(vocab.embedding.idx_to_vec.asnumpy()), freeze=freeze,
        #                              padding_idx=self._padding_idx)
        self.word_embeds = nn.Embedding.from_pretrained(torch.from_numpy(vocab.embedding.idx_to_vec.asnumpy()),freeze=True)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
                            num_layers=1, bidirectional=True, batch_first=True)

        # Maps the output of the LSTM into tag space.
        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)   #

        # Matrix of transition parameters.  Entry i,j is the score of
        # transitioning *to* i *from* j.
        self.transitions = nn.Parameter(
            torch.randn(self.tagset_size, self.tagset_size))

        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag
        self.transitions.data[tag_to_ix[START_TAG], :] = -10000
        self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000

        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (torch.randn(2, BATCH_SIZE, self.hidden_dim // 2),
                torch.randn(2, BATCH_SIZE, self.hidden_dim // 2))

    def _forward_alg(self, feats, lens):  # fetures from LSTM: batch x seq_len x tag_size
        # ref: https://github.com/kaniblu/pytorch-bilstmcrf/blob/master/model.py
        # Do the forward algorithm to compute the partition function
        batch_size, seq_len, target_size = feats.size()
        init_alphas = torch.full((batch_size, target_size), -10000.)  # B x tag_size
        # START_TAG has all of the score.
        init_alphas[:, self.tag_to_ix[START_TAG]] = 0.

        # Wrap in a variable so that we will get automatic backprop
        forward_var = init_alphas  # [B, C]
        trans = self.transitions.unsqueeze(0)  # 1 x C x C

        for t in range(seq_len):  # recursion through the seq
            emit_score = feats[:, t].unsqueeze(2)  # B x C x 1
            next_tag_var = forward_var.unsqueeze(1) + emit_score + trans  # [B, 1, C] -> [B, C, C]
            forward_var = log_sum_exp(next_tag_var)  # [B, C, C] -> [B, C]

        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        alpha = log_sum_exp(terminal_var)
        return alpha

        # # Iterate through the sentence
        # for i, feat in enumerate(feats):  # feat : (seq_len, tag_size)
        #     alphas_t = []  # The forward tensors at this timestep
        #     for next_tag in range(self.tagset_size):
        #         # broadcast the emission score: it is the same regardless of the previous tag
        #         # The emission potential for the word at index i comes
        #         # from the hidden state of the Bi-LSTM at timestep i
        #         emit_score = feat[next_tag].view(1, -1).expand(1, self.tagset_size)  # (1 x tag_size)
        #         # the ith entry of trans_score is the score of transitioning to next_tag from i
        #         trans_score = self.transitions[next_tag].view(1, -1)  # (1 x tag_size)
        #         # The ith entry of next_tag_var is the value for the
        #         # edge (i -> next_tag) before we do log-sum-exp
        #         next_tag_var = forward_var[i] + trans_score + emit_score  # 다음 테그로 갈 확률
        #         # The forward variable for this tag is log-sum-exp of all the scores.
        #         alphas_t.append(log_sum_exp(next_tag_var).view(1))  # alphas_t: tag_size길이의 List
        #     forward_var[i] = torch.cat(alphas_t).view(1, -1)  # (1 x tag_size)
        # terminal_var = forward_var + self.transitions.unsqueeze(0)[:, self.tag_to_ix[STOP_TAG]]
        # alpha = log_sum_exp(terminal_var)
        # return alpha


    def _get_lstm_features(self, sentence):
        self.hidden = self.init_hidden()
        #embeds = self.word_embeds(sentence).view(len(sentence), 1, -1)
        embeds = self.word_embeds(sentence)  # batch x seq_len x embed_dim
        # embeds shape: seq_len x 1 x embed_dim
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        # lstm_out : batch x seq_len x hidden_dim
        # hidden : [(2 x batch x hidden_dim//2), (2 x batch x hidden_dim//2)]
        lstm_feats = self.hidden2tag(lstm_out)  # batch x seq_len x tag_size
        return lstm_feats

    def _score_batch_sentences(self, batch_feats, batch_tags):
        batch_scores = []
        for f, t in zip(batch_feats, batch_tags):
            batch_scores.append(self._score_sentence(f, t))

        return torch.cat(batch_scores)

    def _score_sentence(self, feats, tags):
        # Gives the score of a provided tag sequence
        score = torch.zeros(1)
        tags = torch.cat([torch.tensor([self.tag_to_ix[START_TAG]], dtype=torch.long), tags])

        for i, feat in enumerate(feats):
            score = score + self.transitions[tags[i + 1], tags[i]] + feats[tags[i + 1]]
        score = score + self.transitions[self.tag_to_ix[STOP_TAG], tags[-1]]
        return score

    def _viterbi_decode(self, feats):
        backpointers = []

        # Initialize the viterbi variables in log space
        init_vvars = torch.full((1, self.tagset_size), -10000.)
        init_vvars[0][self.tag_to_ix[START_TAG]] = 0

        # forward_var at step i holds the viterbi variables for step i-1
        forward_var = init_vvars
        for feat in feats:
            bptrs_t = []  # holds the backpointers for this step
            viterbivars_t = []  # holds the viterbi variables for this step

            for next_tag in range(self.tagset_size):
                # next_tag_var[i] holds the viterbi variable for tag i at the
                # previous step, plus the score of transitioning
                # from tag i to next_tag.
                # We don't include the emission scores here because the max
                # does not depend on them (we add them in below)
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
            # Now add in the emission scores, and assign forward_var to the set
            # of viterbi variables we just computed
            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)

        # Transition to STOP_TAG
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        # Follow the back pointers to decode the best path.
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)
        # Pop off the start tag (we dont want to return that to the caller)
        start = best_path.pop()
        assert start == self.tag_to_ix[START_TAG]  # Sanity check
        best_path.reverse()
        return path_score, best_path

    def neg_log_likelihood(self, sentence, tags):
        feats = self._get_lstm_features(sentence)  # seq_len x tag_size
        forward_score = self._forward_alg(feats)
        gold_score = self._score_batch_sentences(feats, tags)
        return torch.mean(forward_score - gold_score)

    def forward(self, sentence):  # dont confuse this with _forward_alg above.
        # Get the emission scores from the BiLSTM
        lstm_feats = self._get_lstm_features(sentence)

        scores, tag_seqs = [], []
        # Find the best path, given the features.
        for feat in lstm_feats:
            score, tag_seq = self._viterbi_decode(feat)
            scores.append(score)
            tag_seqs.append(tag_seq)

        scores = torch.stack(scores)
        tag_seqs = torch.stack(tag_seqs)

        return scores, tag_seqs

