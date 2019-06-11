import torch
import torch.nn as nn
from utils import argmax, log_sum_exp
import mxnet
import gluonnlp as nlp
from typing import Dict
from model.modules import BiLSTM, Embedding



class BiLSTM_CRF(nn.Module):
    """ class for BiLSTM-CRF
        notation:
            batch size: B
            sequence length: L
            tag size: C
    """
    def __init__(self, vocab:nlp.Vocab, tag_to_ix:Dict, embedding_dim:int, hidden_dim:int, dev,
                 start_tag:str="<START>", stop_tag:str="<STOP>" ):
        """ initializaing the class """
        super(BiLSTM_CRF, self).__init__()
        self._embedding_dim = embedding_dim
        self._hidden_dim = hidden_dim
        self._vocab_size = len(vocab.token_to_idx)
        self._tag_to_ix = tag_to_ix
        self._tagset_size = len(tag_to_ix)
        self._dev = dev

        self._word_embeds = Embedding(self._vocab_size, self._embedding_dim, is_pretrained=True, idx_to_vec=vocab.embedding.idx_to_vec)
        self._lstm = BiLSTM(embedding_dim, self._hidden_dim // 2)
        self._hidden2tag = nn.Linear(self._hidden_dim, self._tagset_size)   # Maps the output of the LSTM into tag space.

        # Matrix of transition parameters. Entry i,j is the score of transitioning *from* i *to* j.
        self._transitions = nn.Parameter(torch.randn(self._tagset_size, self._tagset_size))

        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag
        self.START_TAG = start_tag
        self.STOP_TAG = stop_tag
        self._transitions.data[:, self._tag_to_ix[self.START_TAG]] = -10000
        self._transitions.data[self._tag_to_ix[self.STOP_TAG], :] = -10000

    def neg_log_likelihood(self, sentence, tags, mask=None):
        """ Compute the negative probability of a sequence of tags given a sequence
        :param sentence (torch.Tensor): Input sentence sequence (B, L)
        :param tags (torch.Tensor): Input tags sequence (B, L)
        :param mask (torch.Tensor): mask for valid indices
        """
        emissions = self._get_emissions(sentence)  # get emission scores (B x L x C)

        if mask is None:
            mask = torch.ones(emissions.shape[:2], dtype=torch.float)

        scores = self._compute_scores(emissions, tags, mask)  # get scores (numerator)
        partition = self._compute_log_partition(emissions, mask)    # partition (denominator)
        # print("score size: {}, {}".format(forward_score.size(), gold_score.size()))
        return torch.mean(partition - scores)  # scalar

    def _compute_scores(self, emissions:torch.Tensor, tags:torch.Tensor, mask:torch.Tensor) -> torch.Tensor:
        """Compute the scores for a given batch of emissions with their tags.
        :param emissions (torch.Tensor): (B, L, C)
        :param tags (Torch.LongTensor): (B, L)
        :param mask (Torch.FloatTensor): (B, L) (Optional)
        Returns:
            torch.Tensor: Scores for each batch.
            Shape of (B,)
        """
        batch_size, seq_len = tags.size()
        scores = torch.zeros(batch_size)

        # save first and last tags to be used later
        first_tags = tags[:, 0]
        last_valid_idx = mask.int().sum(1) - 1
        last_tags = tags.gather(1, last_valid_idx.unsqueeze(1)).squeeze()

        # Initialize: add the transition from BOS to the first tags for each batch
        t_scores = self._transitions[self._tag_to_ix[self.START_TAG], first_tags]

        # add the [unary] emission scores for the first tags for each batch
        # for all batches, the first word, see the correspondent emissions
        # for the first tags (which is a list of ids):
        # emissions[:, 0, [tag_1, tag_2, ..., tag_size]]
        e_scores = emissions[:, 0].gather(1, first_tags.unsqueeze(1)).squeeze()

        # the scores for a word is just the sum of both scores
        scores += e_scores + t_scores  # (B, )

        # now lets do this for each remaining word
        for i in range(1, seq_len):
            # we could: iterate over batches, check if we reached a mask symbol
            # and stop the iteration, but vecotrizing is faster due to gpu,
            # so instead we perform an element-wise multiplication
            is_valid = mask[:, i]

            previous_tags = tags[:, i - 1]
            current_tags = tags[:, i]

            # calculate emission and transition scores as we did before
            e_scores = emissions[:, i].gather(1, current_tags.unsqueeze(1)).squeeze()
            t_scores = self._transitions[previous_tags, current_tags]

            # apply the mask
            e_scores = e_scores * is_valid
            t_scores = t_scores * is_valid

            scores += e_scores + t_scores

        # add the transition from the end tag to the STOP_TAG tag for each batch
        scores += self._transitions[last_tags, self._tag_to_ix[self.STOP_TAG]]

        return scores

    def _compute_log_partition(self, emissions:torch.Tensor, mask:torch.Tensor=None) -> torch.Tensor:
        """
        Do the forward algorithm to compute the partition function in log-space
        :param emissions: (B, L, C)
        :param mask: (B, L)
        :return:
        """
        # ref: https://github.com/kaniblu/pytorch-bilstmcrf/blob/master/model.py
        batch_size, seq_len, tag_size = emissions.size()

        # init_alphas = torch.full((batch_size, tag_size), -10000.)  # B x C
        # init_alphas[:, self._tag_to_ix[self.START_TAG]] = 0.

        # in the first iteration, BOS will have all the scores
        alphas = emissions[:, 0] + self._transitions[self._tag_to_ix[self.START_TAG], :].unsqueeze(0)  # B x C

        # Wrap in a variable so that we will get automatic backprop
        # forward_var = init_alphas  # [B, C]
        # trans = self._transitions.unsqueeze(0)  # 1 x C x C

        for t in range(1, seq_len):  # recursion through the seq
            alpha_t = []

            for tag in range(tag_size):
                e_scores = emissions[:, t, tag]  # get emission for current tag
                e_scores = e_scores.unsqueeze(1) # broadcast emission score for all previous tags

                t_scores = self._transitions[:, tag]
                t_scores = t_scores.unsqueeze(0) # broadcast transition score for all batches

                # combine current scores with previous alphas
                # since alphas are in log space (see logsumexp below),
                # we add them instead of multiplying
                scores = e_scores + t_scores + alphas

                # add the new alphas for the current tag
                alpha_t.append(torch.logsumexp(scores, dim=1))

            # create a torch matrix from alpha_t
            new_alphas = torch.stack(alpha_t, dim=1)  # B x C

            # set alphas if the mask is valid, otherwise keep the current values
            is_valid = mask[: , t].unsqueeze(-1)
            alphas = is_valid * new_alphas + (1-is_valid) * alphas

        # add the scores for the final transition
        last_transition = self._transitions[:, self._tag_to_ix[self.STOP_TAG]]
        end_scores = alphas + last_transition

        return torch.logsumexp(end_scores, dim=1)


    def _get_emissions(self, sentence):
        """ get emission scores representing how likely is y_k given the input x_k,
        which is modeled by LSTM """
        batch_size = sentence.size(0)
        self._hidden = self._init_hidden(batch_size, self._dev)
        embeds = self._word_embeds(sentence)  # batch x seq_len x embed_dim
        lstm_out, self._hidden = self._lstm(embeds)
        # lstm_out : batch x seq_len x hidden_dim
        # hidden : [(2 x batch x hidden_dim//2), (2 x batch x hidden_dim//2)]
        emissions = self._hidden2tag(lstm_out)  # B x L x C
        return emissions

    def _init_hidden(self, batch_size, dev):
        return (torch.randn(2, batch_size, self._hidden_dim // 2).to(dev),
                torch.randn(2, batch_size, self._hidden_dim // 2).to(dev))

    def decode(self, emissions, mask=None):
        """ find the most probable sequence of labels given the emissions using the Viterbi algorithm

        :param emissions (torch.Tensor): Sequence of emissions for each label. Shape: (B, L, C)
        :param mask (torch.FloatTensor, optional): Tensor representing valid positions.
            If None, all positions are valid. Shape: (B, L)
        :return (torch.Tensor): the viterbi score for given sequences of emissions for each batch. Shape: (B, )
        :return (list of lists): the best viterbi sequence of labels for each batch
        """

        if mask is None:
            mask = torch.ones(emissions.shape[:2], dtype=torch.float)

        scores, sequences = self._viterbi_decode(emissions, mask)
        return scores, sequences

    def _viterbi_decode(self, emissions, mask):
        """ Compute the viterbi algorithm to find the most probable sequence of labels
        given a sequence of emissions
        :param emissions
        :param mask
        :return
            scores (torch.Tensor)
            sequences (list of lists)
        """
        batch_size, seq_len, tag_size = emissions.size()

        # in the first iteration, BOS will have all the scores and the, the max
        alphas = self._transitions[self._tag_to_ix[self.START_TAG]].unsqueeze(0) + emissions[:, 0]

        backpointers = []

        for i in range (1, seq_len):
            alpha_t = []
            backpointers_t = []

            for tag in range(tag_size):
                # get the emission for the current tag
                e_scores = emissions[:, i, tag]  # (B, )
                e_scores = e_scores.unsqueeze(1)  # (B, 1)

                # transition score
                t_scores = self._transitions[i, tag]
                t_scores = t_scores.unsqueeze(0)  # (B, 1)

                # combine current scores with previous alphas
                scores = e_scores + t_scores + alphas

                # find the highest score and the tag associated with it
                max_score, max_score_tag = torch.max(scores, dim=-1)

                # add the max score for the current tag
                alpha_t.append(max_score)

                # add the max_score_tag for our list of backpointers
                backpointers_t.append(max_score_tag)

            # create a torch matrix from alpha_t
            new_alphas = torch.stack(alpha_t, dim=1)  # (B, C)

            # apply mask
            is_valid = mask[:, i].unsqueeze(-1)
            alphas = is_valid * new_alphas + (1-is_valid) * alphas

            # append the new backpointers
            backpointers.append(backpointers_t)

        # add the scores for the final transition
        last_transition = self._transitions[:, self._tag_to_ix[self.STOP_TAG]]
        end_scores = alphas + last_transition.unsqueeze(0)

        # get the final most probable score and the final most probable tag
        max_final_scores, max_final_tags = torch.max(end_scores, dim=1)


        # find the best sequence of labels for each sample in the batch
        best_sequences = []
        emission_lengths = mask.int().sum(dim=1)

        for i in range(batch_size):
            sample_length = emission_lengths[i].item()

            # recover the max tag for the last timestep
            sample_final_tag = max_final_tags[i].item()

            # limit the backpointers until the last but one
            # since the last corresponds to the sample_final_tag  --> ????
            sample_backpointers = backpointers[: sample_length - 1]

            # follow the backpointers to build the sequence of labels
            sample_path = self._find_best_path(i, sample_final_tag, sample_backpointers)

            # add this path to the list of best sequences
            best_sequences.append(sample_path)

        return max_final_scores, best_sequences


    def _find_best_path(self, sample_id, best_tag, backpointers):
        """ Auxiliary function to find the best path sequence for a specific sample
        :param sample_id (int)
        :param best_tag
        :param backpointers(list of lists of tensors)
        :return:
            list of ints: a list of tag indexes representing the best path
        """

        # add the final best_tag to our best path
        best_path = [best_tag]

        # traverse the backpointers in backwards
        for backpointers_t in reversed(backpointers):

            best_tag = backpointers[best_tag][sample_id].item()
            best_path.insert(0, best_tag)

        return best_path





        #
        # # Initialize the viterbi variables in log space
        # init_vvars = torch.full((1, self.tagset_size), -10000.)
        # init_vvars[0][self.tag_to_ix[START_TAG]] = 0
        #
        # # forward_var at step i holds the viterbi variables for step i-1
        # forward_var = init_vvars
        # for feat in feats:
        #     bptrs_t = []  # holds the backpointers for this step
        #     viterbivars_t = []  # holds the viterbi variables for this step
        #
        #     for next_tag in range(self.tagset_size):
        #         # next_tag_var[i] holds the viterbi variable for tag i at the
        #         # previous step, plus the score of transitioning
        #         # from tag i to next_tag.
        #         # We don't include the emission scores here because the max
        #         # does not depend on them (we add them in below)
        #         next_tag_var = forward_var + self.transitions[next_tag]
        #         best_tag_id = argmax(next_tag_var)
        #         bptrs_t.append(best_tag_id)
        #         viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
        #     # Now add in the emission scores, and assign forward_var to the set
        #     # of viterbi variables we just computed
        #     forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
        #     backpointers.append(bptrs_t)
        #
        # # Transition to STOP_TAG
        # terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        # best_tag_id = argmax(terminal_var)
        # path_score = terminal_var[0][best_tag_id]
        #
        # # Follow the back pointers to decode the best path.
        # best_path = [best_tag_id]
        # for bptrs_t in reversed(backpointers):
        #     best_tag_id = bptrs_t[best_tag_id]
        #     best_path.append(best_tag_id)
        # # Pop off the start tag (we dont want to return that to the caller)
        # start = best_path.pop()
        # assert start == self.tag_to_ix[START_TAG]  # Sanity check
        # best_path.reverse()
        # return path_score, best_path







################## Previous Version ##################
    def forward(self, sentence):  # dont confuse this with _forward_alg above.
        # Get the emission scores from the BiLSTM
        lstm_feats = self._get_emissions(sentence) # batch x seq_len x tag_size
        print("forward lstm feats size: ", lstm_feats.size())
        scores, tag_seqs = [], []
        # Find the best path, given the features.
        for feat in lstm_feats:  # feat: seq x tag_size
            score, tag_seq = self._viterbi_decode(feat)
            tag_seq = torch.tensor(tag_seq)
            scores.append(score)
            tag_seqs.append(tag_seq)


        scores = torch.stack(scores)
        # print("scores size: ", scores.size())
        # print(type(tag_seqs), len(tag_seqs))
        # print([len(tag_seq) for tag_seq in tag_seqs])
        # tag_seqs = torch.tensor(tag_seqs)
        tag_seqs = torch.stack(tag_seqs)


        return scores, tag_seqs


    def _score_batch_sentences(self, batch_feats, batch_tags):
        batch_scores = []
        for f, t in zip(batch_feats, batch_tags):
            batch_scores.append(self._score_sentence(f, t))

        return torch.cat(batch_scores)

    def _score_sentence(self, feats, tags):
        # Gives the score of a provided tag sequence
        score = torch.zeros(1)
        tags = torch.cat([torch.tensor([self._tag_to_ix[self.START_TAG]], dtype=torch.long), tags])

        for i, feat in enumerate(feats):
            score = score + self._transitions[tags[i], tags[i + 1]] + feat[tags[i + 1]]
        score = score + self._transitions[tags[-1], self._tag_to_ix[self.STOP_TAG]]
        return score    # tag_size
