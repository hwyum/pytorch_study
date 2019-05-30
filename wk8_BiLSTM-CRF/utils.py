import torch
from typing import List, Tuple
from torch.nn.utils.rnn import pad_sequence

def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx.item()


def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)


# # Compute log sum exp in a numerically stable way for the forward algorithm
# def log_sum_exp(vec):
#     max_score = vec[0, argmax(vec)]
#     max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
#     return max_score + \
#         torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))

def log_sum_exp(x):
    m = torch.max(x, -1)[0]
    return m + torch.log(torch.sum(torch.exp(x - m.unsqueeze(-1)), -1))


def collate_fn(inputs: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sentence, tags, length = list(zip(*inputs))
    sentence = pad_sequence(sentence, batch_first=True, padding_value=1)
    tags = pad_sequence(tags, batch_first=True, padding_value=1)
    return sentence, tags, length


