import torch
from typing import List, Tuple
from torch.nn.utils.rnn import pad_sequence


def collate_fn(inputs: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    src, tgt_input, tgt_output = list(zip(*inputs))
    src = pad_sequence(src, batch_first=True, padding_value=1)
    tgt_input = pad_sequence(tgt_input, batch_first=True, padding_value=1)
    tgt_output = pad_sequence(tgt_output, batch_first=True, padding_value=1)
    return src, tgt_input, tgt_output