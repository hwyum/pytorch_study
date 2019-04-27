import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pad_packed_sequence, pack_sequence, pack_padded_sequence
from pprint import pprint

# 참고: https://github.com/aisolab/PyTorch_code_snippets/blob/master/RNN/torch.nn.utils.rnn.ipynb
# https://simonjisu.github.io/nlp/2018/07/05/packedsequence.html

# Example Data
data = ['hello world', 'midnight', 'calculation', 'path', 'short circuit']

char_set = ['<pad>'] + sorted(set(''.join(data)))
char2idx = {token : idx for idx, token in enumerate(char_set)}
indices = list(map(lambda string: torch.tensor([char2idx.get(char) for char in string], dtype=torch.float32), data)) # list of Tensors
pprint(indices)

# Pad Sequence
padded_indices_wbf = pad_sequence(indices, batch_first=True)
padded_indices_wobf  = pad_sequence(indices)

print(padded_indices_wbf, padded_indices_wbf.size()) # (batch, seq_len)
print(padded_indices_wobf, padded_indices_wobf.size()) # (seq_len, batch)

# Pack Sequence
# https://pytorch.org/docs/stable/nn.html#pack-sequence
sorted_indices = sorted(indices, key=lambda tensor: tensor.size()[0], reverse=True)
pprint(sorted_indices)

packed_indices = pack_sequence(sorted_indices)
print(packed_indices)
print(type(packed_indices))
