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

# pad_packed_sequence
# https://pytorch.org/docs/stable/nn.html#pad-packed-sequence
padded_indices_wbf_tuple = pad_packed_sequence(packed_indices, batch_first=True)
padded_indices_wobf_tuple = pad_packed_sequence(packed_indices, batch_first=False)

print(padded_indices_wbf_tuple)
print(padded_indices_wobf_tuple)

# Pack_padded_sequence
# https://pytorch.org/docs/stable/nn.html#pack-padded-sequence
pack_padded_indices_wbf = pack_padded_sequence(*padded_indices_wbf_tuple, batch_first=True)
pack_padded_indices_wobf = pack_padded_sequence(*padded_indices_wobf_tuple, batch_first=False)
print(pack_padded_indices_wbf)
print(pack_padded_indices_wobf)
print(type(pack_padded_indices_wbf))


# ArgSort Test
from pprint import pprint
data = ['hello world', 'midnight', 'calculation', 'path', 'short circuit']
char_set = ['<pad>'] + sorted(set(''.join(data)))
char2idx = {token : idx for idx, token in enumerate(char_set)}
indices = list(map(lambda string: torch.tensor([char2idx.get(char) for char in string], dtype=torch.float32), data)) # list of Tensors
pprint(indices)
len_indicies = [11, 8, 11, 4, 13]
padded_indices_wbf = pad_sequence(indices, batch_first=True)
sort = torch.argsort(torch.tensor(len_indicies), descending=True)
pack_padded = pack_padded_sequence(padded_indices_wbf[sort], torch.tensor(len_indicies)[sort], batch_first=True)
