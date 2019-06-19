import torch
import torch.nn as nn
import torch.nn.functional as F
from gluonnlp import Vocab
from model.modules import Embedding

class movieCNN(nn.Module):
    ''' CNN Network implementation'''
    def __init__(self, vocab:Vocab, class_num):
        """Instantiating movieCNN class
        Args:
            vocab : vocabulary built from gluonnlp
            class_num

        """

        super(movieCNN, self).__init__()

        ptr_weight = torch.from_numpy(vocab.embedding.idx_to_vec.asnumpy())

        # static : non-trainable
        self.st_embed = Embedding(num_embedding=len(vocab), embedding_dim=300, is_pretrained=True,
                                  idx_to_vec=ptr_weight, freeze=False)
        # non-static : trainable
        self.non_st_embed = Embedding(num_embedding=len(vocab), embedding_dim=300, is_pretrained=True,
                                  idx_to_vec=ptr_weight, freeze=True)

        # CNN
        self.conv_w3 = nn.Conv1d(in_channels=300, out_channels=100, kernel_size=3)
        self.conv_w4 = nn.Conv1d(in_channels=300, out_channels=100, kernel_size=4)
        self.conv_w5 = nn.Conv1d(in_channels=300, out_channels=100, kernel_size=5)

        # Dropout
        self.dropout = nn.Dropout(p=0.5)

        # Fully connected
        self.linear = nn.Linear(300, class_num)

        # initialization
        self.apply(self.__init_weights)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        static_batch  = self.st_embed(x) # shape : batch * seq_len * embedding_dim
        static_batch = static_batch.permute(0, 2, 1) # for Conv1d

        non_static_batch = self.non_st_embed(x)
        non_static_batch = non_static_batch.permute(0, 2, 1)

        # Conv1d
        conv_w3_output = F.relu(self.conv_w3(static_batch)) + F.relu(self.conv_w3(non_static_batch)) # N x 100 x (L_out*2)
        conv_w4_output = F.relu(self.conv_w4(static_batch)) + F.relu(self.conv_w4(non_static_batch)) # N x 100 x (L_out*2)
        conv_w5_output = F.relu(self.conv_w5(static_batch)) + F.relu(self.conv_w5(non_static_batch)) # N x 100 x (L_out*2)

        # Max-over-time pooling
        conv_w3_output = torch.max(conv_w3_output, 2)[0] # torch.max() -> (Tensor, LongTensor): the result tuple of two output tensors (max, max_indices)
        conv_w4_output = torch.max(conv_w4_output, 2)[0]
        conv_w5_output = torch.max(conv_w5_output, 2)[0]
        conv_output = torch.cat((conv_w3_output, conv_w4_output, conv_w5_output), dim=1)

        # Dropout
        conv_output = self.dropout(conv_output)

        # Linear
        output = self.linear(conv_output)

        return output

    def __init_weights(self, layer) -> None:
        if isinstance(layer, nn.Conv1d):
            nn.init.kaiming_uniform_(layer.weight)
        elif isinstance(layer, nn.Linear):
            nn.init.xavier_normal_(layer.weight)