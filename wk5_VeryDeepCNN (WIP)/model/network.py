import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    """ implementation of Convolutional block"""
    def __init__(self, in_channels, out_channels, shortcut:bool):
        super(ConvBlock, self).__init__()

        self.shortcut = shortcut
        self._conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self._bn1 = nn.BatchNorm1d(num_features=out_channels)
        self._conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self._bn2 = nn.BatchNorm1d(num_features=out_channels)


    def forward(self, input:torch.tensor) -> torch.tensor:
        residual = input
        output = F.relu(self._bn1(self._conv1(input)))
        output = self._bn2(self._conv2(output))

        if self.shortcut:
            output += residual

        return F.relu(output)

class Flatten(nn.Module):
    """ flattening conv output to feed into FC layers"""
    def forward(self, input:torch.tensor) -> torch.tensor:
        return torch.flatten(input, start_dim=1)

class VDCNN(nn.Module):
    """ implementation of VDCNN model architecture """
    def __init__(self, num_embedding:int, embedding_dim:int, class_num=2) -> None:
        """
        initialization of VDCNN class
        :param num_embedding: length of the token2index dictionary (size of the character set)
        :param embedding_dim: Embedding dimension
        :param class_num: Number of final classes (default=2)
        """

        super(VDCNN, self).__init__()

        self._embedding = nn.Embedding(num_embedding, embedding_dim, padding_idx=0)
        self._tempConv = nn.Conv1d(in_channels=embedding_dim, out_channels=64, kernel_size=3, stride=1, padding=1)

        self._convLayers = nn.Sequential(self._embedding,
                                     self._tempConv,
                                     ConvBlock(64,64,True),
                                     ConvBlock(64,64,True),
                                     nn.MaxPool1d(2),
                                     ConvBlock(128,128,True),
                                     ConvBlock(128,128,True),
                                     nn.MaxPool1d(2),
                                     ConvBlock(256, 256,True),
                                     ConvBlock(256, 256,True),
                                     nn.MaxPool1d(2),
                                     ConvBlock(512, 512,True),
                                     ConvBlock(512, 512,True),
                                     nn.AdaptiveMaxPool1d(8))

        self._fcLayers = nn.Sequential(Flatten,
                                       nn.Linear(512*8, 2048),
                                       nn.Linear(2048, 2048),
                                       nn.Linear(2048, class_num))



    def forward(self, input:torch.tensor) -> torch.tensor:
        convOut = self._convLayers(input)
        fcOut = self._fcLayers(convOut)
        return fcOut

    def __init_weights(self, layer) -> None:
        return


