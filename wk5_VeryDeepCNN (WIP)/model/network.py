import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    """ implementation of Convolutional block"""
    def __init__(self, in_channels, out_channels, shortcut:bool):
        """
        initialization of ConvBlock
        :param in_channels: number of input channels (int)
        :param out_channels: number of output channels (int)
        :param shortcut: whether or not shortcut is used
        """
        super(ConvBlock, self).__init__()

        self.shortcut = shortcut
        self.in_channels = in_channels
        self.out_channels = out_channels
        self._conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self._bn1 = nn.BatchNorm1d(num_features=out_channels)
        self._conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self._bn2 = nn.BatchNorm1d(num_features=out_channels)
        self._conv1x1 = nn.Conv1d(self.in_channels, self.out_channels, kernel_size=1)


    def forward(self, input:torch.tensor) -> torch.tensor:
        residual = input
        output1 = self._conv1(input)
        output1 = F.relu(self._bn1(output1))

        output2 = self._conv2(output1)

        if self.shortcut:
            if self.in_channels != self.out_channels:
                residual = self._conv1x1(residual)
            output2 += residual

        output_fin = F.relu(self._bn2(output2))

        return output_fin

class Flatten(nn.Module):
    """ flattening conv output to feed into FC layers"""
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, input:torch.tensor) -> torch.tensor:
        return torch.flatten(input, start_dim=1)

class Permute(nn.Module):
    """ puermutation """
    def forward(self, input:torch.tensor) -> torch.tensor:
        return input.permute(0,2,1)

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
                                         Permute(),
                                         self._tempConv,
                                         ConvBlock(64,64,True),
                                         ConvBlock(64,64,True),
                                         nn.MaxPool1d(2),
                                         ConvBlock(64,128,True),
                                         ConvBlock(128,128,True),
                                         nn.MaxPool1d(2),
                                         ConvBlock(128, 256,True),
                                         ConvBlock(256, 256,True),
                                         nn.MaxPool1d(2),
                                         ConvBlock(256, 512,True),
                                         ConvBlock(512, 512,True),
                                         nn.AdaptiveMaxPool1d(8),
                                         Flatten())

        self._fcLayers = nn.Sequential(
                                       nn.Linear(512*8, 2048),
                                       nn.Linear(2048, 2048),
                                       nn.Linear(2048, class_num))



    def forward(self, input:torch.tensor) -> torch.tensor:
        convOut = self._convLayers(input)
        fcOut = self._fcLayers(convOut)
        return fcOut

    def __init_weights(self, layer) -> None:
        return


