import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional


class Tacotron2(nn.Module):
    """ Neural Speech-To-Text Models called Tacotron2 """
    def __init__(self, encoder, decoder):
        super(Tacotron2, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, inputs: Tensor, input_lengths: Optional[Tensor] = None):
        output = self.encoder(inputs, input_lengths)
        output = self.decoder(output)

        return output
