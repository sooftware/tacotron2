import torch.nn as nn
from torch import Tensor
from typing import Optional


class Tacotron2(nn.Module):
    """ Neural Speech-To-Text Models called Tacotron2 """
    def __init__(self, encoder: nn.Module, decoder: nn.Module, postnet: nn.Module) -> None:
        super(Tacotron2, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.postnet = postnet

    def forward(self, inputs: Tensor, input_lengths: Optional[Tensor] = None):
        encoder_outputs = self.encoder(inputs, input_lengths)
        decoder_outputs = self.decoder(encoder_outputs)

        output = self.postnet(decoder_outputs["feat_outputs"])
        output += decoder_outputs["feat_outputs"]

        return output
