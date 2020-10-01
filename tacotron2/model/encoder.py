import torch.nn as nn
from torch import Tensor
from typing import Optional
from tacotron2.model.modules import BaseRNN, ConvBlock


class Encoder(BaseRNN):
    def __init__(
            self,
            vocab_size: int,
            embedding_dim: int = 512,
            hidden_dim: int = 256,
            num_rnn_layers: int = 1,
            rnn_type: str = 'lstm',
            conv_dropout_p: float = 0.5,
            num_conv_layers: int = 3,
            conv_kernel_size: int = 5,
            bidirectional: bool = True,
            device: str = 'cuda'
    ) -> None:
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.conv = nn.Sequential(
            *[ConvBlock(embedding_dim, embedding_dim, conv_kernel_size, conv_dropout_p) for _ in range(num_conv_layers)]
        )
        super(Encoder, self).__init__(embedding_dim, hidden_dim, num_rnn_layers, rnn_type, bidirectional, device)

    def forward(self, inputs: Tensor, input_lengths: Optional[Tensor] = None):
        """
        :param inputs: B x T x D
        :param input_lengths: B,
        :return:
        """
        inputs = self.embedding(inputs)
        inputs = inputs.transpose(1, 2)  # B x D x T

        self.rnn.flatten_parameters()

        if input_lengths is not None:
            output = nn.utils.rnn.pack_padded_sequence(inputs, input_lengths, batch_first=True)
            output, _ = self.rnn(output)
            output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)

        else:
            output = self.rnn(inputs)

        return output
