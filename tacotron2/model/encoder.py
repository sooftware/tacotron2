import torch.nn as nn
from torch import Tensor
from typing import Optional
from tacotron2.model.sublayers import ConvBlock


class Encoder(nn.Module):
    """
    Encoder of Tacotron2`s Spectrogram Prediction Network.
    The encoder converts a character sequence into a hidden feature representation which the decoder
    consumes to predict a spectrogram. Default values are those in the paper.

    Args:
         vocab_size (int): size of character vocab
         embedding_dim (int): dimension of character embedding layer (default: 512)
         encoder_lstm_dim (int): dimension of rnn hidden state vector (default: 256)
         num_lstm_layers (int): number of rnn layers (default: 1)
         conv_dropout_p (float): dropout probability of convolution layer (default: 0.5)
         num_conv_layers (int): number of convolution layers (default: 3)
         conv_kernel_size (int): size of convolution layer`s kernel (default: 5)
         lstm_bidirectional (bool): if True, becomes bidirectional rnn (default: True)
         device (str): cuda or cpu (default: cuda)

    Inputs: inputs, input_lengths
        - **inputs**: list of sequences, whose length is the batch size and within which each sequence is list of tokens
        - **input_lengths**: list of sequence lengths

    Returns: output
        - **output**: tensor containing the encoded features of the input character sequences
    """

    def __init__(
            self,
            vocab_size: int,                    # size of character vocab
            embedding_dim: int = 512,           # dimension of character embedding layer
            encoder_lstm_dim: int = 256,        # dimension of lstm hidden state vector
            num_lstm_layers: int = 1,           # number of lstm layers
            conv_dropout_p: float = 0.5,        # dropout probability of convolution layer
            num_conv_layers: int = 3,           # number of convolution layers
            conv_kernel_size: int = 5,          # size of convolution layer`s kernel
            lstm_bidirectional: bool = True,    # if True, becomes bidirectional lstm
            device: str = 'cuda'                # cuda or cpu
    ) -> None:
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.conv = nn.Sequential(*[
            ConvBlock(
                embedding_dim,
                embedding_dim,
                kernel_size=conv_kernel_size,
                padding=int((conv_kernel_size - 1) / 2),
                dropout_p=conv_dropout_p
            ) for _ in range(num_conv_layers)
        ])
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=encoder_lstm_dim,
            num_layers=num_lstm_layers,
            batch_first=True,
            bias=True,
            bidirectional=lstm_bidirectional
        )
        self.device = device

    def forward(
            self,
            inputs: Tensor,                             # B x T
            input_lengths: Optional[Tensor] = None      # B,
    ) -> Tensor:
        inputs = self.embedding(inputs)

        if input_lengths is not None:
            output = nn.utils.rnn.pack_padded_sequence(inputs, input_lengths, batch_first=True)
            self.lstm.flatten_parameters()
            output, _ = self.lstm(output)
            output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)

        else:
            self.lstm.flatten_parameters()
            output, _ = self.lstm(inputs)

        return output
