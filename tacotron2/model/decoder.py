import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from tacotron2.model.attention import LocationSensitiveAttention
from tacotron2.model.sublayers import Linear, PreNet
from typing import Optional, Dict


class Decoder(nn.Module):
    """
    The decoder is an autoregressive recurrent neural network which predicts
    a mel spectrogram from the encoded input sequence one frame at a time.

    Args:
        n_mels: number of mel filters
        n_frames_per_step: number of frames per step. currently support just 1
        prenet_dim: dimension of prenet
        decoder_lstm_dim: dimension of decoder lstm network
        attention_lstm_dim: dimension of attention lstm network
        embedding_dim: dimension of embedding network
        attn_dim: dimension of attention layer
        location_conv_filter_size: size of location convolution filter
        location_conv_kernel_size: size of location convolution kernel
        prenet_dropout_p: dropout probability of prenet
        attn_dropout_p: dropout probability of attention network
        decoder_dropout_p: dropout probability of decoder network
        max_length: max length when inference
        stop_threshold: stop threshold

    Inputs:
        - **encoder_outputs**: tensor containing the encoded features of the input character sequences
        - **inputs**: target mel-spectrogram for training

    Returns:
        - **output**: dictionary contains feat_outputs, stop_outputs, alignment_energies
    """
    def __init__(
            self,
            n_mels: int = 80,
            n_frames_per_step: int = 1,
            prenet_dim: int = 256,
            decoder_lstm_dim: int = 1024,
            attention_lstm_dim: int = 1024,
            embedding_dim: int = 512,
            attn_dim: int = 128,
            location_conv_filter_size: int = 32,
            location_conv_kernel_size: int = 31,
            prenet_dropout_p: float = 0.5,
            attn_dropout_p: float = 0.1,
            decoder_dropout_p: float = 0.1,
            max_length: int = 1000,
            stop_threshold: float = 0.5
    ):
        super(Decoder, self).__init__()
        self.n_mels = n_mels
        self.n_frames_per_step = n_frames_per_step
        self.max_length = max_length
        self.decoder_lstm_dim = decoder_lstm_dim
        self.attention_lstm_dim = attention_lstm_dim
        self.embedding_dim = embedding_dim
        self.attn_dropout_p = attn_dropout_p
        self.decoder_dropout_p = decoder_dropout_p
        self.stop_threshold = stop_threshold

        self.context_vector = None
        self.attention_output = None
        self.attention_hidden = None
        self.alignment_energy = None
        self.alignment_energy_cum = None
        self.decoder_output = None
        self.decoder_hidden = None

        self.prenet = PreNet(self.n_mels * self.n_frames_per_step, prenet_dim, prenet_dropout_p)
        self.attention_lstm = nn.LSTMCell(
            input_size=prenet_dim + embedding_dim,
            hidden_size=attention_lstm_dim,
            bias=True
        )
        self.decoder_lstm = nn.LSTMCell(
            input_size=attention_lstm_dim + embedding_dim,
            hidden_size=decoder_lstm_dim,
            bias=True
        )
        self.attention = LocationSensitiveAttention(
            decoder_lstm_dim, embedding_dim, attn_dim, location_conv_filter_size, location_conv_kernel_size
        )
        self.feat_linear_projection = Linear(decoder_lstm_dim + embedding_dim, n_mels * n_frames_per_step)
        self.stop_linear_projection = Linear(decoder_lstm_dim + embedding_dim, 1)

    def _init_decoder_states(self, encoder_outputs):
        batch_size = encoder_outputs.size(0)
        seq_length = encoder_outputs.size(1)

        self.attention_output = encoder_outputs.new_zeros(batch_size, self.attention_lstm_dim)
        self.attention_hidden = encoder_outputs.new_zeros(batch_size, self.attention_lstm_dim)
        self.decoder_output = encoder_outputs.new_zeros(batch_size, self.decoder_lstm_dim)
        self.decoder_hidden = encoder_outputs.new_zeros(batch_size, self.decoder_lstm_dim)

        self.alignment_energy = encoder_outputs.new_zeros(batch_size, seq_length)
        self.alignment_energy_cum = encoder_outputs.new_zeros(batch_size, seq_length)
        self.context_vector = encoder_outputs.new_zeros(batch_size, self.embedding_dim)

    def parse_decoder_outputs(self, feat_outputs, stop_outputs, alignment_energies):
        stop_outputs = torch.stack(stop_outputs).transpose(0, 1).contiguous()
        alignment_energies = torch.stack(alignment_energies).transpose(0, 1)

        feat_outputs = torch.stack(feat_outputs).transpose(0, 1).contiguous()
        feat_outputs = feat_outputs.view(feat_outputs.size(0), -1, self.n_mels)
        feat_outputs = feat_outputs.transpose(1, 2)

        return {
            "feat_outputs": feat_outputs,
            "stop_outputs": stop_outputs,
            "alignment_energies": alignment_energies
        }

    def forward_step(self, input_var: Tensor, encoder_outputs: Tensor):
        input_var = torch.cat((input_var.squeeze(1), self.context_vector), dim=-1)

        self.attention_output, self.attention_hidden = self.attention_lstm(
            input_var, (self.attention_output, self.attention_hidden)
        )
        self.attention_output = F.dropout(self.attention_output, self.attn_dropout_p)

        concated_alignment_energy = torch.cat(
            (self.alignment_energy.unsqueeze(1), self.alignment_energy_cum.unsqueeze(1)), dim=1
        )
        self.context_vector, self.alignment_energy = self.attention(
            self.attention_output, encoder_outputs, concated_alignment_energy
        )
        self.alignment_energy_cum += self.alignment_energy

        input_var = torch.cat((self.attention_output, self.context_vector), dim=-1)

        self.decoder_output, self.decoder_hidden = self.decoder_lstm(input_var, (self.decoder_output, self.decoder_hidden))
        self.decoder_output = F.dropout(self.decoder_output, p=self.decoder_dropout_p)

        output = torch.cat((self.decoder_hidden, self.context_vector), dim=-1)

        feat_output = self.feat_linear_projection(output)
        stop_output = self.stop_linear_projection(output)

        return feat_output, stop_output, self.alignment_energy

    def forward(
            self,
            encoder_outputs: Tensor,
            inputs: Optional[Tensor] = None
    ):
        """
        Args:
            inputs: (batch, seq_length, n_mels)
            encoder_outputs: encoder outputs
        """
        feat_outputs = list()
        stop_outputs = list()
        alignment_energies = list()

        inputs, max_length, train = self.validate_args(inputs, encoder_outputs)
        self._init_decoder_states(encoder_outputs)

        if train:
            inputs = self.prenet(inputs)  # B x T x 256

            for di in range(max_length):
                feat_output, stop_output, alignment_energy = self.forward_step(inputs[:, di, :].unsqueeze(1), encoder_outputs)
                feat_outputs.append(feat_output)
                stop_outputs.append(stop_output)
                alignment_energies.append(alignment_energy)

        else:
            input_var = inputs

            for di in range(max_length):
                input_var = self.prenet(input_var)
                feat_output, stop_output, alignment_energy = self.forward_step(input_var, encoder_outputs)
                feat_outputs.append(feat_output)
                stop_outputs.append(stop_output)
                alignment_energies.append(alignment_energy)

                if torch.sigmoid(stop_output.item()) > self.stop_threshold:
                    break

                input_var = feat_output

        output = self.parse_decoder_outputs(feat_outputs, stop_outputs, alignment_energies)

        return output

    def validate_args(
            self,
            inputs: Optional[Tensor] = None,
            encoder_outputs: Tensor = None
    ):
        assert encoder_outputs is not None

        batch_size = encoder_outputs.size(0)

        if input is None:  # inference
            inputs = encoder_outputs.new_zeros(batch_size, self.n_mels * self.n_frames_per_step)
            max_length = self.max_length
            train = False

        else:  # training
            go_frame = encoder_outputs.new_zeros(batch_size, self.n_mels * self.n_frames_per_step).unsqueeze(1)
            inputs = inputs.view(batch_size, int(inputs.size(1) / self.n_frames_per_step), -1)

            inputs = torch.cat((go_frame, inputs), dim=1)
            train = True

            max_length = inputs.size(1) - 1

        return inputs, max_length, train
