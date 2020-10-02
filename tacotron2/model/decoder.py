import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from tacotron2.model.attention import LocationSensitiveAttention
from tacotron2.model.modules import Linear
from typing import Optional


class PreNet(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, dropout_p: float) -> None:
        super(PreNet, self).__init__()
        self.fully_connected_layers = nn.Sequential(
            Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            Linear(output_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_p)
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.fully_connected_layers(inputs)


class Decoder(nn.Module):
    """
    The decoder is an autoregressive recurrent neural network which predicts
    a mel spectrogram from the encoded input sequence one frame at a time.
    """
    def __init__(
            self,
            n_mels: int = 80,
            n_frames_per_step: int = 1,
            prenet_dim: int = 256,
            dropout_p: float = 0.5,
            decoder_lstm_dim: int = 1024,
            attention_lstm_dim: int = 1024,
            embedding_dim: int = 512,
            attn_dim: int = 128,
            location_conv_filter_size: int = 32,
            location_conv_kenel_size: int = 31,
            attn_dropout_p: float = 0.1,
            decoder_dropout_p: float = 0.1,
            max_length: int = 1000
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

        self.context_vector = None
        self.attention_output = None
        self.attention_hidden = None
        self.alignment_energy = None
        self.alignment_energy_cum = None
        self.decoder_output = None
        self.decoder_hidden = None

        self.prenet = PreNet(self.n_mels * self.n_frames_per_step, prenet_dim, dropout_p)
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
            decoder_lstm_dim, embedding_dim, attn_dim, location_conv_filter_size, location_conv_kenel_size
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
        stop_outputs = torch.stack(stop_outputs).t().contiguous()
        alignment_energies = torch.stack(alignment_energies).t()

        feat_outputs = torch.stack(feat_outputs).t().contiguous()
        feat_outputs = feat_outputs.view(feat_outputs.size(0), -1, self.n_mels)
        feat_outputs = feat_outputs.transpose(1, 2)

        return {
            "feat_outputs": feat_outputs,
            "stop_outputs": stop_outputs,
            "alignment_energies": alignment_energies
        }

    def forward_step(self, input_var: Tensor, encoder_outputs: Tensor):
        input_var = torch.cat((input_var, self.context_vector), dim=-1)

        self.attention_output, self.attention_hidden = self.attention_lstm(
            input_var, (self.attention_output, self.attention_hidden)
        )
        self.attention_output = F.dropout(self.attention_output, self.attention_dropout_p)

        concated_alignment_energy = torch.cat(
            (self.alignment_energy.unsqueeze(1), self.alignment_energy_cum.unsqueeze(1)), dim=-1
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

    def forward(self, inputs, encoder_outputs):
        """
        Args:
            inputs: (batch, seq_length, n_mels)
            encoder_outputs: encoder outputs
        """
        feat_outputs = list()
        stop_outputs = list()
        alignment_energies = list()

        inputs, max_length = self.validate_args(inputs, encoder_outputs)
        self._init_decoder_states(encoder_outputs)

        inputs = self.prenet(inputs)  # B x T x 256

        for di in range(max_length):
            feat_output, stop_output, alignment_energy = self.forrward_step(inputs[di])
            feat_outputs.append(feat_output)
            stop_outputs.append(stop_output)
            alignment_energies.append(alignment_energy)

        output = self.parse_decoder_outputs(feat_outputs, stop_outputs, alignment_energies)

        return output

    def validate_args(
            self,
            inputs: Optional[Tensor] = None,
            encoder_outputs: Tensor = None
    ):
        assert encoder_outputs is not None
        batch_size = encoder_outputs.size(0)
        seq_length = encoder_outputs.size(1)

        if input is None:  # inference
            inputs = encoder_outputs.new_zeros(batch_size, self.n_mels * self.n_frames_per_step)
            max_length = self.max_length

        else:  # training
            go_frame = encoder_outputs.new_zeros(batch_size, self.n_mels * self.n_frames_per_step)
            inputs = inputs.view(batch_size, int(seq_length / self.n_frames_per_step), -1)
            inputs = torch.cat((go_frame, inputs), dim=1)

            max_length = inputs.size(1) - 1

        return inputs, max_length
