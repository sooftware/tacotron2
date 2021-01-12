# Copyright (c) 2020, Soohwan Kim. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch.nn as nn
from torch import Tensor
from typing import Optional
from tacotron2.encoder import Encoder
from tacotron2.decoder import Decoder
from tacotron2.postnet import PostNet


class Tacotron2(nn.Module):
    """ Neural Speech-To-Text Models called Tacotron2 """
    def __init__(self, args) -> None:
        super(Tacotron2, self).__init__()
        self.encoder = Encoder(
            vocab_size=args.vocab_size,
            embedding_dim=args.embedding_dim,
            encoder_lstm_dim=args.encoder_lstm_dim,
            num_lstm_layers=args.num_encoder_lstm_layers,
            conv_dropout_p=args.conv_dropout_p,
            num_conv_layers=args.num_encoder_conv_layers,
            conv_kernel_size=args.encoder_conv_kernel_size,
            lstm_bidirectional=args.encoder_lstm_bidirectional,
            device=args.device
        )
        self.decoder = Decoder(
            num_mel_bins=args.num_mel_bins,
            prenet_dim=args.prenet_dim,
            decoder_lstm_dim=args.decoder_lstm_dim,
            attn_lstm_dim=args.attn_lstm_dim,
            embedding_dim=args.embedding_dim,
            attn_dim=args.attn_dim,
            location_conv_filter_size=args.location_conv_filter_size,
            location_conv_kernel_size=args.location_conv_kernel_size,
            prenet_dropout_p=args.prenet_dropout_p,
            attn_dropout_p=args.attn_dropout_p,
            decoder_dropout_p=args.decoder_dropout_p,
            max_decoding_step=args.max_decoding_step,
            stop_threshold=args.stop_threshold
        )
        self.postnet = PostNet(
            num_mel_bins=args.num_mel_bins,
            postnet_dim=args.postnet_dim,
            num_conv_layers=args.num_postnet_conv_layers,
            kernel_size=args.postnet_conv_kernel_size,
            dropout_p=args.postnet_dropout_p
        )

    def forward(
            self,
            inputs: Tensor,
            input_lengths: Optional[Tensor] = None,
            targets: Optional[Tensor] = None,
            teacher_forcing_ratio: float = 1.0
    ):
        encoder_outputs = self.encoder(inputs, input_lengths)
        decoder_outputs = self.decoder(encoder_outputs, targets, teacher_forcing_ratio)

        postnet_outputs = self.postnet(decoder_outputs["mel_outputs"])
        decoder_outputs["mel_outputs"] += postnet_outputs

        return decoder_outputs
