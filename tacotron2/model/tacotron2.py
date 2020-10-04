# -*- coding: utf-8 -*-
# Soohwan Kim @sooftware
# This source code is licensed under the Apache 2.0 License license found in the
# LICENSE file in the root directory of this source tree

import torch.nn as nn
from torch import Tensor
from typing import Optional
from tacotron2.model.encoder import Encoder
from tacotron2.model.decoder import Decoder
from tacotron2.model.postnet import PostNet


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
            targets: Optional[Tensor] = None,
            input_lengths: Optional[Tensor] = None,
            teacher_forcing_ratio: float = 1.0
    ):
        encoder_outputs = self.encoder(inputs, input_lengths)
        decoder_outputs = self.decoder(encoder_outputs, targets, teacher_forcing_ratio)

        postnet_outputs = self.postnet(decoder_outputs["feat_outputs"])
        decoder_outputs["feat_outputs"] += postnet_outputs

        return decoder_outputs
