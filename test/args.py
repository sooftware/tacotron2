# -*- coding: utf-8 -*-
# Soohwan Kim @sooftware
# This source code is licensed under the Apache 2.0 License license found in the
# LICENSE file in the root directory of this source tree

class DefaultArgument:
    def __init__(self):
        # encoder arguments
        self.vocab_size = 10
        self.embedding_dim = 512
        self.encoder_lstm_dim = 256
        self.num_encoder_lstm_layers = 1
        self.conv_dropout_p = 0.5
        self.num_encoder_conv_layers = 3
        self.encoder_conv_kernel_size = 5
        self.encoder_lstm_bidirectional = True
        self.device = 'cuda'

        # decoder arguments
        self.num_mel_bins = 80
        self.prenet_dim = 256
        self.decoder_lstm_dim = 1024
        self.attn_lstm_dim = 1024
        self.attn_dim = 128
        self.location_conv_filter_size = 32
        self.location_conv_kernel_size = 31
        self.prenet_dropout_p = 0.5
        self.attn_dropout_p = 0.1
        self.decoder_dropout_p = 0.1
        self.max_decoding_step = 1000
        self.stop_threshold = 0.5

        # postnet arguments
        self.postnet_dim = 512
        self.num_postnet_conv_layers = 5
        self.postnet_conv_kernel_size = 5
        self.postnet_dropout_p = 0.5