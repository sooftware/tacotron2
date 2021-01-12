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