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
from tacotron2.modules import ConvBlock


class PostNet(nn.Module):
    """

    Args:
        num_mel_filters: number of mel filters
        postnet_dim: dimension of postnet
        num_conv_layers: number of convolution layers
        kernel_size: size of convolution kernel
        dropout_p: probability of dropout
    """
    def __init__(
            self,
            num_mel_bins: int = 80,
            postnet_dim: int = 512,
            num_conv_layers: int = 3,
            kernel_size: int = 5,
            dropout_p: float = 0.5
    ):
        super(PostNet, self).__init__()

        assert num_conv_layers > 2, "PostNet num_conv_layers should be bigger than 2"

        self.conv_layers = nn.ModuleList()
        self.conv_layers.append(ConvBlock(
            input_dim=num_mel_bins,
            output_dim=postnet_dim,
            kernel_size=kernel_size,
            padding=int((kernel_size - 1) / 2),
            dropout_p=dropout_p,
            activation='tanh'
        ))

        for _ in range(num_conv_layers - 2):
            self.conv_layers.append(ConvBlock(
                input_dim=postnet_dim,
                output_dim=postnet_dim,
                kernel_size=kernel_size,
                padding=int((kernel_size - 1) / 2),
                dropout_p=dropout_p,
                activation='tanh'
            ))

        self.conv_layers.append(ConvBlock(
            input_dim=postnet_dim,
            output_dim=num_mel_bins,
            kernel_size=kernel_size,
            padding=int((kernel_size - 1) / 2),
            dropout_p=dropout_p,
            activation='tanh'
        ))

    def forward(self, x: Tensor):
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
        return x
