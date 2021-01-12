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
import torch.nn.init as init
from torch import Tensor


class ConvBlock(nn.Module):
    """
    Convolutional Block comprises of convolution, batch normalization, activation, dropout

    Args:
        input_dim: dimension of input
        output_dim: dimension of output
        kernel_size: size of convolution layer`s kernel
        padding: size of padding
        dropout_p: probability of dropout (default: rely)
        activation: activation function [relu, tanh] (default: relu)
    """
    supported_activations = {
        'relu': nn.ReLU(),
        'tanh': nn.Tanh()
    }

    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            kernel_size: int,
            padding: int,
            dropout_p: float = 0.5,
            activation='relu'
    ) -> None:
        super(ConvBlock, self).__init__()
        assert activation in ConvBlock.supported_activations, "Unsupported activation function !!"

        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, output_dim, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm1d(output_dim),
            ConvBlock.supported_activations[activation],
            nn.Dropout(p=dropout_p)
        )

    def forward(self, inputs: Tensor) -> Tensor:
        return self.conv(inputs)


class Linear(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, bias: bool = True):
        super(Linear, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=bias)
        self.linear = nn.Linear(input_dim, output_dim, bias=bias)
        init.xavier_uniform_(self.linear.weight)
        if bias:
            init.zeros_(self.linear.bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.linear(x)
