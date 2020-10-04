# -*- coding: utf-8 -*-
# Soohwan Kim @sooftware
# This source code is licensed under the Apache 2.0 License license found in the
# LICENSE file in the root directory of this source tree

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
