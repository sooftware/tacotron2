import torch.nn as nn
import torch.nn.init as init
from torch import Tensor


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


class PostNet(nn.Module):
    def __init__(
            self,
            n_mels: int = 80,
            postnet_dim: int = 512 ,
            num_conv_layers: int = 3,
            kernel_size: int = 5,
            dropout_p: float = 0.5
    ):
        super(PostNet, self).__init__()
        self.conv_layers = nn.ModuleList()
        self.conv_layers.append(ConvBlock(
            input_dim=n_mels,
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
            output_dim=n_mels,
            kernel_size=kernel_size,
            padding=int((kernel_size - 1) / 2),
            dropout_p=dropout_p,
            activation='tanh'
        ))

    def forward(self, inputs):
        return self.conv_layers(inputs)


class ConvBlock(nn.Module):
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
