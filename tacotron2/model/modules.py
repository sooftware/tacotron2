import torch.nn as nn
import torch.nn.init as init
from torch import Tensor


class ConvBlock(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, kernel_size: int, dropout_p: float = 0.5) -> None:
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, output_dim, kernel_size=kernel_size, stride=1),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
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
