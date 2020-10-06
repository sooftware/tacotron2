import torch.nn as nn
from torch import Tensor
from tacotron2.model.modules import Linear


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