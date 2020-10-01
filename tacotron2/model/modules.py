import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_p):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, output_dim, kernel_size=5, stride=1),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_p)
        )

    def forward(self, inputs):
        return self.conv(inputs)


class BaseRNN(nn.Module):
    """
    Applies a multi-layer RNN to an input sequence.

    Note:
        Do not use this class directly, use one of the sub classes.

    Args:
        input_dim (int): size of input
        hidden_dim (int): dimension of RNN`s hidden state vector
        num_layers (int, optional): number of RNN layers (default: 1)
        bidirectional (bool, optional): if True, becomes a bidirectional RNN (defulat: False)
        rnn_type (str, optional): type of RNN cell (default: gru)
        device (torch.device): device - 'cuda' or 'cpu'

    Attributes:
          supported_rnns = Dictionary of supported rnns
    """
    supported_rnns = {
        'lstm': nn.LSTM,
        'gru': nn.GRU,
        'rnn': nn.RNN
    }

    def __init__(
            self,
            input_dim: int,                       # size of input
            hidden_dim: int = 512,                 # dimension of RNN`s hidden state vector
            num_layers: int = 1,                   # number of recurrent layers
            rnn_type: str = 'lstm',                # number of RNN layers
            bidirectional: bool = True,            # if True, becomes a bidirectional rnn
            device: str = 'cuda'                   # device - 'cuda' or 'cpu'
    ) -> None:
        super(BaseRNN, self).__init__()
        rnn_cell = self.supported_rnns[rnn_type]
        self.rnn = rnn_cell(input_dim, hidden_dim, num_layers, True, True, bidirectional)
        self.hidden_dim = hidden_dim
        self.device = device

    def forward(self, *args, **kwargs):
        raise NotImplementedError
