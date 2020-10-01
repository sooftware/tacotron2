import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from tacotron2.model.modules import Linear


class LocationSensitiveAttention(nn.Module):
    def __init__(
            self,
            rnn_dim: int = 1024,
            embedding_dim: int = 512,
            attn_dim: int = 124,
            location_conv_filter_size: int = 32,
            location_conv_kernel_size: int = 31
    ) -> None:
        super(LocationSensitiveAttention, self).__init__()
        self.query_proj = Linear(rnn_dim, attn_dim, bias=False)
        self.valid_proj = Linear(embedding_dim, attn_dim, bias=False)
        self.score_proj = Linear(attn_dim, 1, bias=True)
        self.bias = nn.Parameter(torch.rand(attn_dim).uniform_(-0.1, 0.1))

        self.location_conv = nn.Conv1d(1, location_conv_filter_size, kernel_size=location_conv_kernel_size, bias=False)
        self.location_proj = Linear(location_conv_filter_size, attn_dim, bias=False)

    def forward(self, query: Tensor, value: Tensor, last_alignment_energy: Tensor):
        batch_size, hidden_dim, seq_length = query.size(0), query.size(2), value.size(1)

        if last_alignment_energy is None:
            last_alignment_energy = value.new_zeros(batch_size, seq_length)

        alignment_energy = self.get_alignment_energy(query, value, last_alignment_energy)
        alignment_energy = F.softmax(alignment_energy, dim=-1)

        context_vector = torch.bmm(alignment_energy.unsqueeze(1), value)
        context_vector = context_vector.squeeze(1)

        return context_vector, alignment_energy

    def get_alignment_energy(self, query: Tensor, value: Tensor, last_alignment_energy: Tensor):
        batch_size = query.size(0)
        hidden_dim = query.size(2)

        last_alignment_energy = self.location_conv(last_alignment_energy.unsqueeze(1))
        last_alignment_energy = last_alignment_energy.transpose(1, 2)
        last_alignment_energy = self.location_proj(last_alignment_energy)

        alignment_energy = self.score_proj(torch.tanh(
            self.query_proj(query.reshape(-1, hidden_dim)).view(batch_size, -1, hidden_dim)
            + self.value_proj(value.reshape(-1, hidden_dim)).view(batch_size, -1, hidden_dim)
            + last_alignment_energy
            + self.bias
        )).squeeze(-1)

        return alignment_energy

