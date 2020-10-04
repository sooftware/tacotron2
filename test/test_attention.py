# -*- coding: utf-8 -*-
# Soohwan Kim @sooftware
# This source code is licensed under the Apache 2.0 License license found in the
# LICENSE file in the root directory of this source tree

import torch
from tacotron2.model.attention import LocationSensitiveAttention

batch_size = 3
seq_length = 100
query_dim = 1024
value_dim = 512
align_dim = 2

query = torch.FloatTensor(batch_size, 1, query_dim).uniform_(-0.01, 0.01)
value = torch.FloatTensor(batch_size, seq_length, value_dim).uniform_(-0.01, 0.01)
align = torch.FloatTensor(batch_size, seq_length, align_dim).uniform_(-0.01, 0.01)

attention = LocationSensitiveAttention()
output = attention(query, value, align)
print(output)
