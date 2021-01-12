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

import torch
from tacotron2.attention import LocationSensitiveAttention

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
