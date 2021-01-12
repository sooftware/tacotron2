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
import numpy as np
from tacotron2 import Tacotron2
from .args import DefaultArgument

batch_size = 3
seq_length = 3

inputs = torch.LongTensor(np.arange(batch_size * seq_length).reshape(batch_size, seq_length))
input_lengths = torch.LongTensor([3, 3, 2])
targets = torch.FloatTensor(batch_size, 100, 80).uniform_(-0.1, 0.1)

args = DefaultArgument()
model = Tacotron2(args)
output = model(inputs, targets, input_lengths)

print(model)
print(output)
