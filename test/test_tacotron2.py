# -*- coding: utf-8 -*-
# Soohwan Kim @sooftware
# This source code is licensed under the Apache 2.0 License license found in the
# LICENSE file in the root directory of this source tree

import torch
import numpy as np
from tacotron2.model.tacotron2 import Tacotron2
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
