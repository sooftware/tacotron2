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
from tacotron2.decoder import Decoder

batch_size = 3
input_seq_length = 10
output_seq_length = 100
encoder_embedding_dim = 512
n_mels = 80

encoder_outputs = torch.FloatTensor(batch_size, input_seq_length, encoder_embedding_dim).uniform_(-0.1, 0.1)
decoder_inputs = torch.FloatTensor(batch_size, output_seq_length, n_mels).uniform_(-0.1, 0.1)

decoder = Decoder()
output = decoder(encoder_outputs, decoder_inputs)
print(output)
