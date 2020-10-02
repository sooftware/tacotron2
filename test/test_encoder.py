import torch
import numpy as np
from tacotron2.model.encoder import Encoder

batch_size = 3
seq_length = 3

encoder = Encoder(vocab_size=10)

inputs = torch.LongTensor(np.arange(batch_size * seq_length).reshape(batch_size, seq_length))
input_lengths = torch.LongTensor([3, 3, 2])

output = encoder(inputs, input_lengths)
print("input_lengths is not None ==")
print(output)

output = encoder(inputs)
print("\ninput_lengths is None ==")
print(output)
