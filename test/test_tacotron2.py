import torch
import numpy as np
from tacotron2.model.encoder import Encoder
from tacotron2.model.decoder import Decoder
from tacotron2.model.sublayers import PostNet
from tacotron2.model.tacotron2 import Tacotron2

encoder = Encoder(vocab_size=10)
decoder = Decoder()
postnet = PostNet()

batch_size = 3
seq_length = 3

inputs = torch.LongTensor(np.arange(batch_size * seq_length).reshape(batch_size, seq_length))
input_lengths = torch.LongTensor([3, 3, 2])
targets = torch.FloatTensor(batch_size, 100, 80).uniform_(-0.1, 0.1)

tacotron2 = Tacotron2(encoder, decoder, postnet)
output = tacotron2(inputs, targets, input_lengths)

print(output)
