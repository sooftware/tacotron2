import torch
from tacotron2.model.decoder import Decoder

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
