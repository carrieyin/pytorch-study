import torch

from pytorch.src.seq_2_seq.decoder import Decoder
from pytorch.src.seq_2_seq.encoder import Encoder

vocab_size, embed_size, num_hiddens, num_layers = 10, 3, 5, 2
encoder = Encoder(vocab_size, embed_size, num_hiddens, num_layers)
encoder.eval()
x = torch.zeros((2, 5), dtype=torch.long)
outputs, hiddens = encoder(x)

decoder = Decoder(vocab_size, embed_size, num_hiddens, num_layers)
decoder.eval()
input_x = torch.zeros((2, 3), dtype=torch.long)
state = decoder.init_state(hiddens)
outputs, hiddens = decoder(input_x, hiddens)

print(outputs.shape)

