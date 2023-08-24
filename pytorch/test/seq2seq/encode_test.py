import torch

from pytorch.src.seq_2_seq.encoder import Encoder

vocab_size, embed_size, num_hiddens, num_layers = 10, 3, 5, 2

encoder = Encoder(vocab_size, embed_size, num_hiddens, num_layers)

encoder.eval()
x = torch.zeros((2, 5), dtype=torch.long)
output, hidden = encoder(x)

print(output.shape, hidden.shape)