import torch

from pytorch.src.seq_2_seq.decoder import Decoder
from pytorch.src.seq_2_seq.encoder import Encoder
from pytorch.src.seq_2_seq.encoderdecoder import EncoderDecoder

vocab_size, embed_size, num_hiddens, num_layers = 10, 3, 5, 2

encoder = Encoder(vocab_size, embed_size, num_hiddens, num_layers)
decoder = Decoder(vocab_size, embed_size, num_hiddens, num_layers)
encoder.eval()
decoder.eval()

input_x = torch.zeros((2, 3), dtype=torch.long)
dec_x = torch.zeros((2,7), dtype=torch.long)

en_de = EncoderDecoder(encoder, decoder)
output = en_de(input_x, dec_x)
print(output.shape)
