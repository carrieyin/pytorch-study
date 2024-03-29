import torch

from pytorch.src.seq_2_seq import config
from pytorch.src.seq_2_seq.config import device
from pytorch.src.seq_2_seq.decoder import Decoder
from pytorch.src.seq_2_seq.encoder import Encoder
from pytorch.src.seq_2_seq.encoderdecoder import EncoderDecoder
from pytorch.src.seq_2_seq.preprocess_data import load_data_nmt
from pytorch.src.seq_2_seq.train import train

data_iter, src_vocab, tgt_vocab = load_data_nmt(config.batch_size, config.num_steps, num_examples=600)

encoder = Encoder(len(src_vocab), config.embed_size, config.hidden_size, config.num_layers)
decoder = Decoder(len(tgt_vocab), config.embed_size, config.hidden_size, config.num_layers)
net = EncoderDecoder(encoder, decoder)

lr, num_epochs = 0.01, 5
train(net, data_iter, lr, num_epochs, tgt_vocab, device)