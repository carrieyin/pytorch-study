import time

import torch

from pytorch.src.seq_2_seq import config
from pytorch.src.seq_2_seq.config import device
from pytorch.src.seq_2_seq.decoder import Decoder
from pytorch.src.seq_2_seq.encoder import Encoder
from pytorch.src.seq_2_seq.encoderdecoder import EncoderDecoder
from pytorch.src.seq_2_seq.predict_seq2seq import predict_seq2seq
from pytorch.src.seq_2_seq.preprocess_data import load_data_nmt
from pytorch.src.seq_2_seq.train import train

start = time.time()
data_iter, src_vocab, tgt_vocab = load_data_nmt(config.batch_size, config.num_steps, num_examples=60000)

print("load data end")
encoder = Encoder(len(src_vocab), config.embed_size, config.hidden_size, config.num_layers)
decoder = Decoder(len(tgt_vocab), config.embed_size, config.hidden_size, config.num_layers)
net = EncoderDecoder(encoder, decoder)

lr, num_epochs = 0.01, 5
print("traing start")
train(net, data_iter, lr, num_epochs, tgt_vocab, device)
print("traing end")

print('predict start')
fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'On ne vous a pas vu depuis longtemps .']
engs = ['go .', "i lost .", 'he\'s calm .', 'We haven\'t seen you in a while .']
for fra in fras:
    output_setence, _ = predict_seq2seq(net, fra, src_vocab, tgt_vocab, config.num_steps, device)
    print('{} = > {}'.format(fra, output_setence))

end = time.time()
print("运行时间为：", end - start, '秒')