from pytorch.src.d2l.torch import d2l
from pytorch.src.seq2seq.decoder import Seq2SeqDecoder
from pytorch.src.seq2seq.encode import Seq2SeqEncoder
from pytorch.src.seq2seq.encoderdecoder import EncoderDecoder
from pytorch.src.seq2seq.train import train_seq2seq

embed_size, num_hiddens, num_layers, dropout = 32, 32, 2, 0.1
batch_size, num_steps = 64, 10
lr, num_epochs, device = 0.005, 300, d2l.try_gpu()

train_iter, src_vocab, tgt_vocab = d2l.load_data_nmt(batch_size, num_steps)
encoder = Seq2SeqEncoder(len(src_vocab), embed_size, num_hiddens, num_layers,
                        dropout)
decoder = Seq2SeqDecoder(len(tgt_vocab), embed_size, num_hiddens, num_layers,
                        dropout)
net = EncoderDecoder(encoder, decoder)
train_seq2seq(net, train_iter, lr, num_epochs, tgt_vocab, device)