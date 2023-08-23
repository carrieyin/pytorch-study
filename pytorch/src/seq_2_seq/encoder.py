"""
实现seq2seq的encoder部分
此处与普通的rnn无区别
"""
from torch import nn

from pytorch.src.seq_2_seq import config


class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.encode = nn.GRU()

