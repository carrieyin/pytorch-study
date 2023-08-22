"""
实现seq2seq的encoder部分
此处与普通的rnn无区别
"""
from torch import nn

from pytorch.src.seq_2_seq import config


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.encode = nn.GRU(input_size=,num_layers=config, )

