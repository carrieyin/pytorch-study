"""
实现seq2seq的encoder部分
此处与普通的gru无区别
"""
from torch import nn

from pytorch.src.seq_2_seq import config


class Encoder(nn.Module):
    """用于序列到序列学习的循环神经网络 编码器"""
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.gru = nn.GRU(embed_size, num_hiddens, num_layers, dropout=dropout)

    def forward(self, X, *args):
        # input shape (batchsize, num_steps)-> (batchsize, num_steps, embedingdim)
        X = self.embedding(X)
        # 交换dim，pythorch要求batchsize位置
        X = X.permute(1, 0, 2)
        # encode编码
        # out的形状 (num_steps, batch_size, num_hiddens)
        # state的形状: (num_layers, batch_size, num_hiddens)
        output, state = self.gru(X)
        return output, state

