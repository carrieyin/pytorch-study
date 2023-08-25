"""
实现seq2seq的decoder部分
此处与普通的gru有区别
"""
import torch
from torch import nn


class Decoder(nn.Module):
    """用于序列到序列学习的循环神经网络 解码器"""
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0, **kwargs):
        super(Decoder, self).__init__(**kwargs)
        self.embedding = nn.Embedding(vocab_size, embed_size)
        # 与普通gru区别：input_size增加num_hiddens，用于input输入解码器encode的输出
        self.rnn = nn.GRU(embed_size + num_hiddens, num_hiddens, num_layers, dropout=dropout)
        self.dense = nn.Linear(num_hiddens, vocab_size)

    def init_state(self, enc_outputs):
        # 初始化decode的hidden, 使用enc_outputs[1],enc_outputs格式(output, hidden state)
        return enc_outputs[1]

    def forward(self, X, state):
        """
        :param X:     input,        shape is (num_steps, batch_size, embed_size)
        :param state: hidden state, shape is( num_layers,batch_size, num_hiddens)
        :return:
        """
        # 输出'X'的形状：(batch_size,num_steps,embed_size)
        X = self.embedding(X).permute(1, 0, 2)
        # 广播state的0维，使它与X具有相同的num_steps的维度，方便后续拼接，输出context的shape(num_steps, batch_size, num_hiddens)
        context = state[-1].repeat(X.shape[0], 1, 1)
        # conect input and context (num_steps, batch_size, embed_size+num_hiddens)
        x_and_context = torch.cat((X, context), 2)
        # output的形状:(num_steps, batch_size, num_hiddens)
        # state的形状:(num_layers,batch_size,num_hiddens)
        output, state = self.rnn(x_and_context, state)
        # output的形状(batch_size,num_steps,vocab_size)
        output = self.dense(output).permute(1, 0, 2)

        return output, state

