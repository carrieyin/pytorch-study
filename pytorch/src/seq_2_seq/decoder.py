"""
实现seq2seq的decoder部分
此处与普通的gru有区别
"""
from torch import nn


class Decoder(nn.Moudle):
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

    def forward(self, X):
        # 输出'X'的形状：(batch_size,num_steps,embed_size)
        X = self.embedding(X).permute(1, 0, 2)
