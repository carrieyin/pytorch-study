from torch import nn as nn


class BiRNN(nn.Module):
    def __int__(self, vocabulary, embed_len, hidden_len, num_layer):
        super(BiRNN, self).__int__()
        self.embedding = nn.Embedding()
        self.encoder = nn.LSTM(input_size=len(vocabulary),
                               embed_size=embed_len)

