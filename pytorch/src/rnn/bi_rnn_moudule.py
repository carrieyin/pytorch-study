import torch.nn as nn
class BiRNN(nn.Module):
    def __int__(self, vocab, embed_size, num_hiddens, num_layers):
        super(BiRNN, self).__int__()

        # 词向量
        self.embedding = nn.Embedding(len(vocab), embed_size)
        #bidirectional true即为双向LSTM
        self.encoder = nn.LSTM(input_size=embed_size,
                                hidden_size=num_hiddens,
                                num_layers=num_layers,
                                bidirectional=True)
        self.decoder = nn.Linear(4* num_hiddens, 2)

    def forward(self, inputs):
        pass