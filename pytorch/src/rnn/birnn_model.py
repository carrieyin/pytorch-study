from torch import nn as nn
import torch

from pytorch.src.rnn.data_load import ImdbLoader
from pytorch.src.rnn.data_preprocess import get_tokenized, get_vocab


class BiRNN(nn.Module):
    def __init__(self, vocabulary, embed_len, hidden_len, num_layer):
        super(BiRNN, self).__init__()
        self.embedding = nn.Embedding(len(vocabulary), embed_len)
        self.encoder = nn.LSTM(input_size=embed_len,
                               hidden_size=hidden_len,
                               num_layers=num_layer,
                               bidirectional=True)

        # 本次使用起始和最终时间步的隐藏状态座位全连接层的输入
        self.decoder = nn.Linear(2*2*hidden_len, 2)

    def forward(self, inputs):
        print('rnn model py: input_shape: ', inputs.shape)
        embeddings = self.embedding(inputs)
        print('after embed input shape:', embeddings.shape)
        embeddings = embeddings.permute(1, 0, 2)
        output_sequence, _ = self.encoder(embeddings)
        concat_out = torch.cat((output_sequence[0], output_sequence[-1]), -1)
        outputs = self.decoder(concat_out)
        return outputs


if __name__ == '__main__':
    train_data = [['"dick tracy" is one of our"', 1],
                    ['arguably this is a  the )', 1],
                    ["i don't  just to warn anyone ", 0]]
    # 1.获取分词数据形式[[评论1分词1,评论1分词n], [评论i分词1， 评论i分词m].....]
    tokenized_data = get_tokenized(train_data)

    # 2. 获取分词词汇表(vocab类)
    vo = get_vocab(tokenized_data)

    # 3. 构建模型
    embed_size, hidden_size, num_layers = 100, 100, 2
    net = BiRNN(vo, embed_size, hidden_size, num_layers)

    loader = ImdbLoader('train', 3)
    data_loader = loader.get_data_loader()
    for idx, (inputs, target) in enumerate(data_loader):
        print(inputs.shape,  target.shape)

        out = net(inputs)
        print(out.shape)
        break