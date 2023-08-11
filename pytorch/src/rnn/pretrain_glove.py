import os.path

import torch
from torchtext import vocab

from pytorch.src.rnn.birnn_moudle import BiRNN
from pytorch.src.rnn.data_preprocess import get_tokenized, get_vocab


def getGlove():
    go = vocab.GloVe(name='6B', dim=100, cache=os.path.join("../../resources", "golve"))


def load_pretrained_embedding(words, pretrained_vocabu):
    # 初始化
    embed_word = torch.zeros(len(words), pretrained_vocabu.vectors[0].shap[0])
    count_words = 0
    for i, word in enumerate(words):
        try:
            idx = pretrained_vocabu[word]
            embed_word[i, :] = pretrained_vocabu.vectors[idx]
        except KeyError:
            count_words += 1
    return embed_word


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
    net.embedding.weight.data.copy_(load_pretrained_embedding(vocab.itos, getGlove))
    net.embedding.weight.requires_grad = False
