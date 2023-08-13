"""
测试nn.embeding
"""
import torch
from torch import nn as nn

from pytorch.src.rnn.data_preprocess import get_tokenized, get_vocab


# pad数据，方便bath操作
def pad(x, max_len):
    if len(x) > max_len:
        return x[:max_len]
    else:
        return x + [0] * (max_len - len(x))

def embed_test(train_data):

    # 1.获取分词数据形式[[评论1分词1,评论1分词n], [评论i分词1， 评论i分词m].....]
    tokenized_data = get_tokenized(train_data)
    print(tokenized_data)

    # 2. 获取分词词汇表(vocab类)
    vo = get_vocab(tokenized_data)

    # 3. 分词索引表
    pad_token_words = [pad([vo[item] for item in items], 10) for items in tokenized_data]
    print(pad_token_words)

    tokenized_tensor = torch.tensor(pad_token_words)
    print(tokenized_tensor.shape)

    dim = 5
    embed_mod = nn.Embedding(len(vo), dim)
    embed_out = embed_mod(tokenized_tensor)
    print(embed_out.shape)


if __name__ == '__main__':
    train_data = [['"dick tracy" is one of our"', 1],
                  ['arguably this is a  the )', 1],
                  ["i don't  just to warn anyone ", 0]]
    embed_test(train_data)