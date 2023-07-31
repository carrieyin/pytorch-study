import torch
import torchtext
import collections
from torchtext import vocab
from tkinter import _flatten


def tokenized_data(data):
    def tokenizer(text):
        return [item.lower() for item in text.split(" ")]
    return [tokenizer(context) for context, _ in data]


def getvocabu(etl_data):
    counter = collections.Counter(_flatten(etl_data))
    vb = vocab.vocab(counter)
    return vb

def pad(x, max_len):
    if len(x) > max_len:
        return x[:max_len]
    else:
        return x + [0] * (max_len - len(x))


def pad_token_words(token_data, pad_len):
    return [pad(words, pad_len) for words in token_data]


def reflex_to_indices(data, v):
    vlist = []
    for its in data:
        vitem = []
        for it in its:
            index = v[it]
            vitem.append(index)
        vlist.append(vitem)

    index_pad_list = pad_token_words(vlist, 10)
    return index_pad_list


if __name__ == '__main__':
    # 1. 准备数据
    train_data = [['"dick tracy" is one of our"', 1],
                  ['arguably this is a  the )', 1],
                  ["i don't  just to warn anyone ", 0]]
    # 2. 分词
    etl_data = tokenized_data(train_data)
    print(etl_data)

    # 3. 创建词表
    v = getvocabu(etl_data)
    print('v size:', len(v))

    # 4. 分词 词汇表映射
    #refelex_data = reflex_to_indices(etl_data, v)
    #print(refelex_data)
    reflex_data = [pad([v[item] for item in items], 10) for items in etl_data]
    print(reflex_data)
    features = torch.tensor(reflex_data)
    print(features)
