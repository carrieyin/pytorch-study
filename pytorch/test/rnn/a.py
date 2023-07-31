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
    vb = vocab.vocab(counter, min_freq=2)
    return vb

def testOrderedDict(etl_data):
    counter = collections.Counter(_flatten(etl_data))
    print('counter:------------ ', counter)
    count2 = collections.Counter([item for token_item in etl_data for item in token_item])
    print('counter2-------------', count2)

    sortlist  = sorted(counter.items())
    order_dict = collections.OrderedDict(sortlist)
    print('order dict:', order_dict)

    vb = vocab.vocab(count2, min_freq=2)
    print('vb~', len(vb))

    v1= vocab.vocab(order_dict)
    print(type(v1), len(v1))


def testCounter(etl_data):
    vocabu = torchtext.vocab.vocab(collections.Counter([item for token_item in etl_data for item in token_item]))
    print('test', len(vocabu))
    #v = torchtext.vocab.vocab(counter)
    #print('v--!!:',v)


def pad(x, max_len):
    if len(x) > max_len:
        return x[:max_len]
    else:
        return x + [0] * (max_len - len(x))


def pad_token_words(token_data, pad_len):
    return [pad(words, pad_len) for words in token_data]


if __name__ == '__main__':
    # 1. 准备数据
    train_data = [['"dick tracy" is one of our"', 1],
                  ['arguably this is a  the )', 1],
                  ["i don't  just to warn anyone ", 0]]
    #2. 清洗数据
    etl_data = tokenized_data(train_data)
    print(etl_data)
    # job1 处理数据
    #testOrderedDict(etl_data)


    v = getvocabu(etl_data)
    # 对应词表中的索引
    # pad成10的长度
    pad_value = pad_token_words(etl_data, 10)
    print(pad_value)

    #features = torch.tensor()
    #print('feature', features)

