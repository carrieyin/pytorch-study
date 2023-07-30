import torch
import torchtext
import collections

from torchtext import vocab
from tkinter import _flatten

print(vocab)
# def tokenizer(content):
#     return [item for item in content.split(' ')]


def testVu():
    # 构建词汇表
    tokenizer = lambda x: x.split()  # 分词函数，这里简单地按空格划分单词
    train_data = ['apple banana', 'banana', 'orange grape']
    vu = torchtext.vocab.build_vocab_from_iterator(map(tokenizer, train_data))
    # 使用Vocab对象将单词映射为索引
    word = 'banana'
    index = vu.lookup_indices([word])
    print(index[0])


# testVu()




test_data = [['apple banana'], ['banana'], ['orange grape']]
# max_len = 500
def etlArrayWithLength(arr, maxLen):
    if len(arr) >= maxLen:
        return arr[:maxLen]
    return arr + [0] * (maxLen - len(arr))


#print(etlArrayWithLength(["abd"], 3))
etl_data = [etlArrayWithLength(words, 2) for words in test_data]
# for words in test_data:
#     etl_data.append(etlArrayWithLength(words, 3))

#print(etl_data)
#torch.tensor(etl_data)


train_data = [['"dick tracy" is one of our"', 1],
                ['arguably this is a  the )', 1],
            ["i don't  just to warn anyone ", 0]]

def get_tokenized(data):
    def tokenizer(text):
        return [item.lower() for item in text.split(" ")]
    return [tokenizer(context) for context, _ in data]

tokenized_data = get_tokenized(train_data)
print(tokenized_data)


# items=[]
# for token_item in tokenized_data:
#     for item in token_item:
#         items.append(item)
#
# print(items)

counter = collections.Counter(_flatten(tokenized_data))
print('counter:------------ ', counter)
count2 = collections.Counter([item for token_item in tokenized_data for item in token_item])
print('counter2-------------', count2)

sortlist  = sorted(counter.items())
order_dict = collections.OrderedDict(sortlist)
print('order dict:', order_dict)

vb = vocab.vocab(count2, min_freq=2)
print('vb~', len(vb))

vocabu = torchtext.vocab.vocab(collections.Counter([item for token_item in tokenized_data for item in token_item]))
print('test', len(vocabu))


#v = torchtext.vocab.vocab(counter)
#print('v--!!:',v)

v1= torchtext.vocab.vocab(order_dict)
print(type(v1), len(v1))