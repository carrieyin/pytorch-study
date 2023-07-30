import torch
import torchtext
import collections

from torchtext import vocab
# def tokenizer(content):
#     return [item for item in content.split(' ')]


def testVu(train_data):
    # 构建词汇表
    tokenizer = lambda x: x.split()  # 分词函数，这里简单地按空格划分单词
    vu = vocab.build_vocab_from_iterator(map(tokenizer, train_data))
    # 使用Vocab对象将单词映射为索引
    word = 'banana'
    index = vu.lookup_indices([word])
    print(index[0])


if __name__ == '__main__':
    train_data = ['apple banana', 'banana', 'orange grape']
    testVu(train_data)