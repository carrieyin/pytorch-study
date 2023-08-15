import collections
import os.path
import random
import re

import torch
import tarfile
from torch.nn import RNN
from torchtext import vocab
from tkinter import _flatten
from tqdm import tqdm

# 解压缩数据集
dataPath = "..\\..\\..\\dataset\\"
imdb_zip_path = os.path.join(dataPath, "aclImdb")
# 如果不存在则解压压缩包
if not os.path.exists(imdb_zip_path):
    print("解压缩")
    with tarfile.open(os.path.join(dataPath, "aclImdb_v1.tar.gz")) as f:
        f.extractall(dataPath)


# 读取数据集数据
def read_imdb(datafolder ='train', dataroot=imdb_zip_path):
    data=[]
    for label in ['pos', 'neg']:
        filepath = os.path.join(imdb_zip_path, datafolder, label)
        for file in tqdm(os.listdir(filepath)):
            with open(os.path.join(filepath,file), 'rb') as f:
                content = f.read().decode('utf-8').replace('\n', ' ').lower()
                data.append([content, 1 if label == 'pos' else 0])
    random.shuffle(data)
    return data


# 预处理数据 -1
# 数据分词，使用最简单的空格分词
def get_tokenized(data):
    def tokenizer(text):
        filters = ['!', '"', '#', '$', '%', '&', '\(', '\)', '\*', '\+', ',', '-', '\.', '/', ':', ';', '<', '=', '>',
                    '\?', '@', '\[', '\\', '\]', '^', '_', '`', '\{', '\|', '\}', '~', '\t', '\n', '\x97', '\x96', '”', '“', ]
        text = re.sub("<.*?>", " ", text, flags=re.S)
        text = re.sub("|".join(filters), " ", text, flags=re.S)
        return [i.strip().lower() for i in text.split()]
    return [tokenizer(context) for context, _ in data]


# 创建分词后的词典
def get_vocab(data):
    counter = collections.Counter(_flatten(data))
    return vocab.vocab(counter)


# pad数据，方便bath操作
def pad(x, max_len):
    if len(x) > max_len:
        return x[:max_len]
    else:
        return x + [0] * (max_len - len(x))


# 数据预处理
def preprocess(data):
    max_len = 500
    # 1.获取分词数据形式[[评论1分词1,评论1分词n], [评论i分词1， 评论i分词m].....]
    tokenized_data = get_tokenized(data)
    # print(tokenized_data)

    # 2. 获取分词词汇表(vocab类)
    vo = get_vocab(tokenized_data)
    # print('len vo ', len(vo))

    # 3. 更新分词形式，将分词表中的单词替换为词汇表中下标，并将每条评论填充至500个分词的长度
    # 形式[[评论1下标...评论1分词500下标]...[评论i下标...评论i分词500下标]]
    pad_token_words = [pad([vo[item] for item in items], max_len) for items in tokenized_data]

    # 4. 转换为tensor
    features = torch.tensor(pad_token_words)
    labels = torch.tensor([label for _, label in data])
    return features, labels


if __name__ == '__main__':

    train_data = [['"dick tracy" is one of our"', 1],
    ['arguably this is a  the )', 1],
    ["i don't  just to warn anyone ", 0]]
    # train_data = read_imdb('train')
    preprocess(train_data)