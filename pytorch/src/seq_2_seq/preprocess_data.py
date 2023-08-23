"""
预处理NMT 法-英 数据集数据

"""
import os
import zipfile
import torch.utils.data as Data
import torch

from pytorch.src.seq_2_seq import config
from pytorch.src.seq_2_seq.config import raw_file_path, raw_zip_path
from pytorch.src.seq_2_seq.vocabulary import Vocabulary


def extract_content():
    # folder_name = os.path.dirname(file_path)
    # file_name = os.path.basename(file_path)

    if not os.path.exists(raw_file_path):
        with zipfile.ZipFile(raw_zip_path, 'r') as zip_ref:
            zip_ref.extractall(os.path.dirname(raw_file_path))

    print('解压缩完成')

    with open(raw_file_path, 'r', encoding='UTF-8') as f:
        content = f.read()
        return content


def download_extract():
    # file_path = r"../../../dataset/nmt/frg-eng.txt"
    # if os.path.exists(file_path):
    #     return extract_content(file_path)
    #
    # data_url = "http://www.manythings.org/anki/fra-eng.zip"
    # zip_path = r"../../../dataset/nmt/frg-eng.zip"
    # zip_file = requests.get(data_url, stream=True, verify=True)
    # with open(zip_path, 'wb') as f:
    #     f.write(zip_file.content)

    url = 'http://www.manythings.org/anki/fra-eng.zip'
    save_path = r"..\..\..\dataset\nmt"

    # 发起GET请求并下载文件
    # response = requests.get(url)

    # 保存文件
    # response = requests.get(url, stream=True)
    # with open(save_path, 'wb') as file:
    #     file.write(response.content)
    #
    # print("文件已下载并保存到", save_path)

def preprocess_nmt(text):
    """预处理“英语－法语”数据集
    数据集中的标点符号与句子分开
    多个空格处理成一个空格
    """
    def no_space(char, prev_char):
        return char in set(',.!?') and prev_char != ' '

    # 使用空格替换不间断空格
    # 使用小写字母替换大写字母
    text = text.replace('\u202f', ' ').replace('\xa0', ' ').lower()
    # 在单词和标点符号之间插入空格
    out = [' ' + char if i > 0 and no_space(char, text[i - 1]) else char
           for i, char in enumerate(text)]
    return ''.join(out)


def tokenize_nmt(text, num_examples=None):
    """词元化“英语－法语”数据数据集
    将输入的每一行按照tab分割成source和target两个列表，其中存放了按照空格分割的单词的列表
    返回source和target两个列表，source中第i个列表的翻译对应于target中第i个列表的翻译
    eg:source[['hello', 'world']..]  target[['bojour'], []]
    """

    source, target = [], []
    for i, line in enumerate(text.split('\n')):
        if num_examples and i > num_examples:
            break
        parts = line.split('\t')
        if len(parts) == 3:
            source.append(parts[0].split(' '))
            target.append(parts[1].split(' '))

    return source, target

def build_array_nmt(lines, vocab, num_steps):
    """将机器翻译的文本序列转换成小批量
    """
    lines = [vocab[l] for l in lines]
    lines = [l + [vocab['<eos>']] for l in lines]
    array = torch.tensor([truncate_pad(
        l, num_steps, vocab['<pad>']) for l in lines])
    valid_len = reduce_sum(
        astype(array != vocab['<pad>'], torch.int32), 1)
    #print('valid len', valid_len.shape, valid_len)
    return array, valid_len

def truncate_pad(line, num_steps, padding_token):
    """截断或填充文本序列
    num_steps为最大长度，line长度小于该长度，pad填充
    """
    if len(line) > num_steps:
        return line[:num_steps]  # 截断
    return line + [padding_token] * (num_steps - len(line))  # 填充


def load_array(data_arrays, batch_size, is_train=True):
    """构造一个数据迭代器"""
    dataset = Data.TensorDataset(*data_arrays)
    return Data.DataLoader(dataset, batch_size, shuffle=is_train)

def load_data_nmt(batch_size, num_steps, num_examples=600):
    """返回翻译数据集的迭代器和词表
    """
    text = preprocess_nmt(extract_content())
    source, target = tokenize_nmt(text, num_examples)
    src_vocab = Vocabulary(source, min_freq=2,
                          reserved_tokens=['<pad>', '<bos>', '<eos>'])
    tgt_vocab = Vocabulary(target, min_freq=2,
                          reserved_tokens=['<pad>', '<bos>', '<eos>'])
    src_array, src_valid_len = build_array_nmt(source, src_vocab, num_steps)
    tgt_array, tgt_valid_len = build_array_nmt(target, tgt_vocab, num_steps)
    data_arrays = (src_array, src_valid_len, tgt_array, tgt_valid_len)
    data_iter = load_array(data_arrays, batch_size)
    return data_iter, src_vocab, tgt_vocab



reduce_sum = lambda x, *args, **kwargs: x.sum(*args, **kwargs)
astype = lambda x, *args, **kwargs: x.type(*args, **kwargs)


if __name__ == '__main__':
    # content = extract_content()
    # data = preprocess_nmt(content)
    # #print(data)
    # source, target = tokenize_nmt(data, 600)
    # print(source)
    # print(target)
    load_data_nmt(config.batch_size, config.num_steps, config.num_examples)
