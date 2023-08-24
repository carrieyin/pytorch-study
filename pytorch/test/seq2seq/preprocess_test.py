import torch

from pytorch.src.seq_2_seq import config
from pytorch.src.seq_2_seq.config import raw_file_path, raw_zip_path
from pytorch.src.seq_2_seq.preprocess_data import download_extract, extract_content, preprocess_nmt, tokenize_nmt, \
    astype, reduce_sum, load_data_nmt


# a = torch.tensor(10)
# print(a.type)

# array = torch.tensor([[2,6,7,1,1], [6,7,1,1,1]])
# c = (array != 1)
# print(c)
#
# m = astype(c, torch.int32)
# print(m)
# n = c.type(torch.int32)
# print(n)
#
# d = reduce_sum(c, 1)
# print(d)


def test_extract():
    content = extract_content()
    data = preprocess_nmt(content)
    # print(data)
    token_data = tokenize_nmt(data, 600)
    print(token_data)


def test_load():
    load_data_nmt(config.batch_size, config.num_steps, config.num_examples)


def init_state():
    output, state = ([1, 2, 3], [4, 5, 6])
    return output, state


def test_shape():
    a = torch.tensor(([1, 2, 3], [4, 5, 6]))
    print(a.shape[1])

def test_repeat():
    a = torch.tensor(([[1, 2, 3, 4], [4, 5, 6, 9], [1, 2, 3, 4]]))
    b = torch.tensor(([[1, 2]]))
    print(a.shape, b.shape)

    c = b.repeat(a.shape[0], 1, 1)
    print( c.shape)


def test_concat():
    batch_size, num_steps, embed_size = 2, 3, 4
    a = torch.zeros((batch_size, num_steps, embed_size), dtype=torch.long)
    b = torch.tensor(([[1, 2]]))
    c = torch.concat(a, b)
    print(c)

if __name__ == '__main__':
    #test_load()
    # sts = init_state()
    # print(sts[0])
    # test_shape()
    test_repeat()
    test_concat()

