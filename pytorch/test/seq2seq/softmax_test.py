import torch
from torch import nn

loss_func = nn.CrossEntropyLoss(reduction="none")
pre_data = torch.tensor([[0.8, 0.5, 0.2, 0.5],
                         [0.2, 0.9, 0.3, 0.2],
                         [0.4, 0.3, 0.7, 0.1],
                         [0.1, 0.2, 0.4, 0.8]], dtype=torch.float)
tgt_index_data = torch.tensor([0,
                               1,
                               2,
                               3], dtype=torch.long)
tgt_onehot_data = torch.tensor([[1, 0, 0, 0],
                                [0, 1, 0, 0],
                                [0, 0, 1, 0],
                                [0, 0, 0, 1]], dtype=torch.float)


def simple_single_one():
    pre = pre_data[0]
    tgt_index = tgt_index_data[0]
    tgt_onehot = tgt_onehot_data[0]

    print(torch.softmax(pre, dim=-1))
    print(-torch.sum(torch.mul(torch.log(torch.softmax(pre, dim=-1)), tgt_onehot), dim=-1))
    print(loss_func(pre, tgt_index))
    print(loss_func(pre, tgt_onehot))


if __name__ == '__main__':
    simple_single_one()