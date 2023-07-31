import torch
from torch import nn
from torch.nn import ReLU

input = torch.tensor([[1, -0.5],
                      [-1, 3]])
#input = torch.reshape(input, (-1, 1, 2, 2))  # input必须要指定batch_size，-1表示batch_size自己算，1表示是1维的
print(input.shape)  # torch.Size([1, 1, 2, 2])


# 搭建神经网络
class Test(nn.Module):
    def __init__(self):
        super(Test, self).__init__()
        self.relu1 = ReLU()  # inplace默认为False

    def forward(self, input):
        output = self.relu1(input)
        return output


# 创建网络
test = Test()
output = test(input)
print(output)