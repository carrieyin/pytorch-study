import torch
from torch import nn
from torch.nn import MaxPool2d

input = torch.tensor([[1, 2, 0, 3, 1],
                      [0, 1, 2, 3, 1],
                      [1, 2, 1, 0, 0],
                      [5, 2, 3, 1, 1],
                      [2, 1, 0, 1, 1]], dtype=torch.float32)  # 最大池化无法对long数据类型进行实现,将input变成浮点数的tensor数据类型
input = torch.reshape(input, (-1, 1, 5, 5))  # -1表示torch计算batch_size
print(input.shape)


# 搭建神经网络
class Test(nn.Module):
    def __init__(self):
        super(Test, self).__init__()
        self.maxpool1 = MaxPool2d(kernel_size=3, ceil_mode=True)

    def forward(self, input):
        output = self.maxpool1(input)
        return output


# 创建神经网络
test = Test()
output = test(input)
print(output)