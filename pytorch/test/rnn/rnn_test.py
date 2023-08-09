import torch
from torch.nn import LSTM

lstm = LSTM(10, 20, 2)
inputs = torch.randn(5, 3, 10)
h0 = torch.randn(2, 3, 20)
c0 = torch.randn(2, 3, 20)
output, (hn, cn) = lstm(inputs, (h0, c0))
print(inputs.shape,output.shape, hn.shape, cn.shape)

a = output[-1,:,]
b = hn[-1,:,:]

print(a == b)


