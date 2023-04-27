# by y_dd
# 时间 2023/04/22
import torch

#1.数据准备
x = torch.rand([500, 1])
y = x*0.5 + 0.8
w = torch.rand([1,1],  requires_grad=True, dtype=torch.float32)
b = torch.tensor(0, requires_grad=True, dtype=torch.float32)
#2.预测值准备（此处简单模型直接计算出来）
#y_predict = x * w + b

learningrate=0.001
#3.不断的反向传播,调整
for i in range(2000):

    y_predict = torch.matmul(x,w) + b
    #3.1.计算loss
    loss = (y - y_predict).pow(2).mean()

    if w.grad is not None:
        w.grad.data.zero_()
    if b.grad is not None:
        b.grad.data.zero_()

    #3.2.反向传播，更新参数
    loss.backward();

    w.data = w.data - w.grad * learningrate
    b.data = b.data - b.grad * learningrate
    if (i % 100 == 0):
        print("[w b loss]", w.item(), b.item(), loss.item())

