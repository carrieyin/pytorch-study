# by y_dd
# 时间 2023/04/22
import torch


#1.数据准备
x = torch.rand([500, 1])
y = x*0.5 + 0.8

w = torch.tensor([1,1], requires_grad=True)
b=torch.tensor(0, requires_grad=True,dtype=torch.float32)

learningrate=0.001

for i in range(500):

    # 2.预测值准备（此处简单模型直接计算出来）
    y_predict = torch.matmul(x, w) + b

    if w.grad is not None:
        w.grad.data.zero()
    if b.grad is not None:
        b.grad.data.zero()
    #3.计算损失
    loss = torch.mean((y - y_predict).pow(2))
    # 4.反向传播，更新参数
    loss.backward()
    #更新梯度
    w.data = w.data - learningrate * w.grad
    b.data = b.data - learningrate * b.grad

    print("w, b, loss", w.item(), b.item())