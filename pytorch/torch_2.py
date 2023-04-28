# by y_dd
# 时间 2023/04/28
import torch
x = torch.tensor([[1.,2.],[3.,4.]], requires_grad=True)
print("tensor x:", x)
print("grad   x:", x.grad)
y = x * 2
print("tensor y:", y)
print("grad   y:", y.grad)
out = y.mean()
print("tensor out:", out)
print("grad   out:", out.grad)
print(out)

out.backward()
print(x.grad)
print(y.grad)