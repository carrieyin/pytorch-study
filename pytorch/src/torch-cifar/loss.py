# by y_dd
# 时间 2023/07/03
import torch
from torch.nn import L1Loss
from torch.nn import MSELoss
from torch.nn import CrossEntropyLoss

input= torch.tensor([1.,2,3])
output = torch.tensor([1,4,3])

loss = L1Loss()
result = loss(input, output)

loss = MSELoss()
result = loss(input, output)

x = torch.tensor([0.1,0.2,0.3])
y = torch.tensor([1])
x = torch.reshape(x,(1,3))
loss_cross = CrossEntropyLoss()
result_cross = loss_cross(x,y)


def soft_max(data):
    t1=torch.exp(data)#对所有数据进行指数运算
    s=t1.sum(dim=1) #按行求和
    shape=data.size()
    m=shape[0]#获取行数
    n=shape[1]
    for i in range(m):
          t1[i]=t1[i]/s[i]
    return t1

def cross_entropyloss(input,target):
    shape=data.size()
    m=shape[0]#获取行数
    output=-torch.log(input[range(m),target.flatten()]).sum()/m
    return output


#自己编写
data=torch.tensor([[1,  0.2,  2], [-1,  0.5,  3]])
t1=soft_max(data)#将预测数据转换为概率！！！

print(t1)
#此处为重点
t2=cross_entropyloss(t1, torch.tensor([1,2]))
print(t2)
torch.log(input[range(m),target.flatten()]).sum()/m
output = -torch.log(t2[range(2), torch.tensor([1,2]).flatten()]).sum() / 2
print(output)
#PyTorch中的原函数
#crossentropyloss=CrossEntropyLoss()
#t3=crossentropyloss(data,torch.tensor([1,2]))
#print(t3)

